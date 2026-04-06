"""
graph_builder.py
----------------
Builds the dependency graph for tselect.

Three-phase build:
  Phase 0: AST parsing of SOURCE files → source_reverse_graph
             source_file → [source files that import it]
             Enables transitive dependency resolution.

  Phase 1: AST parsing of TEST files → THREE reverse graphs:
             (a) file_reverse_graph:     source file  → test files that import it
             (b) function_reverse_graph: source symbol → test METHODS that reference it
             (c) file_identifiers:       source file  → all identifiers used in it

  Phase 2: pytest --collect-only in batches → test inventory

  Phase 3: compute dynamic fanout threshold from distribution gap

Changes from 2.1 → 3.0:
  - function_reverse_graph now maps to test METHODS not test FILES
    "sgd.py::SGD" → ["test_optim.py::TestOptimCPU::test_sgd_momentum", ...]
  - file_identifiers: every source file's identifier set
    Used by graph_selector for identifier overlap BFS stopping condition
  - fanout_threshold: computed from actual repo distribution gap

Changes from 3.0 → 3.1 (tree-sitter upgrade):
  - _extract_public_symbols: now uses tree-sitter via fn_diff.get_all_symbols()
    Supports .py AND .cpp/.cu/.h — no more silent skips for C++ source files
  - _extract_all_identifiers: now uses tree-sitter via fn_diff.get_all_identifiers()
    C++ identifier walk via tree-sitter CST instead of ast-only
  - _build_source_reverse_graph: handles #include for C/C++ files in addition to
    Python import statements
  - Test file parsing (_extract_symbol_references, _extract_method_level_references)
    stays ast-based — PyTorch test files are always .py
"""

import ast
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from tselect.core.fn_diff import (
    get_all_symbols,
    get_all_identifiers,
    get_dunder_all,
    get_decorator_registry,
    get_call_sites,
    classify_change,
)

SUPPORTED_LANGUAGES   = {"python"}
COMING_SOON_LANGUAGES = {"java", "javascript", "typescript", "go", "cpp"}
DEFAULT_HIGH_FANOUT_THRESHOLD = 30

CPP_EXTENSIONS = {'.cpp', '.cu', '.cuh', '.h', '.hpp', '.cc', '.c'}
PY_EXTENSIONS  = {'.py'}


class UnsupportedLanguageError(Exception):
    pass


class GraphBuilder:
    def __init__(self, layout, config: dict = None):
        self.layout       = layout
        self.repo_root    = layout.repo_root
        self.language     = layout.language
        self.source_files = layout.source_files
        self.test_files   = layout.test_files
        self.config       = config or {}
        self.batch_size   = (
            self.config.get("graph", {}).get("collect_batch_size", 50)
            or getattr(layout, "collect_batch_size", 50)
        )
        self._validate_language()

    def _validate_language(self):
        if self.language in SUPPORTED_LANGUAGES:
            return
        if self.language in COMING_SOON_LANGUAGES:
            raise UnsupportedLanguageError(
                f"\n  Language '{self.language}' is not yet supported.\n"
                f"  Supported: python\n"
                f"  Coming soon: java, javascript, typescript, go, cpp\n"
            )
        raise UnsupportedLanguageError(
            f"\n  Unknown language '{self.language}'.\n"
            f"  Check 'repo.language' in tselect.yaml.\n"
        )

    # ─────────────────────────────────────────────
    # SOURCE: extract public symbols  (tree-sitter)
    # ─────────────────────────────────────────────

    def _extract_public_symbols(self, file_path: Path) -> set:
        """
        All top-level names from a source file.
        Used for conservative fallback when test imports entire module.

        Now delegates to fn_diff.get_all_symbols() which uses tree-sitter,
        supporting .py AND .cpp/.cu/.h source files.

        Falls back to ast for .py if tree-sitter is not installed.
        """
        return get_all_symbols(file_path)

    # ─────────────────────────────────────────────
    # SOURCE: extract all identifiers  (tree-sitter)
    # ─────────────────────────────────────────────

    def _extract_all_identifiers(self, file_path: Path) -> set:
        """
        Extract every identifier referenced in a source file.
        Used to build file_identifiers for BFS stopping condition.

        Now delegates to fn_diff.get_all_identifiers():
          .py  → ast walk (Name + Attribute nodes)
          .cpp/.cu/.h → tree-sitter identifier node walk

        Example for torch/optim/__init__.py:
            {"SGD", "Adam", "RMSprop", "step", "zero_grad", "momentum", ...}

        Example for torch/csrc/jit/runtime/interpreter.cpp:
            {"InterpreterState", "run", "Frame", "push", "pop", ...}
        """
        return get_all_identifiers(file_path)

    # ─────────────────────────────────────────────
    # DYNAMIC FANOUT THRESHOLD
    # ─────────────────────────────────────────────

    def _compute_fanout_threshold(self, file_reverse_graph: dict) -> int:
        """
        Compute the fanout threshold from the actual repo's distribution.

        Finds the largest gap between consecutive fanout values.
        Threshold = lower bound of that gap.

        PyTorch example:
            fanouts: ..., 28, 29, 30 ... 73, 116 ... 1075
            largest gap: 116 → 1075 (gap of 959)
            threshold = 116
        """
        fanouts = sorted(
            len(v) for v in file_reverse_graph.values() if len(v) > 0
        )
        if len(fanouts) < 2:
            return DEFAULT_HIGH_FANOUT_THRESHOLD

        largest_gap = 0
        threshold   = DEFAULT_HIGH_FANOUT_THRESHOLD

        for i in range(len(fanouts) - 1):
            gap = fanouts[i + 1] - fanouts[i]
            if gap > largest_gap:
                largest_gap = gap
                threshold   = fanouts[i]

        return threshold

    # ─────────────────────────────────────────────
    # PHASE 0: Source → Source reverse graph
    # ─────────────────────────────────────────────

    def _build_source_reverse_graph(self, module_map: dict) -> dict:
        """
        Parse all SOURCE files and build a reverse import graph:
            source_file → [source files that import it]

        Python files: ast-based import parsing (ImportFrom, Import)
        C/C++ files:  regex-based #include parsing

        Example:
            torch/optim/sgd.py → [torch/optim/__init__.py, torch/optim/optimizer.py]
            torch/csrc/jit/ir.h → [torch/csrc/jit/runtime/interpreter.cpp]
        """
        forward = defaultdict(set)

        for src in self.source_files:
            rel    = str(src.relative_to(self.repo_root))
            suffix = src.suffix.lower()

            if suffix in PY_EXTENSIONS:
                self._parse_py_imports(src, rel, module_map, forward)
            elif suffix in CPP_EXTENSIONS:
                self._parse_cpp_includes(src, rel, forward)

        reverse = defaultdict(set)
        for importer, importees in forward.items():
            for importee in importees:
                reverse[importee].add(importer)

        return {k: sorted(v) for k, v in reverse.items()}

    def _parse_py_imports(
        self, src: Path, rel: str, module_map: dict, forward: dict
    ) -> None:
        """Parse Python import statements and populate forward graph."""
        try:
            source = src.read_text(encoding="utf-8", errors="ignore")
            tree   = ast.parse(source)
        except Exception:
            return

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported = module_map.get(node.module)
                if imported and imported != rel:
                    forward[rel].add(imported)

                for alias in node.names:
                    child_mod = f"{node.module}.{alias.name}"
                    child_src = module_map.get(child_mod)
                    if child_src and child_src != rel:
                        forward[rel].add(child_src)

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported = module_map.get(alias.name)
                    if imported and imported != rel:
                        forward[rel].add(imported)

    def _parse_cpp_includes(self, src: Path, rel: str, forward: dict) -> None:
        """
        Parse C/C++ #include directives to build the source import graph.

        Handles:
            #include "torch/csrc/jit/ir.h"      → relative/project includes
            #include <ATen/core/Tensor.h>        → angle-bracket includes

        Skips system headers (no path separator or known system prefixes).
        """
        _SYSTEM_PREFIXES = ('std', 'c++', 'bits/', 'sys/', 'linux/')
        include_re = re.compile(r'#\s*include\s*[<"]([^>"]+)[>"]')

        try:
            source = src.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return

        repo_files = {
            str(f.relative_to(self.repo_root)): str(f.relative_to(self.repo_root))
            for f in self.source_files
            if f.suffix.lower() in CPP_EXTENSIONS
        }

        for match in include_re.finditer(source):
            include_path = match.group(1)

            # skip pure system headers
            if not any(include_path.startswith(p) for p in _SYSTEM_PREFIXES):
                # direct match
                if include_path in repo_files:
                    if repo_files[include_path] != rel:
                        forward[rel].add(repo_files[include_path])
                else:
                    # try matching by filename suffix
                    # e.g. "ir.h" might match "torch/csrc/jit/ir.h"
                    fname = Path(include_path).name
                    for repo_rel in repo_files:
                        if Path(repo_rel).name == fname and repo_rel != rel:
                            forward[rel].add(repo_rel)
                            break  # take first match only

    # ─────────────────────────────────────────────
    # TEST: extract symbol-level references (ast — tests are .py)
    # ─────────────────────────────────────────────

    def _extract_symbol_references(self, test_path: Path, module_map: dict) -> dict:
        """
        Parse a test file and return which symbols from which source files
        it directly references. Used for file_reverse_graph.

        Returns:
            {
                "torch/_inductor/scheduler.py": {"Scheduler", "_fuse_nodes"},
                "torch/_inductor/lowering.py":  {"make_fallback"},
            }
        """
        references = defaultdict(set)

        try:
            source = test_path.read_text(encoding="utf-8", errors="ignore")
            tree   = ast.parse(source)
        except Exception:
            return references

        alias_map = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                src_file = module_map.get(node.module)

                if src_file:
                    for alias in node.names:
                        if alias.name == "*":
                            src_path = self.repo_root / src_file
                            # feature 6: use __all__ if defined, else all symbols
                            explicit = get_dunder_all(src_path)
                            references[src_file].update(
                                explicit if explicit is not None
                                else self._extract_public_symbols(src_path)
                            )
                        else:
                            references[src_file].add(alias.name)
                            local = alias.asname or alias.name
                            alias_map[local] = (src_file, alias.name)

                for alias in node.names:
                    child_mod = f"{node.module}.{alias.name}"
                    child_src = module_map.get(child_mod)
                    if child_src:
                        src_path = self.repo_root / child_src
                        references[child_src].update(
                            self._extract_public_symbols(src_path)
                        )
                        local = alias.asname or alias.name
                        alias_map[local] = (child_src, None)

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    src_file = module_map.get(alias.name)
                    if src_file:
                        local = alias.asname or alias.name.split(".")[-1]
                        alias_map[local] = (src_file, None)

        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                obj = node.value.id
                if obj in alias_map:
                    src_file, _ = alias_map[obj]
                    references[src_file].add(node.attr)

        for local, (src_file, _) in alias_map.items():
            if src_file and src_file not in references:
                src_path = self.repo_root / src_file
                references[src_file].update(
                    self._extract_public_symbols(src_path)
                )

        return dict(references)

    # ─────────────────────────────────────────────
    # TEST: extract method-level references (ast — tests are .py)
    # ─────────────────────────────────────────────

    def _extract_method_level_references(
        self, test_path: Path, module_map: dict,
        call_sites: dict = None, decorator_registry: dict = None
    ) -> dict:
        """
        Parse a test file and return which symbols each TEST METHOD references.

        Path A — direct import intersection (existing):
            method imports SGD AND calls SGD → edge

        Path B — decorator registry matching (feature 15, NEW):
            method calls "torch.ops.aten.max_pool2d_with_indices_backward"
            suffix "aten.max_pool2d_with_indices_backward" matches decorator_registry
            → edge: lowering.py::max_pool2d_with_indices_backward → this method
        """
        method_references = {}
        call_sites        = call_sites or {}
        decorator_registry = decorator_registry or {}

        try:
            source = test_path.read_text(encoding="utf-8", errors="ignore")
            tree   = ast.parse(source)
        except Exception:
            return method_references

        # step 1: file-level alias map + feature 7: skip TYPE_CHECKING imports
        alias_map    = {}
        file_symbols = defaultdict(set)

        for node in ast.walk(tree):
            # feature 7: skip TYPE_CHECKING blocks
            if isinstance(node, ast.If):
                test_val = node.test
                if isinstance(test_val, ast.Name) and test_val.id == 'TYPE_CHECKING':
                    continue
                if isinstance(test_val, ast.Attribute) and test_val.attr == 'TYPE_CHECKING':
                    continue

            if isinstance(node, ast.ImportFrom) and node.module:
                src_file = module_map.get(node.module)
                if src_file:
                    for alias in node.names:
                        if alias.name != "*":
                            local = alias.asname or alias.name
                            alias_map[local] = (src_file, alias.name)
                            file_symbols[src_file].add(alias.name)

                for alias in node.names:
                    child_mod = f"{node.module}.{alias.name}"
                    child_src = module_map.get(child_mod)
                    if child_src:
                        local = alias.asname or alias.name
                        alias_map[local] = (child_src, None)

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    src_file = module_map.get(alias.name)
                    if src_file:
                        local = alias.asname or alias.name.split(".")[-1]
                        alias_map[local] = (src_file, None)

        # step 2: per-method symbol walk
        for class_node in ast.walk(tree):
            if not isinstance(class_node, ast.ClassDef):
                continue

            for method_node in class_node.body:
                if not isinstance(method_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if not method_node.name.startswith("test_"):
                    continue

                method_key   = f"{class_node.name}::{method_node.name}"
                cs_key       = f"{class_node.name}.{method_node.name}"
                actual_calls = call_sites.get(cs_key)
                refs         = defaultdict(set)

                # Path A: direct import intersection
                if alias_map:
                    for node in ast.walk(method_node):
                        if isinstance(node, ast.Name):
                            name = node.id
                            if name in alias_map:
                                if actual_calls is None or name in actual_calls:
                                    src_file, sym = alias_map[name]
                                    refs[src_file].add(sym or name)

                        elif isinstance(node, ast.Attribute):
                            attr = node.attr
                            if isinstance(node.value, ast.Name):
                                obj = node.value.id
                                if obj in alias_map:
                                    if actual_calls is None or attr in actual_calls:
                                        src_file, _ = alias_map[obj]
                                        refs[src_file].add(attr)

                # Path B: decorator registry matching (feature 15)
                # check all call chains in this method against decorator_registry
                if decorator_registry and actual_calls:
                    for chain in actual_calls:
                        # chain could be "aten.add", "ops.aten.add", etc.
                        if chain in decorator_registry:
                            # decorator_registry value is "source_file::fn_name"
                            mapped = decorator_registry[chain]
                            if '::' in mapped:
                                src_file, fn_name = mapped.split('::', 1)
                                refs[src_file].add(fn_name)

                if refs:
                    method_references[method_key] = dict(refs)

        return method_references

    # ─────────────────────────────────────────────
    # PHASE 1: Build reverse graphs
    # ─────────────────────────────────────────────

    def _build_reverse_graphs(self, module_map: dict) -> tuple:
        """
        Build three graphs simultaneously.

        file_reverse_graph:
            "torch/_inductor/scheduler.py" → ["test/inductor/test_scheduler.py", ...]

        function_reverse_graph (maps to test METHODS):
            "torch/_inductor/scheduler.py::_fuse_nodes" →
                ["test/inductor/test_scheduler.py::TestSchedulerCPU::test_fuse_nodes"]

        file_identifiers (tree-sitter powered for .cpp/.cu as well as .py):
            "torch/optim/__init__.py"              → ["SGD", "Adam", "step", ...]
            "torch/csrc/jit/runtime/interpreter.cpp" → ["InterpreterState", "run", ...]
        """
        file_reverse     = defaultdict(set)
        function_reverse = defaultdict(set)
        file_identifiers = {}
        decorator_registry = {}  # feature 8: arg → function name across all source files

        # feature 8: build decorator registry from all source files first
        print("  Building decorator registry (tree-sitter) ...")
        for src in self.source_files:
            rel = str(src.relative_to(self.repo_root))
            reg = get_decorator_registry(src)
            for arg, fn_name in reg.items():
                # store as "source_file::fn_name" so graph_selector can look it up
                decorator_registry[arg] = f"{rel}::{fn_name}"

        total = len(self.test_files)
        for idx, test in enumerate(self.test_files):
            rel_test = str(test.relative_to(self.repo_root))

            file_refs = self._extract_symbol_references(test, module_map)
            for src_file in file_refs:
                file_reverse[src_file].add(rel_test)

            # feature 14+15: use call sites + decorator registry for precise edges
            call_sites  = get_call_sites(test)
            method_refs = self._extract_method_level_references(
                test, module_map, call_sites, decorator_registry
            )
            for method_key, src_refs in method_refs.items():
                full_method_id = f"{rel_test}::{method_key}"
                for src_file, symbols in src_refs.items():
                    for sym in symbols:
                        key = f"{src_file}::{sym}"
                        function_reverse[key].add(full_method_id)

            if (idx + 1) % 100 == 0:
                print(f"    [{idx+1}/{total} test files processed]")

        # file_identifiers — now tree-sitter powered for .cpp/.cu too
        print("  Building file identifier index (tree-sitter) ...")
        for src in self.source_files:
            rel = str(src.relative_to(self.repo_root))
            ids = self._extract_all_identifiers(src)
            if ids:
                file_identifiers[rel] = sorted(ids)

        return (
            {k: sorted(v) for k, v in file_reverse.items()},
            {k: sorted(v) for k, v in function_reverse.items()},
            file_identifiers,
            decorator_registry,
        )

    # ─────────────────────────────────────────────
    # PHASE 2: pytest --collect-only inventory
    # ─────────────────────────────────────────────

    def _collect_batch(self, batch: list) -> str:
        cmd = [
            sys.executable, "-m", "pytest",
            "--collect-only", "-q", "--no-header",
            "--continue-on-collection-errors",
        ] + batch

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=str(self.repo_root), timeout=30,
            )
            return result.stdout
        except Exception:
            return ""

    def _build_test_inventory(self, candidate_test_files: list) -> dict:
        if not candidate_test_files:
            return {}

        batches = [
            candidate_test_files[i:i + self.batch_size]
            for i in range(0, len(candidate_test_files), self.batch_size)
        ]

        print(f"    Collecting {len(candidate_test_files)} test files "
              f"in {len(batches)} batches of {self.batch_size}...")

        inventory  = defaultdict(lambda: defaultdict(lambda: {"node_ids": [], "test_count": 0}))
        successful = 0

        for i, batch in enumerate(batches):
            stdout = self._collect_batch(batch)
            if not stdout:
                continue

            for line in stdout.splitlines():
                line = line.strip()
                if "::" not in line or line.startswith(("ERROR", "Warning", "FAILED")):
                    continue
                parts = line.split("::")
                if len(parts) < 3:
                    continue

                test_file  = parts[0]
                class_name = parts[1].split("[")[0]
                method     = parts[2].split("[")[0]
                node_id    = f"{test_file}::{class_name}::{method}"

                inventory[test_file][class_name]["node_ids"].append(node_id)
                inventory[test_file][class_name]["test_count"] += 1
                successful += 1

            if (i + 1) % 5 == 0 or (i + 1) == len(batches):
                print(f"    [{i+1}/{len(batches)} batches]  "
                      f"{successful} test methods collected")

        if successful == 0:
            print("    Falling back to AST inventory")
            return self._build_ast_inventory(candidate_test_files)

        return {
            tf: {cls: dict(data) for cls, data in classes.items()}
            for tf, classes in inventory.items()
        }

    def _build_ast_inventory(self, test_files: list) -> dict:
        inventory = {}
        for test_path_str in test_files:
            test_path = self.repo_root / test_path_str
            if not test_path.exists():
                continue
            try:
                source = test_path.read_text(encoding="utf-8", errors="ignore")
                tree   = ast.parse(source)
            except Exception:
                continue

            classes = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    is_test = (
                        node.name.startswith("Test") or
                        node.name.endswith("Test") or
                        node.name.endswith("Tests")
                    )
                    if is_test:
                        methods = [
                            m.name for m in node.body
                            if isinstance(m, ast.FunctionDef)
                            and m.name.startswith("test_")
                        ]
                        if methods:
                            classes[node.name] = {
                                "node_ids": [
                                    f"{test_path_str}::{node.name}::{m}"
                                    for m in methods
                                ],
                                "test_count": len(methods),
                            }
            if classes:
                inventory[test_path_str] = classes

        return inventory

    # ─────────────────────────────────────────────
    # BUILD
    # ─────────────────────────────────────────────

    def build(self) -> dict:
        # module map (shared across all phases)
        module_map = {}
        for src in self.source_files:
            rel         = src.relative_to(self.repo_root)
            module_name = ".".join(rel.with_suffix("").parts)
            module_map[module_name] = str(rel)

            if src.name == "__init__.py":
                package_name = ".".join(src.parent.relative_to(self.repo_root).parts)
                module_map[package_name] = str(rel)

        # Phase 0: source → source reverse graph
        print("  Phase 0: Building source import graph (Python + C++ includes)...")
        t0 = time.time()
        source_reverse_graph = self._build_source_reverse_graph(module_map)
        print(f"    Done in {time.time() - t0:.2f}s  —  "
              f"{len(source_reverse_graph)} source files have dependents")

        # Phase 1: test → source reverse graphs + file identifiers (tree-sitter)
        print("  Phase 1: Building file + function graph + identifier index (tree-sitter)...")
        t1 = time.time()
        file_reverse, function_reverse, file_identifiers, decorator_registry = self._build_reverse_graphs(module_map)

        elapsed1      = time.time() - t1
        total_f_edges = sum(len(v) for v in file_reverse.values())
        total_s_edges = sum(len(v) for v in function_reverse.values())
        avg_file      = total_f_edges / max(len(file_reverse), 1)
        avg_sym       = total_s_edges / max(len(function_reverse), 1)

        print(f"    Done in {elapsed1:.2f}s")
        print(f"    File-level graph:     {len(file_reverse)} source files  "
              f"(avg {avg_file:.1f} test files each)")
        print(f"    Function-level graph: {len(function_reverse)} symbols  "
              f"(avg {avg_sym:.1f} test methods each)")
        print(f"    File identifiers:     {len(file_identifiers)} source files indexed")

        # Phase 2: test inventory
        print("  Phase 2: Building test inventory via pytest --collect-only...")
        t2 = time.time()

        all_candidate_tests = sorted(set(
            tf for tests in file_reverse.values() for tf in tests
        ))
        test_inventory = self._build_test_inventory(all_candidate_tests)

        print(f"    Done in {time.time() - t2:.2f}s — "
              f"{len(test_inventory)} test files indexed")

        # Phase 3: compute dynamic fanout threshold
        fanout_threshold = self._compute_fanout_threshold(file_reverse)
        print(f"  Phase 3: Dynamic fanout threshold = {fanout_threshold} "
              f"(computed from distribution gap)")

        return {
            "schema_version":         "3.2",
            "language":               self.language,
            "source_reverse_graph":   source_reverse_graph,
            "full_reverse_graph":     file_reverse,
            "function_reverse_graph": function_reverse,
            "file_identifiers":       file_identifiers,
            "decorator_registry":     decorator_registry,
            "fanout_threshold":       fanout_threshold,
            "test_inventory":         test_inventory,
            "built_at":               time.time(),
        }

    def save(self, graph_data: dict) -> Path:
        graph_dir  = self.repo_root / ".graph" / "tselect"
        graph_dir.mkdir(parents=True, exist_ok=True)
        graph_path = graph_dir / "dependency_graph.json"
        with open(graph_path, "w") as f:
            json.dump(graph_data, f, indent=2)
        return graph_path