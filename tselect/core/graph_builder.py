"""
graph_builder.py
----------------
Builds the dependency graph for tselect.

Three-phase build:
  Phase 0: AST parsing of SOURCE files → source_reverse_graph
             source_file → [source files that import it]
             Enables transitive dependency resolution.

  Phase 1: AST parsing of TEST files → TWO reverse graphs:
             (a) file_reverse_graph:     source file  → test files that import it
             (b) function_reverse_graph: source symbol → test files that reference it

  Phase 2: pytest --collect-only in batches → test inventory

The function_reverse_graph enables function-level selection:
  "_fuse_nodes() changed" → 2 test files  (instead of all 7 that import scheduler.py)

The source_reverse_graph enables transitive selection:
  "sgd.py changed" → optim/__init__.py imports it → test_optim.py imports __init__
  → test_optim.py selected even though it never directly imports sgd.py

How function references are detected — three import patterns:

  Pattern A (most precise) — named import:
      from torch._inductor.scheduler import Scheduler, _fuse_nodes
      → records: scheduler.py::Scheduler, scheduler.py::_fuse_nodes

  Pattern B — module alias + attribute access:
      import torch._inductor.scheduler as sched
      ...sched.try_fusions(...)
      → records: scheduler.py::try_fusions

  Pattern C (least precise) — bare module import:
      import torch._inductor.scheduler
      → records ALL public symbols of scheduler.py (conservative fallback)
"""

import ast
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


SUPPORTED_LANGUAGES   = {"python"}
COMING_SOON_LANGUAGES = {"java", "javascript", "typescript", "go", "cpp"}


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
                f"\n  ❌ Language '{self.language}' is not yet supported.\n"
                f"  Supported: python\n"
                f"  Coming soon: java, javascript, typescript, go, cpp\n"
            )
        raise UnsupportedLanguageError(
            f"\n  ❌ Unknown language '{self.language}'.\n"
            f"  Check 'repo.language' in tselect.yaml.\n"
        )

    # ─────────────────────────────────────────────
    # SOURCE: extract public symbols
    # ─────────────────────────────────────────────

    def _extract_public_symbols(self, file_path: Path) -> set:
        """
        All top-level names from a source file.
        Used for conservative fallback when test imports entire module.
        """
        symbols = set()
        try:
            source = file_path.read_text(encoding="utf-8", errors="ignore")
            tree   = ast.parse(source)
        except Exception:
            return symbols

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                symbols.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        symbols.add(target.id)

        return symbols

    # ─────────────────────────────────────────────
    # PHASE 0: Source → Source reverse graph
    # ─────────────────────────────────────────────

    def _build_source_reverse_graph(self, module_map: dict) -> dict:
        """
        Parse all SOURCE files and build a reverse import graph:
            source_file → [source files that import it]

        Example:
            torch/optim/sgd.py → [torch/optim/__init__.py, torch/optim/optimizer.py]

        This enables BFS transitive expansion in graph_selector:
            sgd.py changed
                → __init__.py imports sgd.py
                    → test_optim.py imports __init__.py
                        → test_optim.py selected ✅
        """
        # forward: src_file → set of src_files it imports
        forward = defaultdict(set)

        for src in self.source_files:
            rel = str(src.relative_to(self.repo_root))

            try:
                source = src.read_text(encoding="utf-8", errors="ignore")
                tree   = ast.parse(source)
            except Exception:
                continue

            for node in ast.walk(tree):
                # from module import name
                if isinstance(node, ast.ImportFrom) and node.module:
                    imported = module_map.get(node.module)
                    if imported and imported != rel:
                        forward[rel].add(imported)

                    # also: from torch.optim import sgd  (submodule import)
                    for alias in node.names:
                        child_mod = f"{node.module}.{alias.name}"
                        child_src = module_map.get(child_mod)
                        if child_src and child_src != rel:
                            forward[rel].add(child_src)

                # import module
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imported = module_map.get(alias.name)
                        if imported and imported != rel:
                            forward[rel].add(imported)

        # reverse: src_file → [src_files that import IT]
        reverse = defaultdict(set)
        for importer, importees in forward.items():
            for importee in importees:
                reverse[importee].add(importer)

        return {k: sorted(v) for k, v in reverse.items()}

    # ─────────────────────────────────────────────
    # TEST: extract symbol-level references
    # ─────────────────────────────────────────────

    def _extract_symbol_references(self, test_path: Path, module_map: dict) -> dict:
        """
        Parse a test file and return which symbols from which source files
        it directly references.

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

        # alias_map: local_name → (src_file, specific_sym_or_None)
        alias_map = {}

        for node in ast.walk(tree):

            # Pattern A: from module import name1, name2
            if isinstance(node, ast.ImportFrom) and node.module:
                src_file = module_map.get(node.module)

                if src_file:
                    for alias in node.names:
                        if alias.name == "*":
                            src_path = self.repo_root / src_file
                            references[src_file].update(
                                self._extract_public_symbols(src_path)
                            )
                        else:
                            references[src_file].add(alias.name)
                            local = alias.asname or alias.name
                            alias_map[local] = (src_file, alias.name)

                # also: from torch._inductor import scheduler
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

            # Pattern B/C: import module or import module as alias
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    src_file = module_map.get(alias.name)
                    if src_file:
                        local = alias.asname or alias.name.split(".")[-1]
                        alias_map[local] = (src_file, None)

        # Resolve attribute accesses: alias.something → symbol
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                obj = node.value.id
                if obj in alias_map:
                    src_file, _ = alias_map[obj]
                    references[src_file].add(node.attr)

        # Conservative fallback: alias with no attribute accesses found
        for local, (src_file, _) in alias_map.items():
            if src_file and src_file not in references:
                src_path = self.repo_root / src_file
                references[src_file].update(
                    self._extract_public_symbols(src_path)
                )

        return dict(references)

    # ─────────────────────────────────────────────
    # PHASE 1: Build both reverse graphs
    # ─────────────────────────────────────────────

    def _build_reverse_graphs(self, module_map: dict) -> tuple:
        """
        Build two graphs simultaneously.

        file_reverse_graph:
            "torch/_inductor/scheduler.py" → ["test/inductor/test_scheduler.py", ...]

        function_reverse_graph:
            "torch/_inductor/scheduler.py::_fuse_nodes" → ["test/inductor/test_scheduler.py"]
            "torch/_inductor/scheduler.py::Scheduler"   → ["test/inductor/test_scheduler.py",
                                                            "test/inductor/test_loop_ordering.py"]
        """
        file_reverse     = defaultdict(set)
        function_reverse = defaultdict(set)

        for test in self.test_files:
            rel_test   = str(test.relative_to(self.repo_root))
            references = self._extract_symbol_references(test, module_map)

            for src_file, symbols in references.items():
                # file-level (coarse)
                file_reverse[src_file].add(rel_test)

                # function-level (precise)
                for sym in symbols:
                    key = f"{src_file}::{sym}"
                    function_reverse[key].add(rel_test)

        return (
            {k: sorted(v) for k, v in file_reverse.items()},
            {k: sorted(v) for k, v in function_reverse.items()},
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
            print("    ⚠️  Falling back to AST inventory")
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
        # ── build module map (shared across all phases) ──
        module_map = {}
        for src in self.source_files:
            rel         = src.relative_to(self.repo_root)
            module_name = ".".join(rel.with_suffix("").parts)
            module_map[module_name] = str(rel)

            # ADD: map torch.optim → torch/optim/__init__.py
            # without this, "from torch.optim import X" never resolves
            if src.name == "__init__.py":                          # ADD
                package_name = ".".join(src.parent.relative_to(self.repo_root).parts)  # ADD
                module_map[package_name] = str(rel)                # ADD

        # ── Phase 0: source → source reverse graph ──
        print("  Phase 0: Building source import graph (transitive deps)...")
        t0 = time.time()
        source_reverse_graph = self._build_source_reverse_graph(module_map)
        print(f"    Done in {time.time() - t0:.2f}s  —  "
              f"{len(source_reverse_graph)} source files have dependents")

        # ── Phase 1: test → source reverse graphs ──
        print("  Phase 1: Building file + function graph via AST...")
        t1 = time.time()
        file_reverse, function_reverse = self._build_reverse_graphs(module_map)

        elapsed1      = time.time() - t1
        total_f_edges = sum(len(v) for v in file_reverse.values())
        total_s_edges = sum(len(v) for v in function_reverse.values())
        avg_file      = total_f_edges / max(len(file_reverse), 1)
        avg_sym       = total_s_edges / max(len(function_reverse), 1)

        print(f"    Done in {elapsed1:.2f}s")
        print(f"    File-level graph:     {len(file_reverse)} source files  "
              f"(avg {avg_file:.1f} test files each)")
        print(f"    Function-level graph: {len(function_reverse)} symbols  "
              f"(avg {avg_sym:.1f} test files each  —  "
              f"{'%.1f' % (avg_file / max(avg_sym, 0.01))}x more precise)")

        # ── Phase 2: test inventory ──
        print("  Phase 2: Building test inventory via pytest --collect-only...")
        t2 = time.time()

        all_candidate_tests = sorted(set(
            tf for tests in file_reverse.values() for tf in tests
        ))
        test_inventory = self._build_test_inventory(all_candidate_tests)

        print(f"    Done in {time.time() - t2:.2f}s — "
              f"{len(test_inventory)} test files indexed")

        return {
            "schema_version":         "2.1",
            "language":               self.language,
            "source_reverse_graph":   source_reverse_graph,   # NEW in 2.1
            "full_reverse_graph":     file_reverse,
            "function_reverse_graph": function_reverse,
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