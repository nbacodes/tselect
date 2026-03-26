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

Changes from 2.1:
  - function_reverse_graph now maps to test METHODS not test FILES
    "sgd.py::SGD" → ["test_optim.py::TestOptimCPU::test_sgd_momentum", ...]
    Enables method-level selection — run 3 tests not 480
  - file_identifiers: every source file's identifier set
    Used by graph_selector for identifier overlap BFS stopping condition
  - fanout_threshold: computed from actual repo distribution gap
    Not hardcoded — works for any repo
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
DEFAULT_HIGH_FANOUT_THRESHOLD = 30


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
    # SOURCE: extract all identifiers (NEW)
    # ─────────────────────────────────────────────

    def _extract_all_identifiers(self, file_path: Path) -> set:
        """
        Extract every identifier referenced in a source file.
        Used to build file_identifiers for BFS stopping condition.

        Collects:
          - ast.Name nodes (variable names, function calls, class references)
          - ast.Attribute nodes (method names, attribute accesses)

        Example for torch/optim/__init__.py:
            {"SGD", "Adam", "RMSprop", "step", "zero_grad", "momentum", ...}

        Example for torch/_dynamo/eval_frame.py:
            {"OptimizedModule", "compiler_fn", "skip", "resume_execution", ...}

        At BFS time: if changed_symbols ∩ file_identifiers[importer] == empty
        → importer doesn't use anything that changed → stop BFS here
        """
        identifiers = set()
        try:
            source = file_path.read_text(encoding="utf-8", errors="ignore")
            tree   = ast.parse(source)
        except Exception:
            return identifiers

        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if node.id and not node.id.startswith("__"):
                    identifiers.add(node.id)
            elif isinstance(node, ast.Attribute):
                if node.attr and not node.attr.startswith("__"):
                    identifiers.add(node.attr)

        return identifiers

    # ─────────────────────────────────────────────
    # DYNAMIC FANOUT THRESHOLD (NEW)
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

        This means: files with > 116 test dependents are infrastructure.
        Files with <= 116 are specific enough to follow in BFS.

        For any repo — threshold is derived from its own data.
        No hardcoding needed.
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

        Example:
            torch/optim/sgd.py → [torch/optim/__init__.py, torch/optim/optimizer.py]
        """
        forward = defaultdict(set)

        for src in self.source_files:
            rel = str(src.relative_to(self.repo_root))

            try:
                source = src.read_text(encoding="utf-8", errors="ignore")
                tree   = ast.parse(source)
            except Exception:
                continue

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

        reverse = defaultdict(set)
        for importer, importees in forward.items():
            for importee in importees:
                reverse[importee].add(importer)

        return {k: sorted(v) for k, v in reverse.items()}

    # ─────────────────────────────────────────────
    # TEST: extract symbol-level references (file level)
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
                            references[src_file].update(
                                self._extract_public_symbols(src_path)
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
    # TEST: extract method-level references (NEW)
    # ─────────────────────────────────────────────

    def _extract_method_level_references(
        self, test_path: Path, module_map: dict
    ) -> dict:
        """
        Parse a test file and return which symbols each TEST METHOD references.

        Returns:
            {
                "TestOptimCPU::test_sgd_momentum": {
                    "torch/optim/sgd.py": {"SGD", "step"},
                },
                "TestOptimCPU::test_adam": {
                    "torch/optim/adam.py": {"Adam"},
                }
            }

        This enables function_reverse_graph to map:
            "sgd.py::SGD" → ["test_optim.py::TestOptimCPU::test_sgd_momentum"]
        Instead of:
            "sgd.py::SGD" → ["test_optim.py"]

        How it works:
            1. Find file-level imports (alias_map) — same as _extract_symbol_references
            2. For each test class → each test method:
               walk only that method's AST subtree
               find Name and Attribute nodes that match imported symbols
               record which source files those symbols came from
        """
        method_references = {}

        try:
            source = test_path.read_text(encoding="utf-8", errors="ignore")
            tree   = ast.parse(source)
        except Exception:
            return method_references

        # step 1: build file-level alias map (same as _extract_symbol_references)
        alias_map    = {}   # local_name → (src_file, sym_or_None)
        file_symbols = defaultdict(set)  # src_file → set of imported symbols

        for node in ast.walk(tree):
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

        if not alias_map:
            return method_references

        # step 2: for each test class → each test method, find symbol usage
        for class_node in ast.walk(tree):
            if not isinstance(class_node, ast.ClassDef):
                continue

            for method_node in class_node.body:
                if not isinstance(method_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if not method_node.name.startswith("test_"):
                    continue

                method_key = f"{class_node.name}::{method_node.name}"
                refs       = defaultdict(set)

                # walk only this method's body
                for node in ast.walk(method_node):
                    # direct Name reference — e.g. SGD(...)
                    if isinstance(node, ast.Name):
                        name = node.id
                        if name in alias_map:
                            src_file, sym = alias_map[name]
                            refs[src_file].add(sym or name)

                    # attribute access — e.g. module.something or opt.step()
                    elif isinstance(node, ast.Attribute):
                        attr = node.attr
                        if isinstance(node.value, ast.Name):
                            obj = node.value.id
                            if obj in alias_map:
                                src_file, _ = alias_map[obj]
                                refs[src_file].add(attr)

                if refs:
                    method_references[method_key] = dict(refs)

        return method_references

    # ─────────────────────────────────────────────
    # PHASE 1: Build reverse graphs (updated)
    # ─────────────────────────────────────────────

    def _build_reverse_graphs(self, module_map: dict) -> tuple:
        """
        Build three graphs simultaneously.

        file_reverse_graph (unchanged):
            "torch/_inductor/scheduler.py" → ["test/inductor/test_scheduler.py", ...]

        function_reverse_graph (UPDATED — now maps to test METHODS):
            "torch/_inductor/scheduler.py::_fuse_nodes" →
                ["test/inductor/test_scheduler.py::TestSchedulerCPU::test_fuse_nodes"]
            "torch/optim/sgd.py::SGD" →
                ["test/optim/test_optim.py::TestOptimCPU::test_sgd_momentum",
                 "test/optim/test_optim.py::TestOptimCPU::test_sgd_dampening"]

        file_identifiers (NEW):
            "torch/optim/__init__.py" → ["SGD", "Adam", "step", "momentum", ...]
            "torch/_dynamo/eval_frame.py" → ["OptimizedModule", "compiler_fn", ...]
        """
        file_reverse     = defaultdict(set)
        function_reverse = defaultdict(set)
        file_identifiers = {}

        total = len(self.test_files)
        for idx, test in enumerate(self.test_files):
            rel_test = str(test.relative_to(self.repo_root))

            # file-level references (for file_reverse_graph)
            file_refs = self._extract_symbol_references(test, module_map)
            for src_file, symbols in file_refs.items():
                file_reverse[src_file].add(rel_test)

            # method-level references (for function_reverse_graph)
            method_refs = self._extract_method_level_references(test, module_map)
            for method_key, src_refs in method_refs.items():
                # method_key = "TestClass::test_method"
                full_method_id = f"{rel_test}::{method_key}"
                for src_file, symbols in src_refs.items():
                    for sym in symbols:
                        key = f"{src_file}::{sym}"
                        function_reverse[key].add(full_method_id)

            if (idx + 1) % 100 == 0:
                print(f"    [{idx+1}/{total} test files processed]")

        # file_identifiers for source files
        print("  Building file identifier index for BFS stopping condition...")
        for src in self.source_files:
            rel = str(src.relative_to(self.repo_root))
            ids = self._extract_all_identifiers(src)
            if ids:
                file_identifiers[rel] = sorted(ids)

        return (
            {k: sorted(v) for k, v in file_reverse.items()},
            {k: sorted(v) for k, v in function_reverse.items()},
            file_identifiers,
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
        print("  Phase 0: Building source import graph (transitive deps)...")
        t0 = time.time()
        source_reverse_graph = self._build_source_reverse_graph(module_map)
        print(f"    Done in {time.time() - t0:.2f}s  —  "
              f"{len(source_reverse_graph)} source files have dependents")

        # Phase 1: test → source reverse graphs + file identifiers
        print("  Phase 1: Building file + function graph + identifier index...")
        t1 = time.time()
        file_reverse, function_reverse, file_identifiers = self._build_reverse_graphs(module_map)

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
            "schema_version":         "3.0",
            "language":               self.language,
            "source_reverse_graph":   source_reverse_graph,
            "full_reverse_graph":     file_reverse,
            "function_reverse_graph": function_reverse,   # now maps to test METHODS
            "file_identifiers":       file_identifiers,   # NEW
            "fanout_threshold":       fanout_threshold,   # NEW
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