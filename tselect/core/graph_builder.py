"""
graph_builder.py
----------------
Builds the dependency graph for tselect.

Two-phase build:
  Phase 1: AST parsing  → reverse graph (source file → test files that import it)
  Phase 2: pytest --collect-only in batches → test inventory (actual runnable class names)

Supported languages for Phase 1:
  python  → fully supported via ast module
  others  → clear error message, not silently broken

Why batched pytest --collect-only:
  - 1256 files at once → timeout (>120s)
  - 50 files per batch → ~3s per batch, ~75s total
  - --continue-on-collection-errors → distributed/CUDA tests that can't
    import on Mac are silently skipped, not errors
"""

import ast
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


SUPPORTED_LANGUAGES = {"python"}

COMING_SOON_LANGUAGES = {"java", "javascript", "typescript", "go", "cpp"}


class UnsupportedLanguageError(Exception):
    pass


class GraphBuilder:
    def __init__(self, layout):
        self.layout       = layout
        self.repo_root    = layout.repo_root
        self.language     = layout.language
        self.source_files = layout.source_files
        self.test_files   = layout.test_files
        self.batch_size   = getattr(layout, "collect_batch_size", 50)

        # validate language upfront — fail clearly, not silently
        self._validate_language()

    def _validate_language(self):
        if self.language in SUPPORTED_LANGUAGES:
            return

        if self.language in COMING_SOON_LANGUAGES:
            raise UnsupportedLanguageError(
                f"\n  ❌ Language '{self.language}' is not yet supported by tselect.\n"
                f"\n"
                f"  Currently supported: python\n"
                f"  Coming soon: java, javascript, typescript, go, cpp\n"
                f"\n"
                f"  If you'd like to help add {self.language} support:\n"
                f"  → https://github.com/your-org/tselect/issues\n"
            )

        raise UnsupportedLanguageError(
            f"\n  ❌ Unknown language '{self.language}'.\n"
            f"  Check 'repo.language' in your tselect.yaml.\n"
            f"  Supported: python\n"
        )

    # ─────────────────────────────────────────────
    # PHASE 1: AST import extraction
    # ─────────────────────────────────────────────

    def _extract_python_imports(self, file_path: Path) -> set:
        imports = set()
        try:
            source = file_path.read_text(encoding="utf-8", errors="ignore")
            tree   = ast.parse(source)
        except Exception:
            return imports

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
                    for alias in node.names:
                        if alias.name != "*":
                            imports.add(f"{node.module}.{alias.name}")

        return imports

    def extract_imports(self, file_path: Path) -> set:
        if self.language == "python":
            return self._extract_python_imports(file_path)
        return set()

    def _build_reverse_graph(self, module_map: dict) -> dict:
        """
        Build: source_file → set of test files that directly import it.
        Direct imports only — no transitive expansion.
        """
        reverse_graph = defaultdict(set)

        for test in self.test_files:
            rel_test = str(test.relative_to(self.repo_root))
            imports  = self.extract_imports(test)

            for imp in imports:
                if imp in module_map:
                    src_file = module_map[imp]
                    reverse_graph[src_file].add(rel_test)

        return {k: sorted(v) for k, v in reverse_graph.items()}

    # ─────────────────────────────────────────────
    # PHASE 2: pytest --collect-only for inventory
    # ─────────────────────────────────────────────

    def _collect_batch(self, batch: list) -> str:
        """
        Run pytest --collect-only on a batch of up to `batch_size` files.
        --continue-on-collection-errors means failed imports (e.g. distributed
        tests needing CUDA) are silently skipped — not errors.
        """
        cmd = [
            sys.executable, "-m", "pytest",
            "--collect-only", "-q", "--no-header",
            "--continue-on-collection-errors",
        ] + batch

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.repo_root),
                timeout=30,
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            return ""
        except Exception:
            return ""

    def _build_test_inventory(self, candidate_test_files: list) -> dict:
        """
        Run pytest --collect-only in small batches to get actual runnable
        test node IDs.

        Batching solves:
          1. Timeout — 1256 files at once exceeds 30s; 50 files is ~3s
          2. Import failures — distributed tests that can't load on Mac
             are silently skipped per-batch without aborting everything
          3. Dynamic test generation — gets TestSchedulerCPU not TestScheduler
        """
        if not candidate_test_files:
            return {}

        batch_size = self.batch_size
        batches    = [
            candidate_test_files[i:i + batch_size]
            for i in range(0, len(candidate_test_files), batch_size)
        ]

        print(f"    Collecting {len(candidate_test_files)} test files "
              f"in {len(batches)} batches of {batch_size}...")

        inventory  = defaultdict(lambda: defaultdict(lambda: {"node_ids": [], "test_count": 0}))
        successful = 0

        for i, batch in enumerate(batches):
            stdout = self._collect_batch(batch)
            if not stdout:
                continue

            for line in stdout.splitlines():
                line = line.strip()
                if "::" not in line:
                    continue
                if line.startswith(("ERROR", "Warning", "FAILED")):
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
                print(f"    [{i+1}/{len(batches)} batches done] "
                      f"{successful} test methods collected")

        if successful == 0:
            print("    ⚠️  Collection returned no results — falling back to AST inventory")
            return self._build_ast_inventory(candidate_test_files)

        return {
            tf: {cls: dict(data) for cls, data in classes.items()}
            for tf, classes in inventory.items()
        }

    def _build_ast_inventory(self, test_files: list) -> dict:
        """
        Fallback AST-based inventory.
        Less accurate for repos using dynamic test generation (e.g. PyTorch).
        """
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
                            item.name for item in node.body
                            if isinstance(item, ast.FunctionDef)
                            and item.name.startswith("test_")
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
        print("  Phase 1: Building import graph via AST...")
        t0 = time.time()

        module_map = {}
        for src in self.source_files:
            rel         = src.relative_to(self.repo_root)
            module_name = ".".join(rel.with_suffix("").parts)
            module_map[module_name] = str(rel)

        reverse_graph = self._build_reverse_graph(module_map)
        print(f"    Done in {time.time() - t0:.2f}s — "
              f"{len(reverse_graph)} source files indexed")

        print("  Phase 2: Building test inventory via pytest --collect-only...")
        t1 = time.time()

        all_candidate_tests = sorted(set(
            tf for tests in reverse_graph.values() for tf in tests
        ))

        test_inventory = self._build_test_inventory(all_candidate_tests)
        print(f"    Done in {time.time() - t1:.2f}s — "
              f"{len(test_inventory)} test files indexed")

        return {
            "language":           self.language,
            "full_reverse_graph": reverse_graph,
            "test_inventory":     test_inventory,
            "built_at":           time.time(),
        }

    # ─────────────────────────────────────────────
    # SAVE
    # ─────────────────────────────────────────────

    def save(self, graph_data: dict) -> Path:
        graph_dir  = self.repo_root / ".graph" / "tselect"
        graph_dir.mkdir(parents=True, exist_ok=True)
        graph_path = graph_dir / "dependency_graph.json"

        with open(graph_path, "w") as f:
            json.dump(graph_data, f, indent=2)

        return graph_path