"""
graph_selector.py
-----------------
Selects tests for changed files using the dependency graph.

Selection strategy (in priority order):

  0. Pre-flight checks:
       - Test file changed → self-select directly, skip graph
       - Non-code file changed → skip entirely

  Tier 1 — Function-level (precise):
    Uses diff_parser to find which functions/classes changed.
    Looks them up in function_reverse_graph.
    Result: only tests that reference those specific symbols.

    Example:
      scheduler.py::_fuse_nodes changed
      function_reverse_graph["scheduler.py::_fuse_nodes"] = [test_scheduler.py]
      → run test_scheduler.py only  (not all 7 files that import scheduler.py)

    Method key fix (1a):
      diff_parser returns "SGD.step" (method inside class)
      graph stores "SGD" (class level, from import side)
      fix: if "SGD.step" not found → strip method → try "SGD"

  Tier 1b — Re-export routing:
    For transitive files that are DIRECT importers of the changed file,
    use the importer's function_reverse_graph entries with the changed
    symbols from the original file — not all file-level dependents.

    Example:
      sgd.py changes (SGD class)
      __init__.py imports sgd.py (direct importer)
      → use function_graph["__init__.py::SGD"] → 17 tests
      → NOT all 22 __init__.py dependents

  Tier 2 — File-level fallback:
    Used when:
      (a) function_reverse_graph not in graph (old schema_version)
      (b) diff_parser couldn't extract function names
      (c) function-level lookup returns empty (new function, no tests yet)
      (d) transitive file is not a direct importer of changed file

  Transitive expansion (schema 2.1):
    BFS through source_reverse_graph — runs to natural termination.
    Fanout guard is the ONLY stopping condition (no depth counter).

    Example:
      sgd.py changed
        → source_reverse_graph["sgd.py"] = [optim/__init__.py]
        → expanded = {sgd.py, optim/__init__.py}
        → file_reverse_graph["optim/__init__.py"] = [test_optim.py]
        → test_optim.py selected ✅

  High-fanout guard:
    Infrastructure files (torch/__init__.py etc.) trigger 100+ test files.
    Default: >30 test files = high-fanout = warn + skip targeted selection.

Selection metadata:
    Every selected test records WHY it was selected:
      - selection_mode: "self" | "function" | "re-export" | "file" | "proximity"
      - triggered_by:   which source file caused this
      - matched_symbols: which functions matched (function/re-export mode only)
    This metadata is the foundation for ML training data.
"""

from pathlib import Path
from collections import defaultdict


DEFAULT_HIGH_FANOUT_THRESHOLD = 30

# File extensions and names that are never source code
# Changes to these files never require test selection
NON_CODE_EXTENSIONS = {
    '.yml', '.yaml', '.rst', '.md', '.txt',
    '.json', '.cfg', '.ini', '.toml', '.csv',
    '.png', '.jpg', '.jpeg', '.gif', '.svg',
    '.sh', '.bash', '.bat', '.ps1',
}
NON_CODE_FILENAMES = {
    'setup.py', 'setup.cfg', 'MANIFEST.in',
    'Makefile', 'CMakeLists.txt', 'LICENSE', 'NOTICE',
}

# Test directory prefixes — files under these are test files
TEST_PATH_PREFIXES = ('test/', 'tests/', 'test\\', 'tests\\')


def _is_test_file(rel: str) -> bool:
    """Return True if this file is a test file (not a source file)."""
    return rel.startswith(TEST_PATH_PREFIXES)


def _is_non_code_file(rel: str) -> bool:
    """Return True if this file is non-code and needs no test selection."""
    p = Path(rel)
    return (
        p.suffix in NON_CODE_EXTENSIONS
        or p.name in NON_CODE_FILENAMES
    )


def _expand_transitively(
    changed_file: str,
    source_reverse_graph: dict,
    file_reverse_graph: dict,
    threshold: int = DEFAULT_HIGH_FANOUT_THRESHOLD,
) -> set:
    """
    BFS through source_reverse_graph to find all source files
    that transitively import the changed file.

    NO depth counter — runs to natural termination.
    Fanout guard is the ONLY stopping condition.

    Example:
        changed:  torch/optim/sgd.py
        hop 1:    torch/optim/__init__.py  (22 tests → safe ✅)
                  torch/__init__.py        (1075 tests → STOP 🛑)
        hop 2:    torch/optim/lr_scheduler.py (from __init__.py, 3 tests → safe ✅)
        hop 3:    frontier empty → BFS ends naturally
        result:   {sgd.py, optim/__init__.py, lr_scheduler.py}
    """
    visited  = {changed_file}
    frontier = {changed_file}

    while frontier:
        next_frontier = set()
        for f in frontier:
            importers = set(source_reverse_graph.get(f, []))
            for importer in importers:
                if importer in visited:
                    continue
                # fanout guard — stop BFS at infrastructure files
                if len(file_reverse_graph.get(importer, [])) > threshold:
                    continue
                next_frontier.add(importer)
                visited.add(importer)
        frontier = next_frontier

    return visited  # includes changed_file + all safe transitive importers


def select_tests_from_graph(
    changed_files: list,
    graph: dict,
    repo_root: Path,
    config: dict = None,
) -> tuple:
    """
    Main entry point. Returns (selected, total_methods).

    selected = {
        "test/inductor/test_scheduler.py": {
            "triggered_by": ["torch/_inductor/scheduler.py"],
            "matched_symbols": ["_fuse_nodes", "Scheduler"],
            "selection_mode": "function",
            "classes": {
                "TestSchedulerCPU": {
                    "node_ids": [...],
                    "test_count": 8
                }
            }
        }
    }
    """
    config    = config or {}
    threshold = config.get("graph", {}).get("fanout_threshold", DEFAULT_HIGH_FANOUT_THRESHOLD)

    reverse_graph        = graph.get("full_reverse_graph", {})
    function_graph       = graph.get("function_reverse_graph", {})
    test_inventory       = graph.get("test_inventory", {})
    source_reverse_graph = graph.get("source_reverse_graph", {})
    has_function_graph   = bool(function_graph)
    has_transitive       = bool(source_reverse_graph)

    # import here to avoid circular imports
    from tselect.core.diff_parser import get_changed_functions

    # get function-level diff info for each changed file
    changed_functions = {}
    if has_function_graph:
        try:
            changed_functions = get_changed_functions(repo_root, changed_files)
        except Exception:
            changed_functions = {}

    # ── selection ──────────────────────────────────
    selected_tests      = {}
    skipped_high_fanout = []
    skipped_non_code    = []

    for cf in changed_files:
        rel = _normalize(cf, repo_root)

        # ── Pre-flight 1: test file self-select (1d) ──
        # A changed test file always needs to run itself.
        # No graph lookup needed — add directly.
        if _is_test_file(rel):
            classes = test_inventory.get(rel, {})
            if classes:
                if rel not in selected_tests:
                    selected_tests[rel] = {
                        "triggered_by":    [rel],
                        "matched_symbols": [],
                        "selection_mode":  "self",
                        "classes":         dict(classes),
                    }
                print(f"    Self-select: {rel} (test file changed directly)")
            else:
                print(f"    Self-select: {rel} (test file changed, not in inventory — may need graph rebuild)")
            continue

        # ── Pre-flight 2: non-code file skip ──
        # Changes to .yml, .md, .rst, config files etc. don't need tests.
        if _is_non_code_file(rel):
            skipped_non_code.append(rel)
            continue

        # ── Transitive expansion ──
        # Always run — outer direct_count check removed (was checking wrong file).
        # Inner fanout guard handles explosion prevention correctly.
        if has_transitive:
            expanded = _expand_transitively(
                changed_file       = rel,
                source_reverse_graph = source_reverse_graph,
                file_reverse_graph   = reverse_graph,
                threshold            = threshold,
            )
        else:
            expanded = {rel}

        transitive_added = expanded - {rel}
        if transitive_added:
            print(f"    Transitive: {rel} → also checking {len(transitive_added)} dependent source files")

        # ── Direct importers of rel — used for re-export routing (1b) ──
        direct_importers_of_rel = set(source_reverse_graph.get(rel, []))

        # ── Get rel's changed symbols — needed for re-export routing ──
        rel_symbols_changed = changed_functions.get(rel, set())

        # ── Process each file in the expanded set ──
        for expanded_file in expanded:

            # ── High-fanout check ──
            file_level_tests = set(reverse_graph.get(expanded_file, []))
            if len(file_level_tests) > threshold:
                if expanded_file == rel:
                    skipped_high_fanout.append((rel, len(file_level_tests)))
                continue

            # ──────────────────────────────────────────────────────────────
            # DIRECTLY CHANGED FILE — Tier 1 function-level selection
            # ──────────────────────────────────────────────────────────────
            if expanded_file == rel:
                symbols_changed = rel_symbols_changed

                if has_function_graph and symbols_changed:
                    function_selected = _function_level_select(
                        expanded_file, symbols_changed, function_graph, test_inventory
                    )

                    if function_selected:
                        _merge_into_selected(selected_tests, function_selected)
                        continue  # function-level succeeded — don't fall to file-level

                # function-level found nothing — file-level fallback
                if not file_level_tests:
                    dir_mapping      = config.get("graph", {}).get("directory_mapping", [])
                    file_level_tests = _proximity_fallback(expanded_file, test_inventory, dir_mapping)
                    mode             = "proximity"
                else:
                    mode = "file"

                _add_file_level_tests(
                    selected_tests, file_level_tests, test_inventory,
                    trigger=rel, mode=mode
                )

            # ──────────────────────────────────────────────────────────────
            # TRANSITIVE FILE — check if re-export routing applies (1b)
            # ──────────────────────────────────────────────────────────────
            else:
                # Re-export routing (1b):
                # If expanded_file directly imports rel, it may re-export
                # symbols from rel. Use function_graph on expanded_file
                # with rel's changed symbols — more precise than all file-level.
                if has_function_graph and expanded_file in direct_importers_of_rel and rel_symbols_changed:
                    reexport_selected = _reexport_level_select(
                        importer_file    = expanded_file,
                        rel_symbols      = rel_symbols_changed,
                        original_trigger = rel,
                        function_graph   = function_graph,
                        test_inventory   = test_inventory,
                    )

                    if reexport_selected:
                        _merge_into_selected(selected_tests, reexport_selected)
                        continue  # re-export routing succeeded

                # Re-export routing found nothing (or not applicable)
                # Fall to file-level for this transitive file
                if not file_level_tests:
                    dir_mapping      = config.get("graph", {}).get("directory_mapping", [])
                    file_level_tests = _proximity_fallback(expanded_file, test_inventory, dir_mapping)
                    mode             = "proximity"
                else:
                    mode = "file"

                _add_file_level_tests(
                    selected_tests, file_level_tests, test_inventory,
                    trigger=rel, mode=mode
                )

    # ── warn about skipped files ──
    if skipped_non_code:
        print()
        for f in skipped_non_code:
            print(f"    Skipping non-code file: {f}")

    if skipped_high_fanout:
        print()
        print("    High-fanout files skipped (infrastructure files):")
        for f, count in skipped_high_fanout:
            print(f"     {f}  ({count} test files depend on it)")
        print()
        print("  These files are imported everywhere — targeted selection not meaningful.")
        print("  Run the full suite manually for these changes:")
        print("    pytest test/")
        print()

    # ── compute total methods ──
    total_methods = sum(
        cls_data["test_count"]
        for data in selected_tests.values()
        for cls_data in data["classes"].values()
    )

    # ── print selection mode summary ──
    mode_counts = defaultdict(int)
    for d in selected_tests.values():
        mode_counts[d.get("selection_mode", "unknown")] += 1

    if selected_tests:
        mode_str = "  |  ".join(
            f"{count} via {mode}"
            for mode, count in sorted(mode_counts.items())
        )
        print(f"  Selection mode: {mode_str}")

    return selected_tests, total_methods


# ─────────────────────────────────────────────────────────────────────────────
# Selection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _function_level_select(
    src_file: str,
    symbols_changed: set,
    function_graph: dict,
    test_inventory: dict,
) -> dict:
    """
    Look up changed symbols in function_reverse_graph.

    Change 1a — method key fix:
        diff_parser returns "SGD.step" (Class.method format)
        graph stores "SGD" (class level, from import side)
        If "src::SGD.step" not found → strip method → try "src::SGD"

    Returns selected test files with which symbols matched.
    Returns {} if nothing found (caller falls to file-level).
    """
    # __module__ expansion: module-level change → use all symbols in graph for this file
    if "__module__" in symbols_changed:
        all_symbols = {
            key.split("::", 1)[1]
            for key in function_graph
            if key.startswith(f"{src_file}::")
        }
        symbols_changed = all_symbols or symbols_changed

    selected  = defaultdict(lambda: {
        "triggered_by":    [],
        "matched_symbols": [],
        "selection_mode":  "function",
        "classes":         {},
    })
    found_any = False

    for sym in symbols_changed:
        key        = f"{src_file}::{sym}"
        test_files = function_graph.get(key, [])

        # 1a: method key fix — strip method suffix and retry at class level
        effective_sym = sym
        if not test_files and "." in sym:
            class_name = sym.split(".")[0]
            class_key  = f"{src_file}::{class_name}"
            test_files = function_graph.get(class_key, [])
            if test_files:
                effective_sym = class_name  # record class name, not method

        for test_file in test_files:
            classes = test_inventory.get(test_file, {})
            if not classes:
                continue
            found_any = True
            selected[test_file]["triggered_by"]    = [src_file]
            selected[test_file]["matched_symbols"].append(effective_sym)
            selected[test_file]["classes"].update(classes)

    if not found_any:
        return {}

    for data in selected.values():
        data["matched_symbols"] = sorted(set(data["matched_symbols"]))

    return dict(selected)


def _reexport_level_select(
    importer_file: str,
    rel_symbols: set,
    original_trigger: str,
    function_graph: dict,
    test_inventory: dict,
) -> dict:
    """
    Re-export routing (1b):

    When a transitive file directly imports the changed file,
    it may re-export symbols from it. Use the importer's function_graph
    entries with the changed symbols from the original file.

    Example:
        original:      sgd.py changed — SGD.step modified
        rel_symbols:   {"SGD.step"}
        importer_file: torch/optim/__init__.py
        → look up function_graph["__init__.py::SGD"]  (strip method → SGD)
        → finds 17 tests
        → mode = "re-export" (one hop more indirect than "function")

    Returns {} if nothing found (caller falls to file-level).
    """
    # normalize symbols — strip method suffix for class-level lookup
    normalized_symbols = set()
    for sym in rel_symbols:
        if sym == "__module__":
            # module-level change in original → use all symbols importer re-exports
            all_importer_syms = {
                key.split("::", 1)[1]
                for key in function_graph
                if key.startswith(f"{importer_file}::")
            }
            normalized_symbols.update(all_importer_syms)
        elif "." in sym:
            normalized_symbols.add(sym.split(".")[0])  # strip method
        else:
            normalized_symbols.add(sym)

    if not normalized_symbols:
        return {}

    selected  = defaultdict(lambda: {
        "triggered_by":    [],
        "matched_symbols": [],
        "selection_mode":  "re-export",
        "classes":         {},
    })
    found_any = False

    for sym in normalized_symbols:
        key        = f"{importer_file}::{sym}"
        test_files = function_graph.get(key, [])

        for test_file in test_files:
            classes = test_inventory.get(test_file, {})
            if not classes:
                continue
            found_any = True
            selected[test_file]["triggered_by"]    = [original_trigger]
            selected[test_file]["matched_symbols"].append(sym)
            selected[test_file]["classes"].update(classes)

    if not found_any:
        return {}

    for data in selected.values():
        data["matched_symbols"] = sorted(set(data["matched_symbols"]))

    return dict(selected)


def _add_file_level_tests(
    selected_tests: dict,
    file_level_tests: set,
    test_inventory: dict,
    trigger: str,
    mode: str,
) -> None:
    """Add file-level or proximity test candidates to selected_tests."""
    for test_file in file_level_tests:
        classes = test_inventory.get(test_file, {})
        if not classes:
            continue

        if test_file not in selected_tests:
            selected_tests[test_file] = {
                "triggered_by":    [trigger],
                "matched_symbols": [],
                "selection_mode":  mode,
                "classes":         dict(classes),
            }
        else:
            if trigger not in selected_tests[test_file]["triggered_by"]:
                selected_tests[test_file]["triggered_by"].append(trigger)


def _merge_into_selected(selected_tests: dict, new_selections: dict) -> None:
    """Merge new selections into existing selected_tests, combining metadata."""
    for test_file, data in new_selections.items():
        if test_file not in selected_tests:
            selected_tests[test_file] = data
        else:
            selected_tests[test_file]["triggered_by"] = sorted(set(
                selected_tests[test_file]["triggered_by"] + data["triggered_by"]
            ))
            selected_tests[test_file]["matched_symbols"] = sorted(set(
                selected_tests[test_file]["matched_symbols"] + data["matched_symbols"]
            ))
            selected_tests[test_file]["classes"].update(data["classes"])


def _proximity_fallback(
    rel: str,
    test_inventory: dict,
    dir_mapping: list,
) -> set:
    """
    Last-resort fallback: find tests by path component overlap.
    Used when graph has no information about this file.
    """
    if dir_mapping:
        for mapping in dir_mapping:
            src_prefix  = mapping.get("source", "")
            test_prefix = mapping.get("tests", "")
            if rel.startswith(src_prefix):
                return {
                    tf for tf in test_inventory
                    if tf.startswith(test_prefix)
                }

    changed_parts = set(Path(rel).parts)
    candidates    = set()
    for test_file in test_inventory:
        test_parts = set(Path(test_file).parts)
        if len(changed_parts & test_parts) >= 2:
            candidates.add(test_file)

    return candidates


def _normalize(path_str: str, repo_root: Path) -> str:
    try:
        return str(Path(path_str).relative_to(repo_root))
    except ValueError:
        return str(path_str).lstrip("./")


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers used by cli/main.py
# ─────────────────────────────────────────────────────────────────────────────

def get_pytest_node_ids(selected: dict) -> list:
    node_ids = []
    for data in selected.values():
        for cls_data in data["classes"].values():
            node_ids.extend(cls_data.get("node_ids", []))
    return sorted(set(node_ids))


def get_summary_info(selected: dict) -> tuple:
    selected_classes = []
    class_test_count = {}

    for test_file, data in selected.items():
        for cls_name, cls_data in data["classes"].items():
            label = f"{test_file}::{cls_name}"
            selected_classes.append(label)
            class_test_count[cls_name] = cls_data["test_count"]

    return selected_classes, class_test_count
