"""
graph_selector.py
-----------------
Selects tests for changed files using the dependency graph.

Two-tier selection strategy:

  Tier 1 — Function-level (precise):
    Uses diff_parser to find which functions/classes changed.
    Looks them up in function_reverse_graph.
    Result: only tests that reference those specific symbols.

    Example:
      scheduler.py::_fuse_nodes changed
      function_reverse_graph["scheduler.py::_fuse_nodes"] = [test_scheduler.py]
      → run test_scheduler.py only  (not all 7 files that import scheduler.py)

  Tier 2 — File-level fallback:
    Used when:
      (a) function_reverse_graph not in graph (old schema_version)
      (b) diff_parser couldn't extract function names
      (c) function-level lookup returns empty (new function, no tests yet)

  High-fanout guard:
    Infrastructure files (config.py, utils.py) trigger 100+ test files.
    These are treated separately — configurable threshold in tselect.yaml.
    Default: >30 test files = high-fanout = warn + skip targeted selection.

Selection metadata:
    Every selected test records WHY it was selected:
      - selection_mode: "function" | "file" | "proximity"
      - triggered_by:   which source file caused this
      - matched_symbols: which functions matched (function mode only)
    This metadata is the foundation for ML training data.
"""

from pathlib import Path
from collections import defaultdict


DEFAULT_HIGH_FANOUT_THRESHOLD = 30


def select_tests_from_graph(
    changed_files: list,
    graph: dict,
    repo_root: Path,
    config: dict = None,
) -> tuple[dict, int]:
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
    threshold = (
        config.get("graph", {}).get("fanout_threshold", DEFAULT_HIGH_FANOUT_THRESHOLD)
    )

    reverse_graph     = graph.get("full_reverse_graph", {})
    function_graph    = graph.get("function_reverse_graph", {})
    test_inventory    = graph.get("test_inventory", {})
    has_function_graph = bool(function_graph)

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
    # selected_tests: test_file → {triggered_by, symbols, mode, classes}
    selected_tests      = {}
    skipped_high_fanout = []

    for cf in changed_files:
        rel = _normalize(cf, repo_root)

        # ── high-fanout check (file-level) ──
        file_level_tests = set(reverse_graph.get(rel, []))
        if len(file_level_tests) > threshold:
            skipped_high_fanout.append((rel, len(file_level_tests)))
            continue

        # ── Tier 1: function-level selection ──
        symbols_changed = changed_functions.get(rel, set())

        if has_function_graph and symbols_changed:
            function_selected = _function_level_select(
                rel, symbols_changed, function_graph, test_inventory
            )

            if function_selected:
                for test_file, data in function_selected.items():
                    if test_file not in selected_tests:
                        selected_tests[test_file] = data
                    else:
                        # merge from multiple changed files
                        selected_tests[test_file]["triggered_by"] = sorted(set(
                            selected_tests[test_file]["triggered_by"] + data["triggered_by"]
                        ))
                        selected_tests[test_file]["matched_symbols"] = sorted(set(
                            selected_tests[test_file]["matched_symbols"] + data["matched_symbols"]
                        ))
                        selected_tests[test_file]["classes"].update(data["classes"])
                continue  # function-level succeeded — skip file-level for this file

        # ── Tier 2: file-level fallback ──
        if not file_level_tests:
            # proximity fallback — use directory mapping if configured
            dir_mapping = config.get("graph", {}).get("directory_mapping", [])
            file_level_tests = _proximity_fallback(rel, test_inventory, dir_mapping)
            mode = "proximity"
        else:
            mode = "file"

        for test_file in file_level_tests:
            classes = test_inventory.get(test_file, {})
            if not classes:
                continue
            if test_file not in selected_tests:
                selected_tests[test_file] = {
                    "triggered_by":    [rel],
                    "matched_symbols": [],
                    "selection_mode":  mode,
                    "classes":         dict(classes),
                }
            else:
                if rel not in selected_tests[test_file]["triggered_by"]:
                    selected_tests[test_file]["triggered_by"].append(rel)

    # ── warn about high-fanout files ──
    if skipped_high_fanout:
        print()
        print("    High-fanout files skipped (infrastructure files):")
        for f, count in skipped_high_fanout:
            print(f"     {f}  ({count} test files depend on it)")
        print()
        print("  These files are imported everywhere — targeted selection not meaningful.")
        print("  Run the full suite manually for these changes:")
        print("    pytest test/inductor/")
        print()

    # ── compute total methods ──
    total_methods = sum(
        cls_data["test_count"]
        for data in selected_tests.values()
        for cls_data in data["classes"].values()
    )

    # ── print selection mode summary ──
    function_mode_count = sum(
        1 for d in selected_tests.values() if d.get("selection_mode") == "function"
    )
    file_mode_count = len(selected_tests) - function_mode_count
    if selected_tests:
        print(f"  Selection mode: "
              f"{function_mode_count} test files via function-level  |  "
              f"{file_mode_count} via file-level fallback")

    return selected_tests, total_methods


def _function_level_select(
    src_file: str,
    symbols_changed: set,
    function_graph: dict,
    test_inventory: dict,
) -> dict:
    """
    Look up changed symbols in function_reverse_graph.
    Returns selected test files with which symbols matched.
    """
    # handle __module__ marker — module-level code changed
    # fall back to all symbols for this file
    if "__module__" in symbols_changed:
        # module-level change: collect all symbols for this file
        all_symbols = {
            key.split("::", 1)[1]
            for key in function_graph
            if key.startswith(f"{src_file}::")
        }
        symbols_changed = all_symbols or symbols_changed

    selected   = defaultdict(lambda: {"triggered_by": [], "matched_symbols": [], "selection_mode": "function", "classes": {}})
    found_any  = False

    for sym in symbols_changed:
        key        = f"{src_file}::{sym}"
        test_files = function_graph.get(key, [])

        for test_file in test_files:
            classes = test_inventory.get(test_file, {})
            if not classes:
                continue
            found_any = True
            selected[test_file]["triggered_by"]    = [src_file]
            selected[test_file]["matched_symbols"].append(sym)
            selected[test_file]["classes"].update(classes)

    if not found_any:
        return {}

    # deduplicate matched_symbols
    for data in selected.values():
        data["matched_symbols"] = sorted(set(data["matched_symbols"]))

    return dict(selected)


def _proximity_fallback(
    rel: str,
    test_inventory: dict,
    dir_mapping: list,
) -> set:
    """
    Last resort: find test files that are probably related by directory.

    Two approaches in order:
    1. Explicit directory_mapping from tselect.yaml (reliable)
    2. Path segment overlap heuristic (fallback, imprecise)
    """
    # approach 1: explicit mapping from tselect.yaml
    if dir_mapping:
        for mapping in dir_mapping:
            src_prefix  = mapping.get("source", "")
            test_prefix = mapping.get("tests", "")
            if rel.startswith(src_prefix):
                return {
                    tf for tf in test_inventory
                    if tf.startswith(test_prefix)
                }

    # approach 2: path segment overlap (heuristic)
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


def get_pytest_node_ids(selected: dict) -> list:
    node_ids = []
    for data in selected.values():
        for cls_data in data["classes"].values():
            node_ids.extend(cls_data.get("node_ids", []))
    return sorted(set(node_ids))


def get_summary_info(selected: dict) -> tuple[list, dict]:
    selected_classes = []
    class_test_count = {}

    for test_file, data in selected.items():
        for cls_name, cls_data in data["classes"].items():
            label = f"{test_file}::{cls_name}"
            selected_classes.append(label)
            class_test_count[cls_name] = cls_data["test_count"]

    return selected_classes, class_test_count