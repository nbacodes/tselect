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

  Transitive expansion (NEW in schema 2.1):
    Before any lookup, changed files are expanded through source_reverse_graph.
    Example:
      sgd.py changed
        → source_reverse_graph["sgd.py"] = [optim/__init__.py]
        → expanded = {sgd.py, optim/__init__.py}
        → file_reverse_graph["optim/__init__.py"] = [test_optim.py]
        → test_optim.py selected ✅

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
DEFAULT_TRANSITIVE_DEPTH      = 3   # how many hops to follow in source graph


def _expand_transitively(
    changed_file: str,
    source_reverse_graph: dict,
    file_reverse_graph: dict,
    depth: int = DEFAULT_TRANSITIVE_DEPTH,
    threshold: int = DEFAULT_HIGH_FANOUT_THRESHOLD,
) -> set:
    """
    BFS through source_reverse_graph to find all source files
    that transitively import the changed file.

    Fanout guard: stops BFS at any file that has more than
    `threshold` test dependents — these are infrastructure files
    (torch/__init__.py, common_utils.py etc.) that import everything
    and would cause an explosion in candidate count.

    Example:
        changed:  torch/optim/sgd.py
        depth 1:  torch/optim/__init__.py  (3 tests → safe, follow ✅)
                  torch/__init__.py        (1075 tests → STOP 🛑)
        result:   {sgd.py, optim/__init__.py}
    """
    visited  = {changed_file}
    frontier = {changed_file}

    for _ in range(depth):
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
        if not frontier:
            break

    return visited  # includes changed_file + safe transitive importers


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
    trans_depth = config.get("graph", {}).get("transitive_depth", DEFAULT_TRANSITIVE_DEPTH)

    reverse_graph        = graph.get("full_reverse_graph", {})
    function_graph       = graph.get("function_reverse_graph", {})
    test_inventory       = graph.get("test_inventory", {})
    source_reverse_graph = graph.get("source_reverse_graph", {})   # NEW
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

    for cf in changed_files:
        rel = _normalize(cf, repo_root)

        # ── transitive expansion ──
        if has_transitive:
            expanded = _expand_transitively(
                changed_file         = rel,
                source_reverse_graph = source_reverse_graph,
                file_reverse_graph   = reverse_graph,
                depth                = trans_depth,
                threshold            = threshold,
            )
        else:
            expanded = {rel}

        transitive_added = expanded - {rel}
        if transitive_added:
            print(f"    Transitive: {rel} → also checking {len(transitive_added)} dependent source files")

        # ── process each file in the expanded set ──
        for expanded_file in expanded:

            # ── high-fanout check ──
            file_level_tests = set(reverse_graph.get(expanded_file, []))
            if len(file_level_tests) > threshold:
                # only warn for the directly changed file, not transitive ones
                if expanded_file == rel:
                    skipped_high_fanout.append((rel, len(file_level_tests)))
                continue

            # ── Tier 1: function-level selection ──
            # Only apply function-level for the DIRECTLY changed file
            # (we know exactly which symbols changed there).
            # For transitive files, use file-level (we don't know which symbols
            # were affected by the upstream change).
            symbols_changed = changed_functions.get(expanded_file, set()) if expanded_file == rel else set()

            if has_function_graph and symbols_changed and expanded_file == rel:
                function_selected = _function_level_select(
                    expanded_file, symbols_changed, function_graph, test_inventory
                )

                if function_selected:
                    for test_file, data in function_selected.items():
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
                    continue  # function-level succeeded

            # ── Tier 2: file-level fallback ──
            if not file_level_tests:
                dir_mapping      = config.get("graph", {}).get("directory_mapping", [])
                file_level_tests = _proximity_fallback(expanded_file, test_inventory, dir_mapping)
                mode             = "proximity"
            else:
                mode = "file"

            for test_file in file_level_tests:
                classes = test_inventory.get(test_file, {})
                if not classes:
                    continue

                # triggered_by = the ORIGINAL changed file (not the transitive intermediary)
                trigger = rel

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
    if "__module__" in symbols_changed:
        all_symbols = {
            key.split("::", 1)[1]
            for key in function_graph
            if key.startswith(f"{src_file}::")
        }
        symbols_changed = all_symbols or symbols_changed

    selected  = defaultdict(lambda: {"triggered_by": [], "matched_symbols": [], "selection_mode": "function", "classes": {}})
    found_any = False

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

    for data in selected.values():
        data["matched_symbols"] = sorted(set(data["matched_symbols"]))

    return dict(selected)


def _proximity_fallback(
    rel: str,
    test_inventory: dict,
    dir_mapping: list,
) -> set:
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