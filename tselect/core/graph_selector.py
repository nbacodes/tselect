"""
graph_selector.py
-----------------
Queries the built dependency graph for changed files
and returns the exact pytest node IDs to run.

Key addition: high-fanout filtering.
  Files like config.py, utils.py are imported by 100+ test files.
  Running all their tests defeats the purpose of targeted selection.
  These files are treated as "infrastructure" — changes to them
  should trigger a broader manual run, not tselect's targeted mode.
"""

from pathlib import Path
from collections import defaultdict


# If a source file triggers more than this many test files,
# it's considered "infrastructure" and skipped for targeted selection.
# The user is warned and should run the full suite manually.
HIGH_FANOUT_THRESHOLD = 30


def select_tests_from_graph(
    changed_files: list,
    graph: dict,
    repo_root: Path,
) -> tuple[dict, int]:
    """
    Given changed files + built graph → selected tests.

    Returns:
        selected      : {test_file → {triggered_by, classes}}
        total_methods : int
    """
    
    reverse_graph  = graph.get("full_reverse_graph", {})
    test_inventory = graph.get("test_inventory", {})

    selected   = defaultdict(set)
    impact_map = defaultdict(set)
    skipped_high_fanout = []

    for cf in changed_files:
        try:
            rel = str(Path(cf).relative_to(repo_root))
        except ValueError:
            rel = cf
        rel = rel.lstrip("./")

        affected_tests = set(reverse_graph.get(rel, []))

        # ── high-fanout check ──
        # config.py, utils.py etc. touch 100+ test files
        # skip them for targeted selection and warn user
        if len(affected_tests) > HIGH_FANOUT_THRESHOLD:
            skipped_high_fanout.append((rel, len(affected_tests)))
            continue

        # ── fallback: directory proximity ──
        if not affected_tests:
            changed_parts = set(Path(rel).parts)
            for test_file in test_inventory:
                test_parts = set(Path(test_file).parts)
                if len(changed_parts & test_parts) >= 2:
                    affected_tests.add(test_file)

        for test_file in affected_tests:
            classes = set(test_inventory.get(test_file, {}).keys())
            selected[test_file].update(classes)
            impact_map[test_file].add(rel)

    # warn about high-fanout files
    if skipped_high_fanout:
        print()
        print("  ⚠️  High-fanout files skipped (too many dependents for targeted selection):")
        for f, count in skipped_high_fanout:
            print(f"     {f}  ({count} test files depend on it)")
        print()
        print("  These are infrastructure files (config, utils, etc.).")
        print("  For changes to these, run the full test suite manually:")
        print("    pytest test/inductor/")
        print()

    # ── build result ──
    result        = {}
    total_methods = 0

    for test_file in sorted(selected.keys()):
        inv     = test_inventory.get(test_file, {})
        classes = {}

        for cls_name in selected[test_file]:
            if cls_name in inv:
                classes[cls_name] = inv[cls_name]
                total_methods += inv[cls_name]["test_count"]

        if classes:
            result[test_file] = {
                "triggered_by": sorted(impact_map[test_file]),
                "classes":      classes,
            }

    return result, total_methods


def get_pytest_node_ids(selected: dict) -> list:
    node_ids = []
    for test_file, data in selected.items():
        for cls_name, cls_data in data["classes"].items():
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