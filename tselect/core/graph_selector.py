"""
graph_selector.py
-----------------
Queries the built dependency graph for a set of changed files
and returns the exact pytest node IDs to run.

Replaces:
  - map_files_to_components()        (needed ownership.yaml)
  - collect_tests_from_components()  (needed manual JSON)
"""

from pathlib import Path
from collections import defaultdict


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

    selected   = defaultdict(set)   # test_file → set of class names
    impact_map = defaultdict(set)   # test_file → which changed file triggered it

    for cf in changed_files:
        # normalize to repo-relative path
        try:
            rel = str(Path(cf).relative_to(repo_root))
        except ValueError:
            rel = cf

        # strip leading ./ if present
        rel = rel.lstrip("./")

        affected_tests = set(reverse_graph.get(rel, []))

        # ── fallback: directory proximity ──
        # used when static analysis finds no imports
        # (e.g. new file, or dynamic import pattern)
        if not affected_tests:
            changed_parts = set(Path(rel).parts)
            for test_file in test_inventory:
                test_parts = set(Path(test_file).parts)
                # at least 2 path segments in common (e.g. "test" + "inductor")
                if len(changed_parts & test_parts) >= 2:
                    affected_tests.add(test_file)

        for test_file in affected_tests:
            classes = set(test_inventory.get(test_file, {}).keys())
            selected[test_file].update(classes)
            impact_map[test_file].add(rel)

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
    """
    Extract flat list of pytest node IDs from selected tests.

    Uses the node_ids stored by pytest --collect-only during build-graph.
    These are the exact strings pytest expects — no class name mismatches.

    Example output:
        [
          "test/inductor/test_inductor_scheduler.py::TestSchedulerCPU::test_foo",
          "test/inductor/test_loop_ordering.py::LoopOrderingTestCPU::test_bar",
        ]
    """
    node_ids = []

    for test_file, data in selected.items():
        for cls_name, cls_data in data["classes"].items():
            node_ids.extend(cls_data.get("node_ids", []))

    return sorted(set(node_ids))  # deduplicate


def get_summary_info(selected: dict) -> tuple[list, dict]:
    """
    Return human-readable selected class list + test counts.
    Used for printing "Selected classes:" and summary report.
    """
    selected_classes = []
    class_test_count = {}

    for test_file, data in selected.items():
        for cls_name, cls_data in data["classes"].items():
            label = f"{test_file}::{cls_name}"
            selected_classes.append(label)
            class_test_count[cls_name] = cls_data["test_count"]

    return selected_classes, class_test_count