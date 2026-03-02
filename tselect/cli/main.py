"""
cli/main.py
-----------
tselect CLI — zero user hardship design.

Commands:
  tselect build-graph          → auto-build dependency graph (run once)
  tselect run                  → auto-detect changed files via git, select tests
  tselect run --execute        → select + run tests
  tselect run --changed f1 f2  → manually specify changed files
  tselect baseline --execute   → record full suite baseline time

Design principles:
  - No manual JSON files needed
  - No ownership.yaml needed  
  - Auto git detection — user never needs to type file paths
  - Auto stale graph detection — warns when graph needs rebuild
  - Falls back to legacy mode if no graph built yet
  - Clear, informative output at every step
"""

import argparse
import time
from pathlib import Path

from tselect.utils.loader import load_yaml, load_json
from tselect.core.selector import map_files_to_components, collect_tests_from_components
from tselect.core.graph_loader import GraphLoader
from tselect.core.graph_selector import (
    select_tests_from_graph,
    get_pytest_node_ids,
    get_summary_info,
)
from tselect.adapters.pytest_adapter import (
    build_pytest_command,
    build_pytest_command_from_classes,
    execute_command,
)
from tselect.reporting.summary import generate_summary
from tselect.reporting.cache import load_cache, save_cache
from tselect.adapters.baseline_detector import detect_baseline_command
from tselect.adapters.git_adapter import get_changed_files
from tselect.utils.logger import setup_logger

logger = setup_logger()

# graph is considered stale after 7 days
GRAPH_STALE_DAYS = 7


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def _is_graph_stale(graph: dict) -> bool:
    """Check if graph was built more than GRAPH_STALE_DAYS ago."""
    built_at = graph.get("built_at")
    if not built_at:
        return False  # old graph without timestamp — don't force rebuild
    age_days = (time.time() - built_at) / 86400
    return age_days > GRAPH_STALE_DAYS


def _print_section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _pretty_print_command(node_ids: list[str]):
    """Print the pytest command in a readable format."""
    print("\n=== TSELECT COMMAND ===\n")
    print("pytest \\")
    for nid in node_ids:
        print(f"  {nid} \\")
    print("\nTo execute:")
    print("  tselect run --execute")
    print()


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="tselect",
        description="Targeted test selector — runs only what matters for your changes.",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command")

    # ── build-graph ──
    subparsers.add_parser(
        "build-graph",
        help="Auto-build dependency graph from repo (run once, re-run when repo structure changes)",
    )

    # ── run ──
    run_parser = subparsers.add_parser(
        "run",
        help="Select and optionally run tests for changed files",
    )
    run_parser.add_argument(
        "--changed",
        nargs="+",
        required=False,
        help="Changed files to analyze (auto-detected from git diff if omitted)",
    )
    run_parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the selected tests",
    )

    # ── baseline ──
    baseline_parser = subparsers.add_parser(
        "baseline",
        help="Record full test suite baseline time",
    )
    baseline_parser.add_argument("--execute", action="store_true")

    args = parser.parse_args()

    if args.debug:
        logger.setLevel("DEBUG")

    # ─────────────────────────────────────────────
    # BUILD-GRAPH
    # ─────────────────────────────────────────────

    if args.command == "build-graph":
        logger.info("Building dependency graph")

        from tselect.core.layout import RepoLayoutInferer
        from tselect.core.graph_builder import GraphBuilder

        repo_root = Path.cwd()

        print()
        print("=" * 60)
        print("  tselect — Building Dependency Graph")
        print("=" * 60)
        print(f"  Repo: {repo_root}")
        print()

        print("► Scanning repository layout...")
        layout  = RepoLayoutInferer(repo_root).infer()
        builder = GraphBuilder(layout)

        print(f"  Language  : {layout.language}")
        print(f"  Source    : {len(layout.source_files)} files")
        print(f"  Tests     : {len(layout.test_files)} files")
        print()

        print("► Building graph...")
        t_start = time.time()
        graph   = builder.build()
        t_total = time.time() - t_start

        path = builder.save(graph)

        reverse    = graph.get("full_reverse_graph", {})
        inventory  = graph.get("test_inventory", {})

        print()
        print("─" * 60)
        print("  ✅ Graph built successfully")
        print("─" * 60)
        print(f"  Source files indexed  : {len(reverse)}")
        print(f"  Test files indexed    : {len(inventory)}")
        print(f"  Total build time      : {t_total:.1f}s")
        print(f"  Saved to              : {path}")

        if reverse:
            print()
            print("  Top 5 most impactful source files:")
            sorted_impact = sorted(reverse.items(), key=lambda x: len(x[1]), reverse=True)
            for src, tests in sorted_impact[:5]:
                print(f"    {len(tests):3d} tests ← {src}")

        print()
        print("  Now run:")
        print("    tselect run")
        print("    tselect run --execute")
        print()

    # ─────────────────────────────────────────────
    # RUN
    # ─────────────────────────────────────────────

    elif args.command == "run":
        logger.info("Starting selective test run")
        repo_root = Path.cwd()

        cache         = load_cache(repo_root)
        baseline_time = cache.get("baseline_time")

        print()
        print("=" * 60)
        print("  tselect — Targeted Test Selection")
        print("=" * 60)

        # ── step 1: get changed files ──
        if args.changed:
            logger.info("Using manually provided changed files")
            changed_files = args.changed
            print(f"\n  Changed files (manual):")
        else:
            logger.info("Auto-detecting changed files via git diff")
            changed_files = get_changed_files()
            print(f"\n  Changed files (git diff):")

        if not changed_files:
            print("  No changed files detected. Nothing to run.")
            return

        for f in changed_files:
            print(f"    • {f}")

        # ── step 2: load graph ──
        graph_loader = GraphLoader(repo_root)

        if graph_loader.exists():
            graph = graph_loader.load()

            # warn if stale
            if _is_graph_stale(graph):
                print()
                print("  ⚠️  Dependency graph is older than 7 days.")
                print("     Run 'tselect build-graph' to refresh it.")

            # ── GRAPH MODE ──
            logger.info("Using auto-built dependency graph")

            selected, total_tests = select_tests_from_graph(
                changed_files, graph, repo_root
            )

            node_ids                        = get_pytest_node_ids(selected)
            selected_classes, class_test_count = get_summary_info(selected)

            # components for summary
            components = sorted(set(
                Path(f).stem for f in changed_files
            ))

            print()
            print("─" * 60)
            print(f"  Selected: {len(selected)} test files, "
                  f"{len(selected_classes)} classes, "
                  f"{len(node_ids)} test methods")
            print("─" * 60)

            for test_file, data in selected.items():
                triggered = ", ".join(data["triggered_by"])
                print(f"\n  📄 {test_file}")
                print(f"     ← triggered by: {triggered}")
                for cls_name, cls_data in data["classes"].items():
                    print(f"     • {cls_name} ({cls_data['test_count']} tests)")

            if not node_ids:
                print("\n  No runnable tests found for changed files.")
                print("  Possible reasons:")
                print("    • Changed files have no test coverage in graph")
                print("    • Run 'tselect build-graph' to rebuild graph")
                return

            _pretty_print_command(node_ids)
            cmd = build_pytest_command(node_ids)

        else:
            # ── LEGACY MODE ──
            print()
            print("  ⚠️  No dependency graph found.")
            print("  Run 'tselect build-graph' first for best results.")
            print("  Falling back to ownership.yaml + manual JSON...\n")

            ownership_path = repo_root / "ownership.yaml"
            json_path      = repo_root / "config" / "testSuiteTorchInductor.json"

            ownership = load_yaml(ownership_path)
            test_json = load_json(json_path)

            components = map_files_to_components(changed_files, ownership)
            selected_classes, class_test_count = collect_tests_from_components(
                components, test_json
            )

            total_tests = sum(class_test_count.values())
            node_ids    = selected_classes  # file::class format

            print("  Selected classes:")
            for cls in selected_classes:
                print(f"    • {cls}")
            print(f"\n  Total tests: {total_tests}")

            cmd = build_pytest_command_from_classes(list(selected_classes))

        # ── step 3: execute ──
        if not args.execute:
            print("  To execute, run:")
            print("    tselect run --execute")
            return

        print()
        print("─" * 60)
        print("  ► Executing tests...")
        print("─" * 60)
        print()

        logger.info("Executing pytest run")
        start_time = time.time()
        return_code, passed, failed, skipped = execute_command(cmd)
        duration = time.time() - start_time

        logger.info(
            f"Execution finished in {duration:.2f}s "
            f"(passed={passed}, failed={failed}, skipped={skipped})"
        )

        # save baseline if none exists
        if baseline_time is None:
            logger.info("No baseline found — saving current run as baseline")
            print("\n  No baseline recorded yet — saving this run as baseline.")
            cache["baseline_time"] = duration
            save_cache(repo_root, cache)
            baseline_time = duration

        # determine status
        if failed > 0 and passed > 0:
            status = "PARTIAL_FAIL"
        elif failed > 0:
            status = "FAILED"
        else:
            status = "PASSED"

        logger.info("Generating execution summary")

        generate_summary(
            components=components,
            total_tests=len(node_ids),
            duration=duration,
            status="PASSED" if return_code == 0 else status,
            baseline=baseline_time,
            passed=passed,
            failed=failed,
            skipped=skipped,
        )

    # ─────────────────────────────────────────────
    # BASELINE
    # ─────────────────────────────────────────────

    elif args.command == "baseline":
        logger.info("Running baseline detection")
        repo_root = Path.cwd()

        cmd = detect_baseline_command(repo_root)
        logger.debug(f"Baseline command: {cmd}")

        print("\n=== BASELINE COMMAND ===")
        print(" ".join(cmd))

        if not args.execute:
            print("\nRun with:")
            print("  tselect baseline --execute")
            return

        logger.info("Executing baseline run")

        start_time = time.time()
        return_code, passed, failed, skipped = execute_command(cmd)
        duration = time.time() - start_time

        logger.info(f"Baseline finished in {duration:.2f}s")

        cache                = load_cache(repo_root) or {}
        cache["baseline_time"] = duration
        save_cache(repo_root, cache)

        print(f"\n  ✅ Baseline recorded: {duration:.2f}s")
        print(f"  Passed: {passed} | Failed: {failed} | Skipped: {skipped}")

    else:
        parser.print_help()