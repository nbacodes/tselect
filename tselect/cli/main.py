"""
cli/main.py
-----------
tselect CLI — zero user hardship design.

Commands:
  tselect init             → generate tselect.yaml for this repo
  tselect build-graph      → build dependency graph (run once)
  tselect run              → auto-detect changes, select + optionally run tests
  tselect run --execute    → select + run tests
  tselect run --coverage   → select + run tests + diff_cover confidence score
  tselect baseline --execute → record full suite baseline time
"""

import argparse
import time
from pathlib import Path

from tselect.utils.loader import load_yaml, load_json
from tselect.utils.config_loader import load_tselect_config, should_ignore_file
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
from tselect.core.diff_parser import get_changed_functions
from tselect.utils.logger import setup_logger

logger = setup_logger()


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def _is_graph_stale(graph: dict, rebuild_after_days: int) -> bool:
    built_at = graph.get("built_at")
    if not built_at:
        return False
    age_days = (time.time() - built_at) / 86400
    return age_days > rebuild_after_days


def _print_header(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def _print_section(title: str):
    print()
    print(f"  {'─' * 56}")
    print(f"  {title}")
    print(f"  {'─' * 56}")


def _filter_changed_files(changed_files: list, ignore_patterns: list) -> tuple[list, list]:
    actionable = []
    ignored    = []
    for f in changed_files:
        filename = Path(f).name
        if should_ignore_file(filename, ignore_patterns):
            ignored.append(f)
        else:
            actionable.append(f)
    return actionable, ignored


def _is_ai_enabled(config: dict) -> bool:
    return config.get("ai", {}).get("enabled", True)


def _run_ai_prefilter(selected, changed_files, repo_root, config):
    from tselect.ai.llm_client import LLMClient, LLMClientError
    from tselect.ai.pre_filter import PreFilter

    try:
        llm             = LLMClient(config)
        pf              = PreFilter(llm, config)
        changed_symbols = get_changed_functions(repo_root, changed_files)

        filtered, ai_decisions = pf.filter(
            selected        = selected,
            changed_files   = changed_files,
            changed_symbols = changed_symbols,
            repo_root       = repo_root,
        )
        return filtered, ai_decisions

    except LLMClientError as e:
        print(f"\n  ⚠️  AI pre-filter unavailable: {e}")
        print("  Proceeding with full rule-based selection.")
        return selected, []

    except Exception as e:
        print(f"\n  ⚠️  AI pre-filter error: {e}")
        print("  Proceeding with full rule-based selection.")
        return selected, []


def _run_ai_postanalysis(failed_tests, changed_files, repo_root, config,
                          passed, failed, skipped):
    from tselect.ai.llm_client import LLMClient, LLMClientError
    from tselect.ai.post_analyzer import PostAnalyzer

    if not failed_tests:
        return None

    try:
        llm             = LLMClient(config)
        analyzer        = PostAnalyzer(llm)
        changed_symbols = get_changed_functions(repo_root, changed_files)

        return analyzer.analyze(
            failed_tests    = failed_tests,
            changed_files   = changed_files,
            changed_symbols = changed_symbols,
            passed          = passed,
            failed          = failed,
            skipped         = skipped,
        )

    except LLMClientError as e:
        print(f"\n  ⚠️  AI post-analysis unavailable: {e}")
        return None

    except Exception as e:
        print(f"\n  ⚠️  AI post-analysis error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        prog="tselect",
        description="Targeted test selector — runs only what matters for your changes.",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command")

    # ── init ──
    subparsers.add_parser("init", help="Generate a starter tselect.yaml for this repo")

    # ── build-graph ──
    subparsers.add_parser("build-graph", help="Build dependency graph")

    # ── run ──
    run_parser = subparsers.add_parser("run", help="Select and optionally run tests")
    run_parser.add_argument(
        "--changed", nargs="+", required=False,
        help="Changed files (auto-detected from git diff if omitted)",
    )
    run_parser.add_argument(
        "--execute", action="store_true",
        help="Execute the selected tests",
    )
    run_parser.add_argument(
        "--coverage", action="store_true",
        help="Run with coverage and generate diff_cover confidence score",
    )

    # ── baseline ──
    baseline_parser = subparsers.add_parser("baseline", help="Record full suite baseline time")
    baseline_parser.add_argument("--execute", action="store_true")

    args = parser.parse_args()

    if args.debug:
        logger.setLevel("DEBUG")

    repo_root = Path.cwd()

    config          = load_tselect_config(repo_root)
    ignore_patterns = config["runner"]["ignore_changed_patterns"]
    rebuild_days    = config["graph"]["rebuild_after_days"]

    # ─────────────────────────────────────────────
    # INIT
    # ─────────────────────────────────────────────

    if args.command == "init":
        from tselect.core.init_command import run_init
        run_init(repo_root)

    # ─────────────────────────────────────────────
    # BUILD-GRAPH
    # ─────────────────────────────────────────────

    elif args.command == "build-graph":
        logger.info("Building dependency graph")

        from tselect.core.layout import RepoLayoutInferer
        from tselect.core.graph_builder import GraphBuilder, UnsupportedLanguageError

        _print_header("tselect — Building Dependency Graph")
        print(f"  Repo   : {repo_root}")

        if (repo_root / "tselect.yaml").exists():
            src = config["repo"].get("source_dirs", [])
            tst = config["repo"].get("test_dirs", [])
            print(f"  Source : {src}")
            print(f"  Tests  : {tst}")
        else:
            print("  Config : no tselect.yaml found")
            print("           Run 'tselect init' to generate one")

        print()
        print("► Scanning repository layout...")

        layout = RepoLayoutInferer(repo_root, config).infer()

        try:
            builder = GraphBuilder(layout)
        except UnsupportedLanguageError as e:
            print(e)
            return

        print(f"  Language  : {layout.language}")
        print(f"  Source    : {len(layout.source_files)} files")
        print(f"  Tests     : {len(layout.test_files)} files")
        print()

        print("► Building graph...")
        t_start = time.time()
        graph   = builder.build()
        t_total = time.time() - t_start

        path = builder.save(graph)

        reverse   = graph.get("full_reverse_graph", {})
        inventory = graph.get("test_inventory", {})

        print()
        print("  ✅ Graph built successfully")
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

    # ─────────────────────────────────────────────
    # RUN
    # ─────────────────────────────────────────────

    elif args.command == "run":
        logger.info("Starting selective test run")

        cache         = load_cache(repo_root)
        baseline_time = cache.get("baseline_time")

        _print_header("tselect — Targeted Test Selection")

        # step 1: get changed files
        if args.changed:
            logger.info("Using manually provided changed files")
            all_changed = args.changed
            source      = "manual"
        else:
            logger.info("Auto-detecting changed files via git diff")
            all_changed = get_changed_files()
            source      = "git diff"

        if not all_changed:
            print("\n  No changed files detected. Nothing to run.")
            return

        # step 2: filter
        changed_files, ignored_files = _filter_changed_files(all_changed, ignore_patterns)

        print(f"\n  Changed files ({source}):")
        for f in changed_files:
            print(f"    • {f}")

        if ignored_files:
            print(f"\n  Ignored (non-source) files:")
            for f in ignored_files:
                print(f"    ○ {f}  ← data/config file, skipped")

        if not changed_files:
            print("\n  All changed files are non-source (data/config).")
            print("  Nothing to test.")
            return

        # step 3: load graph
        graph_loader = GraphLoader(repo_root)
        ai_decisions = []
        ai_analysis  = None

        if graph_loader.exists():
            graph = graph_loader.load()

            if _is_graph_stale(graph, rebuild_days):
                print()
                print(f"  ⚠️  Dependency graph is older than {rebuild_days} days.")
                print("     Run 'tselect build-graph' to refresh it.")

            logger.info("Using auto-built dependency graph")

            selected, total_tests = select_tests_from_graph(
                changed_files, graph, repo_root
            )

            if _is_ai_enabled(config):
                selected, ai_decisions = _run_ai_prefilter(
                    selected      = selected,
                    changed_files = changed_files,
                    repo_root     = repo_root,
                    config        = config,
                )

            node_ids                           = get_pytest_node_ids(selected)
            selected_classes, class_test_count = get_summary_info(selected)
            components                         = sorted(set(Path(f).stem for f in changed_files))

            _print_section(
                f"Selected: {len(selected)} test files, "
                f"{len(selected_classes)} classes, "
                f"{len(node_ids)} test methods"
            )

            for test_file, data in selected.items():
                triggered = ", ".join(data["triggered_by"])
                print(f"\n  📄 {test_file}")
                print(f"     ← {triggered}")
                for cls_name, cls_data in data["classes"].items():
                    print(f"     • {cls_name} ({cls_data['test_count']} tests)")

            if not node_ids:
                print("\n  No runnable tests found for the changed files.")
                print("  Possible reasons:")
                print("    • Changed files have no tests in the dependency graph")
                print("    • Run 'tselect build-graph' to rebuild the graph")
                print("    • Check source_dirs in tselect.yaml covers these files")
                return

            cmd = build_pytest_command(node_ids, extra_args=config["runner"]["extra_args"])

        else:
            print()
            print("  ⚠️  No dependency graph found.")
            print("  Run 'tselect build-graph' first for auto mode.")
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
            node_ids    = selected_classes

            print("  Selected classes:")
            for cls in selected_classes:
                print(f"    • {cls}")
            print(f"\n  Total tests: {total_tests}")

            cmd = build_pytest_command_from_classes(list(selected_classes))

        # step 4: print command
        print()
        print("  " + "─" * 56)
        print("  PYTEST COMMAND")
        print("  " + "─" * 56)
        print()
        print("  pytest \\")
        for nid in node_ids[:5]:
            print(f"    {nid} \\")
        if len(node_ids) > 5:
            print(f"    ... and {len(node_ids) - 5} more")
        print()
        print("  To execute: tselect run --execute")

        if not args.execute:
            return

        # step 5: execute
        _print_section("► Executing tests...")
        print()

        # coverage setup — write .tselect_coveragerc, set env var, append cov args
        coverage_data = None
        if args.coverage:
            from tselect.reporting.coverage import prepare_coverage
            source_dir    = config.get("repo", {}).get("source_dirs", ["."])[0]
            coverage_args = prepare_coverage(repo_root, source_dir=source_dir)
            cmd           = cmd + coverage_args
            print(f"  Coverage enabled — source: {source_dir}")
            print()

        logger.info("Executing pytest run")
        start_time = time.time()
        return_code, passed, failed, skipped = execute_command(cmd)
        duration = time.time() - start_time

        logger.info(
            f"Execution finished in {duration:.2f}s "
            f"(passed={passed}, failed={failed}, skipped={skipped})"
        )

        # run diff_cover after pytest
        if args.coverage:
            from tselect.reporting.coverage import run_diff_cover
            compare_branch = config.get("coverage", {}).get("compare_branch", "origin/main")
            print()
            print("  ► Running diff-cover...")
            coverage_data = run_diff_cover(repo_root, compare_branch=compare_branch)

        if baseline_time is None:
            print("\n  No baseline recorded yet — saving this run as baseline.")
            cache["baseline_time"] = duration
            save_cache(repo_root, cache)
            baseline_time = duration

        status = (
            "COLLECTION_ERROR" if passed == 0 and failed == 0 and skipped == 0 and return_code != 0
            else "PARTIAL_FAIL" if failed > 0 and passed > 0
            else "FAILED"       if failed > 0
            else "PASSED"
        )

        if _is_ai_enabled(config) and failed > 0:
            print("\n  🤖 AI analyzing failures...")
            ai_analysis = _run_ai_postanalysis(
                failed_tests  = [],
                changed_files = changed_files,
                repo_root     = repo_root,
                config        = config,
                passed        = passed,
                failed        = failed,
                skipped       = skipped,
            )
            if ai_analysis:
                print()
                print(f"  Root cause : {ai_analysis.get('root_cause_file', 'unknown')}")
                print(f"  Symbol     : {ai_analysis.get('root_cause_symbol', 'unknown')}")
                print(f"  What broke : {ai_analysis.get('failure_pattern', '')}")
                print(f"  Explanation: {ai_analysis.get('explanation', '')}")
                print(f"  Suggestion : {ai_analysis.get('suggested_fix', '')}")

        if status == "COLLECTION_ERROR":
            print()
            print("  ❌ No tests ran — pytest encountered collection errors.")
            print("     This usually means some test files have broken imports")
            print("     (e.g. missing CUDA, distributed backend, or functorch).")
            print()
            print("  These files will be excluded from the graph on next rebuild.")
            print("  Run: tselect build-graph")
            print()

        ai_removed    = sum(1 for d in ai_decisions if not d["should_run"])
        kept          = [d for d in ai_decisions if d["should_run"]]
        ai_confidence = (
            sum(d["confidence"] for d in kept) / len(kept)
            if kept else None
        )

        generate_summary(
            components    = components,
            total_tests   = len(node_ids),
            duration      = duration,
            status        = "PASSED" if return_code == 0 else status,
            baseline      = baseline_time,
            passed        = passed,
            failed        = failed,
            skipped       = skipped,
            ai_decisions  = ai_decisions,
            ai_removed    = ai_removed,
            ai_confidence = ai_confidence,
            ai_analysis   = ai_analysis,
            coverage_data = coverage_data,
        )

        if config["ci"]["fail_on_test_failure"] and return_code != 0:
            raise SystemExit(return_code)

    # ─────────────────────────────────────────────
    # BASELINE
    # ─────────────────────────────────────────────

    elif args.command == "baseline":
        logger.info("Running baseline")
        cmd = detect_baseline_command(repo_root)

        print("\n=== BASELINE COMMAND ===")
        print(" ".join(cmd))

        if not args.execute:
            print("\nRun with: tselect baseline --execute")
            return

        start_time = time.time()
        return_code, passed, failed, skipped = execute_command(cmd)
        duration = time.time() - start_time

        cache                  = load_cache(repo_root) or {}
        cache["baseline_time"] = duration
        save_cache(repo_root, cache)

        print(f"\n  ✅ Baseline recorded: {duration:.2f}s")
        print(f"  Passed: {passed} | Failed: {failed} | Skipped: {skipped}")

    else:
        parser.print_help()