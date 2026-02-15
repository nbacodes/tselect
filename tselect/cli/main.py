import argparse
import time
from pathlib import Path

from tselect.utils.loader import load_yaml, load_json
from tselect.core.selector import (
    map_files_to_components,
    collect_tests_from_components,
)
from tselect.adapters.pytest_adapter import build_pytest_command, execute_command
from tselect.reporting.summary import generate_summary
from tselect.reporting.cache import load_cache, save_cache
from tselect.adapters.baseline_detector import detect_baseline_command
from tselect.adapters.git_adapter import get_changed_files


# ==========================================================
# Pretty Printer
# ==========================================================
def pretty_print_command(cmd, hint):
    print("\n=== TSELECT COMMAND ===\n")

    pytest_parts = cmd[3:] if len(cmd) > 3 else cmd

    print("pytest \\")
    for part in pytest_parts:
        print(f"  {part} \\")

    print("\nTo execute:")
    print(hint)
    print()


# ==========================================================
# MAIN
# ==========================================================
def main():
    parser = argparse.ArgumentParser(
        prog="tselect",
        description="Selective test runner",
    )

    subparsers = parser.add_subparsers(dest="command")

    # ----------------------------------------------------------
    # RUN COMMAND
    # ----------------------------------------------------------
    run_parser = subparsers.add_parser("run", help="Select tests to run")

    run_parser.add_argument(
        "--changed",
        nargs="+",
        required=False,
        help="List of changed files (optional — auto-detected via git if omitted)",
    )

    run_parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute selected tests",
    )

    # ----------------------------------------------------------
    # BASELINE COMMAND
    # ----------------------------------------------------------
    baseline_parser = subparsers.add_parser(
        "baseline", help="Create baseline timing"
    )

    baseline_parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute baseline tests",
    )

    args = parser.parse_args()

    # ==========================================================
    # RUN COMMAND LOGIC
    # ==========================================================
    if args.command == "run":

        repo_root = Path.cwd()

        ownership_path = repo_root / "ownership.yaml"
        json_path = repo_root / "config" / "testSuiteTorchInductor.json"

        cache = load_cache(repo_root)
        baseline_time = cache.get("baseline_time")

        ownership = load_yaml(ownership_path)
        test_json = load_json(json_path)

        # ------------------------------------------------------
        # NEW: AUTO GIT DETECTION
        # ------------------------------------------------------
        if args.changed:
            changed_files = args.changed
        else:
            print("No --changed provided — detecting via git diff...")
            changed_files = get_changed_files()

        print("\nDetected changed files:")
        for f in changed_files:
            print("-", f)

        # ------------------------------------------------------
        # COMPONENT MAPPING
        # ------------------------------------------------------
        components = map_files_to_components(changed_files, ownership)

        print("\nAffected components:", components)

        selected_classes, class_test_count = collect_tests_from_components(
            components,
            test_json,
        )

        print("\nSelected classes:")
        for cls in selected_classes:
            print("-", cls)

        total_tests = sum(class_test_count.values())
        print(f"\nTotal tests inside classes: {total_tests}")

        if not selected_classes:
            print("\nNo classes selected — falling back to full suite.")
            cmd = ["python", "-m", "pytest"]
        else:
            cmd = build_pytest_command(list(selected_classes))

        pretty_print_command(cmd, "tselect run --execute")

        # ------------------------------------------------------
        # EXECUTION
        # ------------------------------------------------------
        if args.execute:
            start_time = time.time()
            return_code = execute_command(cmd)
            duration = time.time() - start_time

            if baseline_time is None:
                print("\nNo baseline found — saving this run as baseline.")
                cache["baseline_time"] = duration
                save_cache(repo_root, cache)
                baseline_time = duration

            generate_summary(
                components=components,
                total_tests=total_tests,
                duration=duration,
                status="PASSED" if return_code == 0 else "FAILED",
                baseline=baseline_time,
            )

    # ==========================================================
    # BASELINE COMMAND
    # ==========================================================
    elif args.command == "baseline":

        repo_root = Path.cwd()

        cmd = detect_baseline_command(repo_root)

        print("\n=== BASELINE COMMAND ===")
        print(" ".join(cmd))

        if not args.execute:
            print("\nUse:")
            print("tselect baseline --execute")
            return

        start_time = time.time()
        return_code = execute_command(cmd)
        duration = time.time() - start_time

        cache = load_cache(repo_root) or {}
        cache["baseline_time"] = duration
        save_cache(repo_root, cache)

        print(f"\nBaseline recorded: {duration:.2f}s")

    else:
        parser.print_help()
