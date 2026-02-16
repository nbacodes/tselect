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
from tselect.utils.logger import setup_logger

logger = setup_logger()


# ==========================================================
#  Pretty Printer
# ==========================================================
def pretty_print_command(cmd, hint):
    logger.debug("Preparing pretty print for pytest command")

    print("\n=== TSELECT COMMAND ===\n")

    pytest_parts = cmd[3:] if len(cmd) > 3 else cmd

    print("pytest \\")
    for part in pytest_parts:
        print(f"  {part} \\")

    print("\nTo execute:")
    print(hint)
    print()

# ==========================================================
# Main
# ==========================================================
def main():
    logger.debug("Initializing CLI parser")

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
        logger.info("Starting selective test run")

        repo_root = Path.cwd()
        logger.debug(f"Repository root: {repo_root}")


        ownership_path = repo_root / "ownership.yaml"
        json_path = repo_root / "config" / "testSuiteTorchInductor.json"
        
        logger.debug(f"Ownership path: {ownership_path}")
        logger.debug(f"Test mapping path: {json_path}")

        cache = load_cache(repo_root)
        baseline_time = cache.get("baseline_time")

        logger.debug(f"Loaded cache: {cache}")


        ownership = load_yaml(ownership_path)
        test_json = load_json(json_path)

        # ------------------------------------------------------
        # NEW: AUTO GIT DETECTION
        # ------------------------------------------------------
        if args.changed:
            logger.info("Using manually provided changed files")
            changed_files = args.changed
        else:
            logger.info("No --changed provided — detecting via git diff")
            changed_files = get_changed_files()

        logger.info(f"Detected {len(changed_files)} changed files")
        logger.debug(f"Changed files list: {changed_files}")

        print("\nDetected changed files:")
        for f in changed_files:
            print("-", f)

        # ------------------------------------------------------
        # COMPONENT MAPPING
        # ------------------------------------------------------
        
        logger.info("Mapping files to components")
        components = map_files_to_components(changed_files, ownership)

        # print("\nAffected components:", components)
        logger.info(f"Mapped to {len(components)} components")
        logger.debug(f"Components: {components}")
        logger.info("Collecting test classes from components")


        selected_classes, class_test_count = collect_tests_from_components(
            components,
            test_json,
        )

        logger.info(f"Selected {len(selected_classes)} test classes")
        logger.debug(f"Selected classes list: {selected_classes}")

        print("\nSelected classes:")
        for cls in selected_classes:
            print("-", cls)

        total_tests = sum(class_test_count.values())
        logger.info(f"Total mapped tests: {total_tests}")

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
            logger.info("Executing pytest run")

            start_time = time.time()
            return_code, passed, failed, skipped = execute_command(cmd)
            duration = time.time() - start_time

            logger.info(
                f"Execution finished in {duration:.2f}s "
                f"(passed={passed}, failed={failed}, skipped={skipped})"
            )

            if baseline_time is None:
                logger.info("No baseline found — saving current run as baseline")
                print("\nNo baseline found — saving this run as baseline.")
                cache["baseline_time"] = duration
                save_cache(repo_root, cache)
                baseline_time = duration

            if failed > 0 and passed > 0:
                status = "PARTIAL_FAIL"
            elif failed > 0:
                status = "FAILED"
            else:
                status = "PASSED"

            logger.debug(f"Final run status: {status}")
            logger.info("Generating execution summary")

            generate_summary(
                components=components,
                total_tests=total_tests,
                duration=duration,
                status="PASSED" if return_code == 0 else "FAILED",
                baseline=baseline_time,
                passed=passed,
                failed=failed,
                skipped=skipped,
            )


    # ==========================================================
    # BASELINE COMMAND
    # ==========================================================
    elif args.command == "baseline":
        logger.info("Running baseline detection")

        repo_root = Path.cwd()

        cmd = detect_baseline_command(repo_root)

        logger.debug(f"Baseline command detected: {cmd}")


        print("\n=== BASELINE COMMAND ===")
        print(" ".join(cmd))

        if not args.execute:
            print("\nUse:")
            print("tselect baseline --execute")
            return

        logger.info("Executing baseline run")

        start_time = time.time()
        return_code, passed, failed, skipped = execute_command(cmd)
        duration = time.time() - start_time

        logger.info(f"Baseline execution finished in {duration:.2f}s")

        cache = load_cache(repo_root) or {}
        cache["baseline_time"] = duration
        save_cache(repo_root, cache)

        print(f"\nBaseline recorded: {duration:.2f}s")

    else:
        parser.print_help()
