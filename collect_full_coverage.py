#!/usr/bin/env python3
"""
collect_full_coverage.py  —  Phase 1
--------------------------------------
Runs ALL test/inductor/ tests one file at a time, accumulating coverage
into coverage_full.xml. Skips files that crash/timeout gracefully.

This is the "ground truth" baseline — run once, cache the result.

Usage:
    cd /Users/nihalkumar/pytorch-pr-176888
    PYTHONPATH=/Users/nihalkumar/pytorch-pr-176888 \
    python /Users/nihalkumar/Desktop/nbaworks/tselect/collect_full_coverage.py

    # Resume interrupted run (skips already-done files):
    python collect_full_coverage.py --resume

    # Test with just 3 files first:
    python collect_full_coverage.py --limit 3
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PYTORCH_ROOT  = Path("/Users/nihalkumar/pytorch-pr-176888")
COVERAGE_XML  = PYTORCH_ROOT / "coverage_full.xml"
COVERAGERC    = PYTORCH_ROOT / ".coveragerc_full"
COVERAGE_DATA = PYTORCH_ROOT / ".coverage_full"
PROGRESS_FILE = PYTORCH_ROOT / ".full_coverage_progress.json"  # for --resume


# ──────────────────────────────────────────────────────────────
# Write .coveragerc
# ──────────────────────────────────────────────────────────────

def write_coveragerc():
    content = f"""\
[run]
source = {PYTORCH_ROOT}/torch/_inductor
omit =
    */test*
    *__pycache__*
    */setup.py
branch = true
parallel = false
data_file = {COVERAGE_DATA}

[report]
show_missing = false

[xml]
output = {COVERAGE_XML}
"""
    COVERAGERC.write_text(content)
    print(f"✅  .coveragerc → {COVERAGERC}")


# ──────────────────────────────────────────────────────────────
# Discover test files
# ──────────────────────────────────────────────────────────────

def discover_test_files(repo_root: Path) -> list:
    test_dir = repo_root / "test" / "inductor"
    files = sorted(test_dir.glob("test_*.py"))
    print(f"✅  Discovered {len(files)} test files in test/inductor/")
    return [str(f.relative_to(repo_root)) for f in files]


# ──────────────────────────────────────────────────────────────
# Progress tracking (for --resume)
# ──────────────────────────────────────────────────────────────

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"done": [], "skipped": [], "failed": []}


def save_progress(progress: dict):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


# ──────────────────────────────────────────────────────────────
# Run one test file
# ──────────────────────────────────────────────────────────────

def run_one_file(test_file: str, repo_root: Path, timeout_seconds: int) -> str:
    """
    Returns: 'passed' | 'failed' | 'collection_error' | 'timeout' | 'no_tests'
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    env["COVERAGE_RCFILE"] = str(COVERAGERC)

    cmd = [
        sys.executable, "-m", "pytest",
        test_file,
        "--tb=no",
        "-q",
        "--no-header",
        f"--timeout={timeout_seconds}",
        "--continue-on-collection-errors",
        "-p", "no:randomly",
        # coverage
        "--cov=torch/_inductor",
        "--cov-append",
        "--cov-report=",                   # no terminal report
        f"--cov-config={COVERAGERC}",
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
        )

        rc = result.returncode
        if rc == 0:
            return "passed"
        elif rc == 1:
            return "failed"
        elif rc == 2:
            return "interrupted"
        elif rc == 3:
            return "collection_error"
        elif rc == 4:
            return "no_tests"
        elif rc == 5:
            return "no_tests"
        else:
            return f"error_{rc}"

    except subprocess.TimeoutExpired:
        return "timeout"


# ──────────────────────────────────────────────────────────────
# Generate final XML
# ──────────────────────────────────────────────────────────────

def generate_xml(repo_root: Path):
    print("\n► Generating coverage_full.xml...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)

    result = subprocess.run(
        [
            sys.executable, "-m", "coverage", "xml",
            f"--rcfile={COVERAGERC}",
            "-o", str(COVERAGE_XML),
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print(f"✅  coverage_full.xml → {COVERAGE_XML}")
    else:
        print(f"❌  coverage xml failed:\n{result.stderr}")


# ──────────────────────────────────────────────────────────────
# Print summary
# ──────────────────────────────────────────────────────────────

def print_summary(progress: dict, total: int, elapsed: float):
    done     = len(progress["done"])
    failed   = len(progress["failed"])
    skipped  = len(progress["skipped"])

    print(f"\n{'='*60}")
    print(f"  Phase 1 — Full Coverage Collection  COMPLETE")
    print(f"{'='*60}")
    print(f"  Total test files   : {total}")
    print(f"  ✅ Contributed     : {done}")
    print(f"  ⚠️  Soft failures   : {failed}  (still ran, some tests failed)")
    print(f"  ❌ Skipped/crashed : {skipped}")
    print(f"  Total time         : {elapsed/60:.1f} min")
    print(f"  Output             : {COVERAGE_XML}")
    print(f"\n  Now run measure_inductor_coverage.py for Phase 2 comparison.")
    print(f"{'='*60}\n")

    if progress["skipped"]:
        print("  Skipped files (crashed/timeout/no tests):")
        for f, reason in progress["skipped"]:
            print(f"    {reason:20s}  {f}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Collect full inductor coverage (Phase 1)")
    parser.add_argument("--resume",  action="store_true", help="Skip already-completed files")
    parser.add_argument("--limit",   type=int, default=0, help="Only run first N files (for testing)")
    parser.add_argument("--timeout", type=int, default=120, help="Per-test timeout in seconds (default: 120)")
    parser.add_argument("--repo",    type=str, default=str(PYTORCH_ROOT))
    args = parser.parse_args()

    repo_root = Path(args.repo)

    print(f"\n{'='*60}")
    print(f"  tselect — Phase 1: Full Inductor Coverage")
    print(f"{'='*60}")
    print(f"  Repo    : {repo_root}")
    print(f"  Output  : {COVERAGE_XML}")
    print(f"  Timeout : {args.timeout}s per test")
    if args.resume:
        print(f"  Mode    : RESUME (skipping completed files)")
    print()

    write_coveragerc()

    test_files = discover_test_files(repo_root)
    if args.limit > 0:
        test_files = test_files[:args.limit]
        print(f"  ⚠️  --limit {args.limit}: only running first {args.limit} files")

    # Load progress for resume
    progress = load_progress() if args.resume else {"done": [], "skipped": [], "failed": []}

    # Erase old coverage data if NOT resuming
    if not args.resume and COVERAGE_DATA.exists():
        COVERAGE_DATA.unlink()
        print(f"  🗑️   Cleared old coverage data")

    already_done = set(progress["done"]) | {f for f, _ in progress["skipped"]}

    total     = len(test_files)
    start     = time.time()

    print(f"\n{'─'*60}")

    for i, test_file in enumerate(test_files, 1):
        if test_file in already_done:
            print(f"  [{i:3}/{total}]  SKIP (already done)  {test_file}")
            continue

        print(f"  [{i:3}/{total}]  {test_file} ...", end="", flush=True)
        t0     = time.time()
        status = run_one_file(test_file, repo_root, args.timeout)
        elapsed = time.time() - t0

        if status == "passed":
            print(f"  ✅  {elapsed:.0f}s")
            progress["done"].append(test_file)
        elif status == "failed":
            print(f"  ⚠️  some tests failed  {elapsed:.0f}s  (coverage still collected)")
            progress["done"].append(test_file)
            progress["failed"].append(test_file)
        elif status in ("collection_error", "timeout", "no_tests", "interrupted") or status.startswith("error_"):
            print(f"  ❌  {status}  {elapsed:.0f}s  (skipped)")
            progress["skipped"].append((test_file, status))
        else:
            print(f"  ?  {status}")
            progress["skipped"].append((test_file, status))

        save_progress(progress)

    generate_xml(repo_root)
    print_summary(progress, total, time.time() - start)


if __name__ == "__main__":
    main()