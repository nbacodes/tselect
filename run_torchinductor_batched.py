#!/usr/bin/env python3
"""
run_torchinductor_batched.py
-----------------------------
Runs test_torchinductor.py::CpuTests in batches of N,
appending coverage to .coverage_full after each batch.

This solves the SIGABRT crash caused by running all 953 tests at once.

Usage:
    cd /Users/nihalkumar/pytorch-pr-176888
    PYTHONPATH=/Users/nihalkumar/pytorch-pr-176888 \
    python /Users/nihalkumar/Desktop/nbaworks/tselect/run_torchinductor_batched.py

    # Resume if interrupted:
    python run_torchinductor_batched.py --resume

    # Adjust batch size (default 30):
    python run_torchinductor_batched.py --batch-size 20
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PYTORCH_ROOT   = Path("/Users/nihalkumar/pytorch-pr-176888")
COVERAGERC     = PYTORCH_ROOT / ".coveragerc_full"
COVERAGE_DATA  = PYTORCH_ROOT / ".coverage_full"
COVERAGE_XML   = PYTORCH_ROOT / "coverage_full.xml"
PROGRESS_FILE  = PYTORCH_ROOT / ".torchinductor_batch_progress.json"

TEST_CLASSES = [
    "test/inductor/test_torchinductor.py::CpuTests",
    # add GPUTests here if needed later
]


def collect_test_ids(cls: str) -> list:
    """Collect all test node IDs for a class."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PYTORCH_ROOT)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", cls,
         "--collect-only", "-q",
         "--continue-on-collection-errors"],
        cwd=str(PYTORCH_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )

    ids = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if "::" in line and not line.startswith("=") and not line.startswith("no tests"):
            ids.append(line)
    return ids


def run_batch(batch: list, batch_num: int, total_batches: int) -> str:
    """Run a batch of test IDs. Returns status string."""
    env = os.environ.copy()
    env["PYTHONPATH"]      = str(PYTORCH_ROOT)
    env["COVERAGE_RCFILE"] = str(COVERAGERC)
    # tell coverage.py to flush on exit even on crash
    env["COVERAGE_PROCESS_START"] = str(COVERAGERC)

    cmd = [
        sys.executable, "-m", "pytest",
        "--tb=no",
        "-q",
        "--no-header",
        "--timeout=120",
        "--continue-on-collection-errors",
        "-p", "no:randomly",
        "--cov=torch/_inductor",
        "--cov-append",
        "--cov-report=",
        f"--cov-config={COVERAGERC}",
    ] + batch

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PYTORCH_ROOT),
            env=env,
            capture_output=True,
            text=True,
        )
    except Exception as e:
        return f"exception: {e}"

    rc = result.returncode
    if rc in (0, 1):
        summary = [l for l in result.stdout.splitlines() if "passed" in l or "failed" in l]
        summary_str = summary[-1] if summary else "done"
        return f"ok  ({summary_str.strip()})"
    elif rc == 3:
        return "collection_error"
    elif rc == 4:
        return "no_tests"
    else:
        return f"error_{rc}"


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"done_batches": [], "failed_batches": []}


def save_progress(progress: dict):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def generate_xml():
    print("\n► Regenerating coverage_full.xml...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PYTORCH_ROOT)
    subprocess.run(
        [sys.executable, "-m", "coverage", "xml",
         f"--rcfile={COVERAGERC}",
         "-o", str(COVERAGE_XML)],
        cwd=str(PYTORCH_ROOT),
        env=env,
    )
    print(f"✅  coverage_full.xml updated → {COVERAGE_XML}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=30,
                        help="Tests per batch (default: 30)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already completed batches")
    args = parser.parse_args()

    print(f"\n{'='*62}")
    print(f"  test_torchinductor.py — Batched Coverage Runner")
    print(f"{'='*62}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Resume     : {args.resume}")
    print()

    # ── Collect all test IDs ──
    print("► Collecting test IDs...")
    all_ids = []
    for cls in TEST_CLASSES:
        ids = collect_test_ids(cls)
        print(f"  {cls}: {len(ids)} tests")
        all_ids.extend(ids)

    if not all_ids:
        print("❌  No tests collected. Check PYTHONPATH.")
        return

    # ── Split into batches ──
    batches = [
        all_ids[i:i+args.batch_size]
        for i in range(0, len(all_ids), args.batch_size)
    ]
    total = len(batches)
    print(f"  Total batches  : {total}  ({len(all_ids)} tests ÷ {args.batch_size})")

    # ── Load progress ──
    progress = load_progress() if args.resume else {"done_batches": [], "failed_batches": []}
    done_set = set(progress["done_batches"])

    print(f"\n{'─'*62}")
    start = time.time()

    for i, batch in enumerate(batches, 1):
        if i in done_set:
            print(f"  [{i:3}/{total}]  SKIP (already done)")
            continue

        print(f"  [{i:3}/{total}]  {len(batch)} tests ...", end="", flush=True)
        t0     = time.time()
        status = run_batch(batch, i, total)
        elapsed = time.time() - t0

        if status.startswith("ok"):
            print(f"  ✅  {elapsed:.0f}s  {status}")
            progress["done_batches"].append(i)
        else:
            print(f"  ⚠️   {elapsed:.0f}s  {status}  (skipping)")
            progress["failed_batches"].append(i)
            progress["done_batches"].append(i)  # mark done so resume skips it

        save_progress(progress)

    elapsed_total = time.time() - start
    done    = len(progress["done_batches"])
    failed  = len(progress["failed_batches"])

    print(f"\n{'─'*62}")
    print(f"  ✅  Completed : {done - failed} / {total} batches")
    print(f"  ⚠️   Failed    : {failed} batches (skipped)")
    print(f"  Total time   : {elapsed_total/60:.1f} min")

    generate_xml()

    print(f"\n  Now run:")
    print(f"  python /Users/nihalkumar/Desktop/nbaworks/tselect/compare_coverage.py \\")
    print(f"    --changed torch/_inductor/lowering.py")
    print()
    if failed > total * 0.3:
        print(f"  ⚠️  More than 30% of batches failed.")
        print(f"  Try smaller batches:")
        print(f"  python run_torchinductor_batched.py --batch-size 10 --resume")


if __name__ == "__main__":
    main()