#!/usr/bin/env python3
"""
measure_inductor_coverage.py
-----------------------------
Uses tselect's graph_selector to select tests, filters to inductor-only,
runs them with --cov, then parses coverage.xml and prints a clean report.

Usage (run from your pytorch repo root):
    cd /Users/nihalkumar/pytorch-pr-176888
    PYTHONPATH=/Users/nihalkumar/pytorch-pr-176888 \
    python /path/to/measure_inductor_coverage.py \
      --changed torch/_inductor/lowering.py \
      [--changed torch/_inductor/another.py ...] \
      [--dry-run]        # just show selected tests, don't run pytest
      [--timeout 120]    # per-test timeout in seconds (default: 120)
      [--batch-size 20]  # run N node_ids per pytest invocation (default: all-at-once)
"""

import argparse
import os
import sys
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

# ──────────────────────────────────────────────────────────────
# 0. Setup: make sure tselect is importable
# ──────────────────────────────────────────────────────────────

TSELECT_ROOT = Path("/Users/nihalkumar/Desktop/nbaworks/tselect")
PYTORCH_ROOT = Path("/Users/nihalkumar/pytorch-pr-176888")
COVERAGE_XML  = PYTORCH_ROOT / "coverage.xml"
COVERAGERC    = PYTORCH_ROOT / ".coveragerc_inductor"


def ensure_tselect_importable():
    tselect_path = str(TSELECT_ROOT)
    if tselect_path not in sys.path:
        sys.path.insert(0, tselect_path)


# ──────────────────────────────────────────────────────────────
# 1. Load graph + run tselect selection
# ──────────────────────────────────────────────────────────────

def load_graph(repo_root: Path) -> dict:
    ensure_tselect_importable()
    from tselect.core.graph_loader import GraphLoader

    loader = GraphLoader(repo_root)
    if not loader.exists():
        print("❌  No dependency graph found.")
        print("    Run: cd {repo_root} && tselect build-graph")
        sys.exit(1)

    graph = loader.load()
    print(f"✅  Graph loaded  (schema: {graph.get('schema_version', '?')})")
    return graph


def select_inductor_tests(changed_files: list, graph: dict, repo_root: Path) -> list:
    ensure_tselect_importable()
    from tselect.core.graph_selector import select_tests_from_graph, get_pytest_node_ids

    print(f"\n► Running tselect for {len(changed_files)} changed file(s):")
    for f in changed_files:
        print(f"    • {f}")

    selected, total_methods = select_tests_from_graph(changed_files, graph, repo_root)

    all_node_ids = get_pytest_node_ids(selected)
    print(f"\n  tselect total selected : {len(all_node_ids)} test methods across {len(selected)} files")

    # ── Filter: keep only test/inductor/ tests ──
    inductor_ids = [
        nid for nid in all_node_ids
        if nid.startswith("test/inductor/") or "/test_inductor" in nid or "inductor" in nid.split("::")[0]
    ]

    print(f"  After inductor filter  : {len(inductor_ids)} test methods\n")

    if not inductor_ids:
        print("⚠️  No inductor tests selected. Possible reasons:")
        print("    • Changed file doesn't link to any inductor test in the graph")
        print("    • Run 'tselect build-graph' to refresh")
        sys.exit(0)

    # Print a preview
    print("  Sample selected tests (first 10):")
    for nid in inductor_ids[:10]:
        print(f"    {nid}")
    if len(inductor_ids) > 10:
        print(f"    ... and {len(inductor_ids) - 10} more")

    return inductor_ids


# ──────────────────────────────────────────────────────────────
# 2. Write .coveragerc so coverage measures only torch/_inductor
# ──────────────────────────────────────────────────────────────

def write_coveragerc(repo_root: Path):
    content = f"""\
[run]
source = {repo_root}/torch/_inductor
omit =
    */test*
    *__pycache__*
    */setup.py
branch = true
parallel = false
data_file = {repo_root}/.coverage_inductor

[report]
show_missing = true

[xml]
output = {COVERAGE_XML}
"""
    COVERAGERC.write_text(content)
    print(f"✅  .coveragerc written → {COVERAGERC}")


# ──────────────────────────────────────────────────────────────
# 3. Run pytest with --cov
# ──────────────────────────────────────────────────────────────

def run_pytest(node_ids: list, timeout: int, batch_size: int, repo_root: Path):
    """
    Run pytest from repo_root so relative test paths resolve correctly.
    Uses --continue-on-collection-errors so one broken import
    doesn't kill the entire run.
    """
    # Split into batches to survive partial crashes
    batches = (
        [node_ids]
        if batch_size <= 0
        else [node_ids[i:i+batch_size] for i in range(0, len(node_ids), batch_size)]
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    env["COVERAGE_RCFILE"] = str(COVERAGERC)

    run_dir = repo_root  # tests are relative paths from pytorch root

    total_passed = total_failed = total_skipped = 0

    for batch_num, batch in enumerate(batches, 1):
        print(f"\n  ── Batch {batch_num}/{len(batches)}  ({len(batch)} tests) ──")

        cmd = [
            sys.executable, "-m", "pytest",
            "--tb=no",                          # no tracebacks (speed)
            "-q",                               # quiet
            "--no-header",
            f"--timeout={timeout}",             # per-test timeout
            "--continue-on-collection-errors",  # don't die on broken imports
            "-p", "no:randomly",                # stable ordering
            # coverage flags
            "--cov=torch/_inductor",
            "--cov-append",                     # accumulate across batches
            "--cov-report=",                    # no terminal report (xml at end)
            f"--cov-config={COVERAGERC}",
        ] + batch

        result = subprocess.run(
            cmd,
            cwd=str(run_dir),
            env=env,
            capture_output=False,   # let output stream live
        )

        # parse counts from return code signals
        if result.returncode == 0:
            total_passed += len(batch)
        elif result.returncode == 1:
            # some tests failed — we don't know exact split without parsing output
            pass
        elif result.returncode == 2:
            print(f"  ⚠️  Batch {batch_num}: pytest interrupted")
        elif result.returncode == 3:
            print(f"  ⚠️  Batch {batch_num}: collection error (continuing)")
        elif result.returncode == 4:
            print(f"  ⚠️  Batch {batch_num}: no tests collected")
        elif result.returncode == 5:
            print(f"  ℹ️   Batch {batch_num}: no tests ran (all deselected/skipped)")

    # write final XML report
    print("\n► Generating coverage.xml...")
    xml_cmd = [
        sys.executable, "-m", "coverage", "xml",
        f"--rcfile={COVERAGERC}",
        "-o", str(COVERAGE_XML),
    ]
    subprocess.run(xml_cmd, cwd=str(repo_root), env=env)
    print(f"✅  coverage.xml → {COVERAGE_XML}")


# ──────────────────────────────────────────────────────────────
# 4. Parse coverage.xml and print report
# ──────────────────────────────────────────────────────────────

def parse_and_report(coverage_xml: Path, node_ids: list):
    if not coverage_xml.exists():
        print(f"\n❌  coverage.xml not found at {coverage_xml}")
        print("    Pytest may have crashed before generating it.")
        print("    Try running with --batch-size 10 to isolate the crash.")
        return

    print(f"\n{'='*62}")
    print("  📊  COVERAGE REPORT  —  torch/_inductor")
    print(f"{'='*62}")

    tree = ET.parse(coverage_xml)
    root = tree.getroot()

    # ── Overall stats from root attributes ──
    lines_valid   = int(root.get("lines-valid",    0))
    lines_covered = int(root.get("lines-covered",  0))
    line_rate     = float(root.get("line-rate",     0))
    branches_valid   = int(root.get("branches-valid",   0))
    branches_covered = int(root.get("branches-covered", 0))

    pct_lines    = line_rate * 100
    pct_branches = (branches_covered / branches_valid * 100) if branches_valid else 0

    print(f"\n  Line coverage   : {lines_covered:>6} / {lines_valid:<6} = {pct_lines:5.1f}%")
    print(f"  Branch coverage : {branches_covered:>6} / {branches_valid:<6} = {pct_branches:5.1f}%")

    # ── Per-file breakdown ──
    print(f"\n  {'─'*58}")
    print(f"  {'File':<45} {'Line%':>6}  {'Status'}")
    print(f"  {'─'*58}")

    zero_coverage = []
    partial_coverage = []
    full_coverage = []

    for pkg in root.iter("package"):
        for cls in pkg.iter("class"):
            fname     = cls.get("filename", "?")
            rate      = float(cls.get("line-rate", 0))
            pct       = rate * 100
            lines     = cls.findall(".//line")
            total_l   = len(lines)
            covered_l = sum(1 for l in lines if int(l.get("hits", 0)) > 0)

            if pct == 0:
                status = "❌ ZERO"
                zero_coverage.append((fname, total_l))
            elif pct < 100:
                status = f"⚠️  {pct:4.1f}%"
                partial_coverage.append((fname, pct, covered_l, total_l))
            else:
                status = "✅ 100%"
                full_coverage.append(fname)

            short = fname[-43:] if len(fname) > 43 else fname
            print(f"  {short:<45} {pct:5.1f}%  {status}")

    # ── Summary ──
    print(f"\n  {'─'*58}")
    print(f"  ✅  Full coverage  : {len(full_coverage)} files")
    print(f"  ⚠️   Partial        : {len(partial_coverage)} files")
    print(f"  ❌  Zero coverage  : {len(zero_coverage)} files")

    # ── Missing / zero-hit files ──
    if zero_coverage:
        print(f"\n  {'─'*58}")
        print(f"  FILES WITH ZERO COVERAGE (not touched by selected tests):")
        for fname, n_lines in sorted(zero_coverage, key=lambda x: -x[1]):
            print(f"    {fname}  ({n_lines} lines)")

    # ── tselect test count ──
    print(f"\n  {'─'*58}")
    print(f"  tselect selected   : {len(node_ids)} test method(s)")
    print(f"  Inductor source    : {lines_valid} traceable lines")
    print(f"  Covered by tselect : {lines_covered} lines  ({pct_lines:.1f}%)")
    print(f"  Missing            : {lines_valid - lines_covered} lines")
    print(f"{'='*62}\n")


# ──────────────────────────────────────────────────────────────
# 5. CLI entrypoint
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Measure inductor coverage via tselect")
    parser.add_argument(
        "--changed", nargs="+", required=True,
        help="Changed files relative to repo root (e.g. torch/_inductor/lowering.py)",
    )
    parser.add_argument("--dry-run",    action="store_true", help="Only show selected tests, skip pytest")
    parser.add_argument("--timeout",    type=int, default=120,  help="Per-test timeout seconds (default: 120)")
    parser.add_argument("--batch-size", type=int, default=0,    help="Run N tests per pytest call (0 = all at once)")
    parser.add_argument("--repo",       type=str, default=str(PYTORCH_ROOT), help="Path to pytorch repo root")
    args = parser.parse_args()

    repo_root = Path(args.repo)

    print(f"\n{'='*62}")
    print("  tselect  ×  coverage  —  Inductor")
    print(f"{'='*62}")
    print(f"  Repo   : {repo_root}")
    print(f"  Timeout: {args.timeout}s per test")

    # ── Step 1: load graph ──
    graph = load_graph(repo_root)

    # ── Step 2: select tests ──
    node_ids = select_inductor_tests(args.changed, graph, repo_root)

    if args.dry_run:
        print("\n  [--dry-run] Skipping pytest execution.")
        print("  All selected node IDs:")
        for nid in node_ids:
            print(f"    {nid}")
        return

    # ── Step 3: write coveragerc ──
    write_coveragerc(repo_root)

    # ── Step 4: run pytest ──
    run_pytest(node_ids, timeout=args.timeout, batch_size=args.batch_size, repo_root=repo_root)

    # ── Step 5: report ──
    parse_and_report(COVERAGE_XML, node_ids)


if __name__ == "__main__":
    main()