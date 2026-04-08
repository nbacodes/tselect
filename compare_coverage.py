#!/usr/bin/env python3
"""
compare_coverage.py  —  Phase 3: The Report
---------------------------------------------
Compares coverage_full.xml (ground truth) vs coverage.xml (tselect)
and produces a clean, presentable evaluation report.

Usage:
    cd /Users/nihalkumar/pytorch-pr-176888
    python /Users/nihalkumar/Desktop/nbaworks/tselect/compare_coverage.py \
      --changed torch/_inductor/lowering.py \
      --tselect-tests 12 \
      --full-tests 800
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

PYTORCH_ROOT    = Path("/Users/nihalkumar/pytorch-pr-176888")
FULL_XML        = PYTORCH_ROOT / "coverage_full.xml"
TSELECT_XML     = PYTORCH_ROOT / "coverage.xml"


# ──────────────────────────────────────────────────────────────
# Parse XML → {filename: set of covered line numbers}
# ──────────────────────────────────────────────────────────────

def parse_coverage(xml_path: Path) -> dict:
    """Returns {filename: set(covered_line_numbers)}"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    result = {}
    for cls in root.iter("class"):
        fname = cls.get("filename", "")
        covered = set()
        all_lines = set()
        for line in cls.findall(".//line"):
            lineno = int(line.get("number", 0))
            hits   = int(line.get("hits", 0))
            all_lines.add(lineno)
            if hits > 0:
                covered.add(lineno)
        result[fname] = {
            "covered":   covered,
            "all_lines": all_lines,
        }
    return result


# ──────────────────────────────────────────────────────────────
# Get function name for a line number (best effort via AST)
# ──────────────────────────────────────────────────────────────

def get_changed_lines(repo_root: Path, changed_file: str) -> set:
    """Get exact line numbers changed in the PR using git diff."""
    import subprocess
    result = subprocess.run(
        ["git", "diff", "HEAD~1", "--unified=0", "--", changed_file],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    changed_lines = set()
    for line in result.stdout.splitlines():
        if line.startswith("@@"):
            # e.g. @@ -5152,11 +5152,6 @@
            import re
            m = re.search(r'\+(\d+)(?:,(\d+))?', line)
            if m:
                start = int(m.group(1))
                count = int(m.group(2)) if m.group(2) else 1
                for i in range(start, start + count):
                    changed_lines.add(i)
    return changed_lines


def get_function_map(filepath: Path) -> dict:
    """Returns {line_number: function_name}"""
    try:
        import ast
        source = filepath.read_text()
        tree   = ast.parse(source)
        mapping = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for lineno in range(node.lineno, node.end_lineno + 1):
                    mapping[lineno] = node.name
        return mapping
    except Exception:
        return {}


# ──────────────────────────────────────────────────────────────
# Print helpers
# ──────────────────────────────────────────────────────────────

def bar(pct: float, width: int = 30) -> str:
    filled = int(width * pct / 100)
    return f"[{'█' * filled}{'░' * (width - filled)}]"


def color_pct(pct: float) -> str:
    if pct >= 90:
        return f"🟢 {pct:.1f}%"
    elif pct >= 70:
        return f"🟡 {pct:.1f}%"
    else:
        return f"🔴 {pct:.1f}%"


# ──────────────────────────────────────────────────────────────
# Main report
# ──────────────────────────────────────────────────────────────

def run_report(changed_files: list, tselect_test_count: int, full_test_count: int, repo_root: Path = PYTORCH_ROOT):

    print(f"\n{'═'*62}")
    print(f"  tselect  —  Evaluation Report")
    print(f"{'═'*62}")

    # ── Load both XMLs ──
    if not FULL_XML.exists():
        print(f"❌  {FULL_XML} not found. Run collect_full_coverage.py first.")
        return
    if not TSELECT_XML.exists():
        print(f"❌  {TSELECT_XML} not found. Run measure_inductor_coverage.py first.")
        return

    full_cov    = parse_coverage(FULL_XML)
    tselect_cov = parse_coverage(TSELECT_XML)

    print(f"\n  PR changed files  : {len(changed_files)}")
    for f in changed_files:
        print(f"    • {f}")

    print(f"\n  Full suite tests  : {full_test_count}")
    print(f"  tselect tests     : {tselect_test_count}")
    reduction = (full_test_count - tselect_test_count) / full_test_count * 100
    print(f"  Tests eliminated  : {full_test_count - tselect_test_count}  ({reduction:.1f}% fewer tests)")

    # ── Per changed file analysis ──
    print(f"\n{'─'*62}")
    print(f"  COVERAGE RECALL  —  Changed Files Only")
    print(f"{'─'*62}")

    total_full_covered    = 0
    total_tselect_covered = 0
    total_missed_lines    = 0

    for changed_file in changed_files:
        short = Path(changed_file).name  # "lowering.py"

        # Print all matching keys found so we can debug
        full_matches = [k for k in full_cov if k.endswith(short) or short in k]
        tsel_matches = [k for k in tselect_cov if k.endswith(short) or short in k]

        # Pick the key with the most lines (the real lowering.py, not a stub)
        full_key = max(full_matches, key=lambda k: len(full_cov[k]["all_lines"]), default=None)
        tsel_key = max(tsel_matches, key=lambda k: len(tselect_cov[k]["all_lines"]), default=None)

        if not full_key:
            print(f"\n  ⚠️  {changed_file} not found in coverage_full.xml")
            print(f"      (file may not have been exercised by any test)")
            continue

        full_lines    = full_cov[full_key]["covered"]
        tselect_lines = tselect_cov.get(tsel_key or "", {}).get("covered", set()) if tsel_key else set()
        all_lines     = full_cov[full_key]["all_lines"]

        missed_lines  = full_lines - tselect_lines
        caught_lines  = full_lines & tselect_lines
        recall = len(caught_lines) / len(full_lines) * 100 if full_lines else 0.0

        # ── Changed lines from git diff (the REAL metric) ──
        changed_lines        = get_changed_lines(repo_root, changed_file)
        changed_covered_full = changed_lines & full_lines
        changed_covered_tsel = changed_lines & tselect_lines
        changed_recall       = (
            len(changed_covered_tsel) / len(changed_covered_full) * 100
            if changed_covered_full else 0.0
        )

        total_full_covered    += len(full_lines)
        total_tselect_covered += len(caught_lines)
        total_missed_lines    += len(missed_lines)

        print(f"\n  📄 {changed_file}")
        print(f"     Total lines in file         : {len(all_lines)}")
        print()
        print(f"     ── Changed lines (PR diff) ──────────────────────────")
        print(f"     Lines changed in PR          : {len(changed_lines)}")
        print(f"     Changed lines covered (full) : {len(changed_covered_full)}")
        print(f"     Changed lines covered (tsel) : {len(changed_covered_tsel)}")
        print()
        print(f"     PR Recall  {bar(changed_recall)}  {color_pct(changed_recall)}")
        print(f"     ← 'Did tselect cover the lines we actually changed?'")
        print()
        print(f"     ── Full file (context) ──────────────────────────────")
        print(f"     Covered by full suite        : {len(full_lines)}")
        print(f"     Covered by tselect           : {len(tselect_lines)}")
        print(f"     Missed by tselect            : {len(missed_lines)}")
        print()
        print(f"     File Recall  {bar(recall)}  {color_pct(recall)}")

        # ── Missed lines broken down by function (top 10 only) ──
        if missed_lines:
            src_path = PYTORCH_ROOT / changed_file
            fn_map   = get_function_map(src_path) if src_path.exists() else {}

            missed_by_fn = {}
            for lineno in sorted(missed_lines):
                fn = fn_map.get(lineno, "<module level>")
                missed_by_fn.setdefault(fn, []).append(lineno)

            # get changed function names to highlight them
            changed_fns = set(fn_map.get(l, "") for l in changed_lines)

            print(f"\n     Top missed functions (full file context):")
            for fn, lines in sorted(missed_by_fn.items(), key=lambda x: -len(x[1]))[:10]:
                marker = " ⬅ CHANGED FUNCTION" if fn in changed_fns else ""
                line_preview = ", ".join(str(l) for l in lines[:3])
                print(f"       {fn:<45} {len(lines):3d} lines{marker}")

    # ── Overall summary ──
    overall_recall = total_tselect_covered / total_full_covered * 100 if total_full_covered else 0

    print(f"\n{'═'*62}")
    print(f"  SUMMARY")
    print(f"{'═'*62}")
    print()
    print(f"  {'Tests in full suite':<35} : {full_test_count} test files")
    print(f"  {'Tests selected by tselect':<35} : {tselect_test_count} test methods")
    print(f"  {'Tests eliminated':<35} : {full_test_count - tselect_test_count}  ({reduction:.1f}%)")
    print()
    print(f"  ── What matters: PR-level recall ────────────────────")
    print(f"  {'Full suite covered (whole file)':<35} : {total_full_covered} lines")
    print(f"  {'tselect covered (whole file)':<35} : {total_tselect_covered} lines")
    print(f"  {'File-level recall':<35} : {overall_recall:.1f}%")
    print()
    print(f"  Recall  {bar(overall_recall)}  {color_pct(overall_recall)}")
    print()

    # ── One-liner verdict ──
    print(f"{'─'*62}")
    print(f"  tselect selected {tselect_test_count} of {full_test_count} test files")
    print(f"  using only {tselect_test_count} targeted tests vs the full suite.")
    print(f"  See PR Recall above for coverage of the actual changed lines.")
    print(f"{'═'*62}\n")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def auto_detect_tselect_count(changed_files: list, repo_root: Path) -> int:
    """Call tselect directly and count selected test methods."""
    try:
        import sys
        sys.path.insert(0, str(Path("/Users/nihalkumar/Desktop/nbaworks/tselect")))
        from tselect.core.graph_loader import GraphLoader
        from tselect.core.graph_selector import select_tests_from_graph, get_pytest_node_ids

        loader = GraphLoader(repo_root)
        if not loader.exists():
            print("⚠️  No graph found — cannot auto-detect tselect test count")
            return 0

        graph = loader.load()
        selected, _ = select_tests_from_graph(changed_files, graph, repo_root)
        node_ids = get_pytest_node_ids(selected)
        return len(node_ids)
    except Exception as e:
        print(f"⚠️  Could not auto-detect tselect count: {e}")
        return 0


def auto_detect_full_suite_count(repo_root: Path) -> int:
    """
    Read Phase 1 progress file to count how many test files ran,
    then estimate total test methods from coverage_full.xml test count.
    Falls back to counting test files discovered.
    """
    import json

    progress_file = repo_root / ".full_coverage_progress.json"
    if progress_file.exists():
        progress = json.loads(progress_file.read_text())
        done    = len(progress.get("done", []))
        skipped = len(progress.get("skipped", []))
        total   = done + skipped
        print(f"  (Phase 1 ran {done} files, skipped {skipped}, total {total} test files)")
        return total

    # fallback: count test files
    test_dir = repo_root / "test" / "inductor"
    return len(list(test_dir.glob("test_*.py")))


def main():
    parser = argparse.ArgumentParser(description="Compare tselect vs full suite coverage")
    parser.add_argument("--changed",       nargs="+", required=True,
                        help="Changed files (e.g. torch/_inductor/lowering.py)")
    parser.add_argument("--repo",          type=str, default=str(PYTORCH_ROOT))
    args = parser.parse_args()

    repo_root = Path(args.repo)

    print("\n  Auto-detecting test counts...")
    tselect_count = auto_detect_tselect_count(args.changed, repo_root)
    full_count    = auto_detect_full_suite_count(repo_root)

    print(f"  tselect selected  : {tselect_count} tests")
    print(f"  Full suite ran    : {full_count} test files")

    run_report(args.changed, tselect_count, full_count, repo_root)


if __name__ == "__main__":
    main()