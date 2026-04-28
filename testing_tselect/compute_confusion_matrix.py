#!/usr/bin/env python3
"""
compute_confusion_matrix.py
============================
Computes per-PR and aggregate confusion matrices at three levels:
  - File level
  - Function level (most meaningful for tselect)
  - Line level

Input layout (from the 3-step pipeline):
  multi_pr_results/
    ground_truth/<module>/coverage_combined.json  ← Phase A: all tests
    <PR>/selected_tests.txt                       ← Phase B: tselect output
    <PR>/selected_coverage/coverage.json          ← Phase C: selected run

Usage:
    python3 compute_confusion_matrix.py --results-dir ./multi_pr_results
    python3 compute_confusion_matrix.py --pr 152993
    python3 compute_confusion_matrix.py --results-dir ./multi_pr_results --format csv
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Set, Tuple


# ── PR metadata ───────────────────────────────────────────────
PR_INFO = {
    "145690": {"desc": "typing to simd.py",             "category": "types-only",  "module": "inductor_codegen"},
    "145691": {"desc": "typing to common.py",            "category": "types-only",  "module": "inductor_codegen"},
    "145692": {"desc": "get_backend_features→OrderedSet","category": "types-runtime","module": "inductor_codegen"},
    "152993": {"desc": "Fix ModularIndexing",            "category": "bug-fix",     "module": "inductor"},
    "121953": {"desc": "Fix addmm fusion check",         "category": "bug-fix",     "module": "inductor_fx_passes"},
    "181666": {"desc": "SDK lib fallback cpp-wrapper",   "category": "feature",     "module": "inductor"},
    "181700": {"desc": "TMA load fused pointwise",       "category": "gpu-codegen", "module": "inductor_codegen"},
}


@dataclass
class ConfusionMatrix:
    """Stores TP/FP/FN/TN at a given granularity level."""
    level: str  # "file", "function", "line"
    tp: int = 0   # tselect selected AND covered in ground truth
    fp: int = 0   # tselect selected BUT not in ground truth
    fn: int = 0   # ground truth covered BUT tselect missed
    tn: int = 0   # not selected AND not in ground truth

    @property
    def precision(self) -> Optional[float]:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else None

    @property
    def recall(self) -> Optional[float]:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else None

    @property
    def f1(self) -> Optional[float]:
        p, r = self.precision, self.recall
        if p is None or r is None or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)

    def summary(self) -> str:
        p = f"{self.precision:.2f}" if self.precision is not None else "N/A"
        r = f"{self.recall:.2f}"    if self.recall    is not None else "N/A"
        f = f"{self.f1:.2f}"        if self.f1        is not None else "N/A"
        return (f"TP={self.tp:4d}  FP={self.fp:4d}  FN={self.fn:4d}  TN={self.tn:4d} | "
                f"Precision={p}  Recall={r}  F1={f}")


@dataclass
class PRResult:
    pr: str
    desc: str
    category: str
    file_matrix:     Optional[ConfusionMatrix] = None
    function_matrix: Optional[ConfusionMatrix] = None
    line_matrix:     Optional[ConfusionMatrix] = None
    error: Optional[str] = None
    selected_count: int = 0
    changed_files: list = field(default_factory=list)


def load_coverage_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  WARNING: could not load {path}: {e}")
        return None


def extract_covered_items(cov_data: dict) -> Tuple[Set[str], Set[str], Set[Tuple[str,int]]]:
    """Extract covered files, functions, and lines from a coverage JSON."""
    covered_files: Set[str] = set()
    covered_fns: Set[str] = set()   # "file::function"
    covered_lines: Set[Tuple[str, int]] = set()

    if not cov_data:
        return covered_files, covered_fns, covered_lines

    for filepath, fdata in cov_data.get("files", {}).items():
        # Normalize path to module-relative
        norm = normalize_path(filepath)
        if not norm:
            continue

        covered_files.add(norm)

        # Functions
        for fn_name, fn_info in fdata.get("functions", {}).items():
            if fn_info.get("executed_lines"):
                covered_fns.add(f"{norm}::{fn_name}")

        # Lines
        for line_no in fdata.get("executed_lines", []):
            covered_lines.add((norm, int(line_no)))

    return covered_files, covered_fns, covered_lines


def normalize_path(filepath: str) -> Optional[str]:
    """Normalize a filesystem path to torch._inductor module path."""
    # Keep paths containing torch/_inductor
    if "torch/_inductor" not in filepath and "torch\\_inductor" not in filepath:
        return None
    # Trim to torch/_inductor/...
    idx = filepath.find("torch/_inductor")
    if idx == -1:
        idx = filepath.find("torch\\_inductor")
    if idx == -1:
        return None
    return filepath[idx:].replace("\\", "/")


def load_selected_tests(selected_file: Path) -> Set[str]:
    if not selected_file.exists():
        return set()
    tests = set()
    with open(selected_file) as f:
        for line in f:
            t = line.strip()
            if t:
                tests.add(t.lstrip("test/"))
    return tests


def load_changed_files(changed_file: Path) -> list:
    if not changed_file.exists():
        return []
    with open(changed_file) as f:
        return [l.strip() for l in f if l.strip()]


def compute_matrix_for_pr(pr: str, results_dir: Path, ground_truth_dir: Path) -> PRResult:
    info = PR_INFO.get(pr, {"desc": "Unknown", "category": "unknown", "module": "inductor"})
    result = PRResult(pr=pr, desc=info["desc"], category=info["category"])

    pr_dir = results_dir / pr
    if not pr_dir.exists():
        result.error = f"PR directory not found: {pr_dir}"
        return result

    # Load selected tests
    selected_tests = load_selected_tests(pr_dir / "selected_tests.txt")
    result.selected_count = len(selected_tests)
    result.changed_files = load_changed_files(pr_dir / "changed_files.txt")

    # Load ground truth coverage (Phase A)
    module = info["module"]
    gt_path = ground_truth_dir / module / "coverage_combined.json"
    gt_data = load_coverage_json(gt_path)
    if not gt_data:
        result.error = f"Ground truth not found: {gt_path}"
        return result

    # Load selected test coverage (Phase C)
    sel_cov_path = pr_dir / "selected_coverage" / "coverage.json"
    sel_data = load_coverage_json(sel_cov_path)
    if not sel_data:
        result.error = f"Selected coverage not found: {sel_cov_path}"
        return result

    # Extract covered items from both
    gt_files, gt_fns, gt_lines = extract_covered_items(gt_data)
    sel_files, sel_fns, sel_lines = extract_covered_items(sel_data)

    # ── File-level matrix ────────────────────────────────────
    fm = ConfusionMatrix(level="file")
    for f in gt_files | sel_files:
        in_gt = f in gt_files
        in_sel = f in sel_files
        if in_gt and in_sel:   fm.tp += 1
        elif in_sel and not in_gt: fm.fp += 1
        elif in_gt and not in_sel: fm.fn += 1
        else:                  fm.tn += 1
    result.file_matrix = fm

    # ── Function-level matrix ─────────────────────────────────
    fnm = ConfusionMatrix(level="function")
    for fn in gt_fns | sel_fns:
        in_gt = fn in gt_fns
        in_sel = fn in sel_fns
        if in_gt and in_sel:       fnm.tp += 1
        elif in_sel and not in_gt: fnm.fp += 1
        elif in_gt and not in_sel: fnm.fn += 1
        else:                      fnm.tn += 1
    result.function_matrix = fnm

    # ── Line-level matrix ─────────────────────────────────────
    lm = ConfusionMatrix(level="line")
    for item in gt_lines | sel_lines:
        in_gt = item in gt_lines
        in_sel = item in sel_lines
        if in_gt and in_sel:       lm.tp += 1
        elif in_sel and not in_gt: lm.fp += 1
        elif in_gt and not in_sel: lm.fn += 1
        else:                      lm.tn += 1
    result.line_matrix = lm

    return result


def print_report(results: list[PRResult], fmt: str = "text"):
    if fmt == "csv":
        print_csv(results)
        return

    SEP = "═" * 82
    print(f"\n{SEP}")
    print("  tselect Multi-PR Confusion Matrix Report")
    print(SEP)

    for r in results:
        print(f"\n{'─'*82}")
        print(f"  PR #{r.pr}  [{r.category}]  {r.desc}")
        print(f"  Changed files : {', '.join(r.changed_files) or 'unknown'}")
        print(f"  Tests selected: {r.selected_count}")

        if r.error:
            print(f"  ERROR: {r.error}")
            continue

        for matrix in [r.file_matrix, r.function_matrix, r.line_matrix]:
            if matrix:
                print(f"  [{matrix.level:8s}]  {matrix.summary()}")

    # ── Aggregate summary ─────────────────────────────────────
    valid = [r for r in results if r.error is None and r.function_matrix]
    if not valid:
        return

    print(f"\n{SEP}")
    print("  Aggregate Summary (function level)")
    print(f"{'─'*82}")

    total_tp = sum(r.function_matrix.tp for r in valid)
    total_fp = sum(r.function_matrix.fp for r in valid)
    total_fn = sum(r.function_matrix.fn for r in valid)
    agg_recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else None
    agg_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else None

    print(f"  PRs evaluated : {len(valid)}")
    print(f"  Total TP      : {total_tp}")
    print(f"  Total FP      : {total_fp}")
    print(f"  Total FN      : {total_fn}")
    print(f"  Agg Recall    : {agg_recall:.3f}" if agg_recall is not None else "  Agg Recall    : N/A")
    print(f"  Agg Precision : {agg_precision:.3f}" if agg_precision is not None else "  Agg Precision : N/A")

    # FN breakdown by category
    print(f"\n  FN breakdown by category:")
    from collections import defaultdict
    cat_fns: Dict[str, int] = defaultdict(int)
    for r in valid:
        cat_fns[r.category] += r.function_matrix.fn
    for cat, fns in sorted(cat_fns.items(), key=lambda x: -x[1]):
        print(f"    {cat:20s}: FN={fns}")

    print(f"\n  Recall by category:")
    cat_tp: Dict[str, int] = defaultdict(int)
    cat_fn: Dict[str, int] = defaultdict(int)
    for r in valid:
        cat_tp[r.category] += r.function_matrix.tp
        cat_fn[r.category] += r.function_matrix.fn
    for cat in sorted(cat_tp.keys()):
        tp, fn = cat_tp[cat], cat_fn[cat]
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        print(f"    {cat:20s}: Recall={'N/A' if recall is None else f'{recall:.3f}'}")

    print(f"\n{SEP}\n")


def print_csv(results: list[PRResult]):
    print("pr,desc,category,level,tp,fp,fn,tn,precision,recall,f1,selected_tests")
    for r in results:
        for matrix in [r.file_matrix, r.function_matrix, r.line_matrix]:
            if not matrix:
                continue
            p = f"{matrix.precision:.4f}" if matrix.precision is not None else ""
            rec = f"{matrix.recall:.4f}"  if matrix.recall    is not None else ""
            f1  = f"{matrix.f1:.4f}"      if matrix.f1        is not None else ""
            print(f"{r.pr},{r.desc},{r.category},{matrix.level},"
                  f"{matrix.tp},{matrix.fp},{matrix.fn},{matrix.tn},"
                  f"{p},{rec},{f1},{r.selected_count}")


def save_json_report(results: list[PRResult], out_path: Path):
    data = []
    for r in results:
        entry = {
            "pr": r.pr,
            "desc": r.desc,
            "category": r.category,
            "selected_count": r.selected_count,
            "changed_files": r.changed_files,
            "error": r.error,
        }
        for attr, key in [("file_matrix","file"), ("function_matrix","function"), ("line_matrix","line")]:
            m = getattr(r, attr)
            if m:
                entry[key] = {
                    "tp": m.tp, "fp": m.fp, "fn": m.fn, "tn": m.tn,
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1": m.f1,
                }
        data.append(entry)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"JSON report saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="./multi_pr_results")
    parser.add_argument("--pr", help="Single PR to evaluate")
    parser.add_argument("--format", choices=["text", "csv"], default="text")
    parser.add_argument("--save-json", help="Save JSON report to this path")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    ground_truth_dir = results_dir / "ground_truth"

    prs = [args.pr] if args.pr else list(PR_INFO.keys())

    results = []
    for pr in prs:
        print(f"Computing matrix for PR #{pr}...", file=sys.stderr)
        result = compute_matrix_for_pr(pr, results_dir, ground_truth_dir)
        results.append(result)

    print_report(results, fmt=args.format)

    if args.save_json:
        save_json_report(results, Path(args.save_json))


if __name__ == "__main__":
    main()
