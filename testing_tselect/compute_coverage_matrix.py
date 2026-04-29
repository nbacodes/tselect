#!/usr/bin/env python3
"""
compute_coverage_matrix.py
---------------------------
Computes confusion matrix by comparing:
  - coverage_full.xml     → ground truth (all inductor tests ran)
  - coverage_selected.xml → tselect's selected tests ran

At FUNCTION level — same method as PR #176888 evaluation.

Usage:
    python3 compute_coverage_matrix.py
    python3 compute_coverage_matrix.py --pr 152993
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────
COVERAGE_FULL = '/Users/nihalkumar/pytorch-pr-176888/coverage_full.xml'
TSELECT_TEST_DIR = Path('/Users/nihalkumar/Desktop/nbaworks/tselect/tselect_test')

PRS = [
    {"pr": "121953", "desc": "Fix addmm fusion check",        "category": "bug-fix",
     "changed_files": ["torch/_inductor/fx_passes/mkldnn_fusion.py"]},
    {"pr": "181666", "desc": "SDK lib fallback cpp-wrapper",  "category": "feature",
     "changed_files": ["torch/_inductor/codecache.py", "torch/_inductor/cpp_builder.py"]},
    {"pr": "145690", "desc": "Add typing to simd.py",         "category": "types-only",
     "changed_files": ["torch/_inductor/codegen/simd.py"]},
    {"pr": "152993", "desc": "Fix ModularIndexing",           "category": "bug-fix",
     "changed_files": ["torch/_inductor/scheduler.py", "torch/_inductor/ir.py"]},
]


def parse_coverage_xml(xml_path: str) -> dict:
    """
    Parse coverage XML → dict of:
      filename → set of (function_name, line_no) tuples that were executed
    Only keeps files with at least one hit line.
    """
    result = {}
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"  ERROR parsing {xml_path}: {e}")
        return result

    for cls in root.iter("class"):
        filename = cls.get("filename", "")
        if not filename:
            continue

        # Only keep torch/_inductor files
        if "torch/_inductor" not in filename and "torch\\_inductor" not in filename:
            continue

        # Normalize
        idx = filename.find("torch/_inductor")
        if idx == -1:
            continue
        norm_file = filename[idx:]

        executed_lines = set()
        for line in cls.iter("line"):
            if int(line.get("hits", 0)) > 0:
                executed_lines.add(int(line.get("number", 0)))

        if executed_lines:
            if norm_file not in result:
                result[norm_file] = set()
            result[norm_file].update(executed_lines)

    return result


def get_executed_functions(coverage_data: dict, xml_path: str) -> set:
    """
    Extract function-level coverage from XML.
    Returns set of "filename::classname::methodname" strings.
    """
    functions = set()
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"  ERROR parsing {xml_path}: {e}")
        return functions

    for cls in root.iter("class"):
        filename = cls.get("filename", "")
        if "torch/_inductor" not in filename:
            continue

        idx = filename.find("torch/_inductor")
        norm_file = filename[idx:]
        class_name = cls.get("name", "")

        for method in cls.iter("method"):
            method_name = method.get("name", "")
            # Check if method has any executed lines
            executed = any(
                int(line.get("hits", 0)) > 0
                for line in method.iter("line")
            )
            if executed:
                functions.add(f"{norm_file}::{class_name}::{method_name}")

    return functions


def compute_matrix_for_pr(pr_info: dict, full_functions: set) -> dict:
    pr = pr_info["pr"]
    sel_xml = TSELECT_TEST_DIR / pr / "coverage_selected.xml"

    if not sel_xml.exists():
        return {
            "pr": pr,
            "desc": pr_info["desc"],
            "category": pr_info["category"],
            "changed_files": pr_info["changed_files"],
            "error": f"coverage_selected.xml not found — run run_coverage_for_pr.sh {pr} first",
        }

    print(f"  Parsing selected coverage for PR #{pr}...")
    sel_functions = get_executed_functions({}, str(sel_xml))

    # Only evaluate functions in changed files
    changed_files = pr_info["changed_files"]

    # Filter to functions in changed files only
    def in_changed(fn_key):
        return any(cf in fn_key for cf in changed_files)

    gt_relevant  = {f for f in full_functions if in_changed(f)}
    sel_relevant = {f for f in sel_functions  if in_changed(f)}

    # Also compute overall (all inductor functions)
    tp_all = len(full_functions & sel_functions)
    fp_all = len(sel_functions - full_functions)
    fn_all = len(full_functions - sel_functions)

    # Changed-file level
    tp = len(gt_relevant & sel_relevant)
    fp = len(sel_relevant - gt_relevant)
    fn = len(gt_relevant - sel_relevant)

    precision_cf = tp / (tp + fp) if (tp + fp) > 0 else None
    recall_cf    = tp / (tp + fn) if (tp + fn) > 0 else None
    f1_cf = (2 * precision_cf * recall_cf / (precision_cf + recall_cf)
             if precision_cf and recall_cf else None)

    # Overall recall
    recall_all = tp_all / (tp_all + fn_all) if (tp_all + fn_all) > 0 else None
    precision_all = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else None

    return {
        "pr":           pr,
        "desc":         pr_info["desc"],
        "category":     pr_info["category"],
        "changed_files": changed_files,
        "error":        None,
        # Changed-file level
        "cf_tp": tp, "cf_fp": fp, "cf_fn": fn,
        "cf_precision": precision_cf,
        "cf_recall":    recall_cf,
        "cf_f1":        f1_cf,
        # Overall inductor level
        "all_tp": tp_all, "all_fp": fp_all, "all_fn": fn_all,
        "all_recall":    recall_all,
        "all_precision": precision_all,
        # FN details
        "fn_functions": sorted(gt_relevant - sel_relevant)[:20],
    }


def print_report(results: list):
    SEP = "═" * 75
    print(f"\n{SEP}")
    print("  tselect Multi-PR Confusion Matrix (coverage-based, function level)")
    print(SEP)

    for r in results:
        print(f"\n{'─'*75}")
        print(f"  PR #{r['pr']}  [{r['category']}]  {r['desc']}")
        print(f"  Changed : {', '.join(r['changed_files'])}")

        if r.get("error"):
            print(f"  ⏳ {r['error']}")
            continue

        # Changed-file level
        p   = f"{r['cf_precision']:.3f}" if r['cf_precision'] is not None else "N/A"
        rec = f"{r['cf_recall']:.3f}"    if r['cf_recall']    is not None else "N/A"
        f1  = f"{r['cf_f1']:.3f}"        if r['cf_f1']        is not None else "N/A"
        print(f"\n  [Changed files scope]")
        print(f"  TP={r['cf_tp']}  FP={r['cf_fp']}  FN={r['cf_fn']}")
        print(f"  Precision={p}  Recall={rec}  F1={f1}")

        # Overall level
        p2   = f"{r['all_precision']:.3f}" if r['all_precision'] is not None else "N/A"
        rec2 = f"{r['all_recall']:.3f}"    if r['all_recall']    is not None else "N/A"
        print(f"\n  [All inductor scope]")
        print(f"  TP={r['all_tp']}  FP={r['all_fp']}  FN={r['all_fn']}")
        print(f"  Precision={p2}  Recall={rec2}")

        if r['fn_functions']:
            print(f"\n  FN functions (in changed files, missed by tselect):")
            for fn in r['fn_functions'][:5]:
                print(f"    ✗ {fn}")
            if len(r['fn_functions']) > 5:
                print(f"    ... and {len(r['fn_functions'])-5} more")

    # Aggregate
    valid = [r for r in results if not r.get("error") and r.get("cf_recall") is not None]
    if valid:
        print(f"\n{SEP}")
        print("  Aggregate (changed-file scope, function level)")
        print(f"{'─'*75}")
        total_tp = sum(r['cf_tp'] for r in valid)
        total_fp = sum(r['cf_fp'] for r in valid)
        total_fn = sum(r['cf_fn'] for r in valid)
        agg_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else None
        agg_pre = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else None
        print(f"  PRs evaluated : {len(valid)}")
        print(f"  Total TP      : {total_tp}")
        print(f"  Total FP      : {total_fp}")
        print(f"  Total FN      : {total_fn}")
        print(f"  Agg Recall    : {agg_rec:.3f}" if agg_rec else "  Agg Recall    : N/A")
        print(f"  Agg Precision : {agg_pre:.3f}" if agg_pre else "  Agg Precision : N/A")

        print(f"\n  Recall by category:")
        cat_tp = defaultdict(int); cat_fn = defaultdict(int)
        for r in valid:
            cat_tp[r['category']] += r['cf_tp']
            cat_fn[r['category']] += r['cf_fn']
        for cat in sorted(cat_tp):
            tp, fn = cat_tp[cat], cat_fn[cat]
            rec = tp / (tp + fn) if (tp + fn) > 0 else None
            print(f"    {cat:20s}: {'N/A' if rec is None else f'{rec:.3f}'}")

    print(f"\n{SEP}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", help="Single PR to evaluate")
    args = parser.parse_args()

    print(f"Parsing ground truth: {COVERAGE_FULL}")
    full_functions = get_executed_functions({}, COVERAGE_FULL)
    print(f"  {len(full_functions)} functions executed in full inductor test suite")

    prs = [p for p in PRS if p["pr"] == args.pr] if args.pr else PRS

    results = []
    for pr_info in prs:
        r = compute_matrix_for_pr(pr_info, full_functions)
        results.append(r)

    print_report(results)

    import json
    out = TSELECT_TEST_DIR / "coverage_matrix_report.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
