"""
compute_matrix_from_graph.py
-----------------------------
Computes confusion matrix for each PR WITHOUT running tests.

Ground truth = all test files that the dependency graph says
               cover the changed files (the "oracle" set).
Selected     = what tselect actually picked (selected_tests.txt).

Also validates against coverage_full.xml to confirm changed files
are actually reachable (non-zero coverage).

Usage:
    python3 compute_matrix_from_graph.py
    python3 compute_matrix_from_graph.py --pr 152993
"""

import json
import sys
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, '/Users/nihalkumar/Desktop/nbaworks/tselect')

# ── Config ────────────────────────────────────────────────────
GRAPH_FILE    = '/Users/nihalkumar/pytorch/.graph/tselect/dependency_graph.json'
COVERAGE_XML  = '/Users/nihalkumar/pytorch-pr-176888/coverage_full.xml'
RESULTS_DIR   = Path('/Users/nihalkumar/Desktop/nbaworks/tselect/tselect_test')
REPO_ROOT     = Path('/Users/nihalkumar/pytorch')

PRS = [
    {
        "pr": "145690",
        "desc": "Add typing to simd.py",
        "category": "types-only",
        "changed_files": ["torch/_inductor/codegen/simd.py"],
    },
    {
        "pr": "145691",
        "desc": "Add typing to common.py",
        "category": "types-only",
        "changed_files": ["torch/_inductor/codegen/common.py"],
    },
    {
        "pr": "145692",
        "desc": "get_backend_features to OrderedSet",
        "category": "types-runtime",
        "changed_files": [
            "torch/_inductor/codegen/common.py",
            "torch/_inductor/scheduler.py",
        ],
    },
    {
        "pr": "152993",
        "desc": "Fix ModularIndexing assumptions",
        "category": "bug-fix",
        "changed_files": [
            "torch/_inductor/scheduler.py",
            "torch/_inductor/ir.py",
        ],
    },
    {
        "pr": "121953",
        "desc": "Fix addmm fusion check",
        "category": "bug-fix",
        "changed_files": ["torch/_inductor/fx_passes/mkldnn_fusion.py"],
    },
    {
        "pr": "181666",
        "desc": "SDK lib fallback cpp-wrapper",
        "category": "feature",
        "changed_files": [
            "torch/_inductor/codecache.py",
            "torch/_inductor/cpp_builder.py",
        ],
    },
    {
        "pr": "181700",
        "desc": "TMA load in fused pointwise epilogues",
        "category": "gpu-codegen",
        "changed_files": [
            "torch/_inductor/codegen/triton.py",
            "torch/_inductor/ir.py",
        ],
    },
]


def load_graph():
    print(f"Loading graph: {GRAPH_FILE}")
    with open(GRAPH_FILE) as f:
        return json.load(f)


def get_oracle_tests(changed_files: list, graph: dict) -> set:
    """
    Get ALL tests that the graph says cover any of the changed files.
    Mirrors tselect's actual traversal logic:
      1. full_reverse_graph  → direct file→test edges
      2. source_reverse_graph → BFS transitive expansion through source files
      3. function_reverse_graph → function-level test references (extract file)
    """
    full_rev     = graph.get("full_reverse_graph", {})
    source_rev   = graph.get("source_reverse_graph", {})
    function_rev = graph.get("function_reverse_graph", {})

    oracle = set()
    FANOUT_THRESHOLD = 30  # match tselect default

    for cf in changed_files:
        # ── 1. Direct full_reverse_graph lookup ──────────────────
        if cf in full_rev:
            oracle.update(full_rev[cf])

        # ── 2. BFS through source_reverse_graph ──────────────────
        # source_rev maps source→source importers, then full_rev maps those to tests
        visited = set()
        queue = [cf]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            importers = source_rev.get(node, [])
            # apply fanout threshold
            if len(importers) > FANOUT_THRESHOLD:
                continue
            for importer in importers:
                # if importer is a test file, add directly
                if importer.startswith("test/") or "/test" in importer:
                    oracle.add(importer)
                else:
                    # it's a source file — get its test dependents
                    tests = full_rev.get(importer, [])
                    if len(tests) <= FANOUT_THRESHOLD:
                        oracle.update(tests)
                    queue.append(importer)

        # ── 3. function_reverse_graph — extract test file paths ──
        for fn_key, test_refs in function_rev.items():
            # fn_key format: "torch/_inductor/scheduler.py::ClassName::method"
            if cf in fn_key:
                for ref in test_refs:
                    # ref format: "test/inductor/test_foo.py::Class::method"
                    test_file = ref.split("::")[0]
                    oracle.add(normalize_test_path(test_file))

    return oracle


def normalize_test_path(p: str) -> str:
    """Normalize to test/... format."""
    p = p.strip()
    if not p.startswith("test/"):
        p = "test/" + p.lstrip("/")
    if not p.endswith(".py"):
        p += ".py"
    return p


def load_selected_tests(pr: str) -> set:
    sel_file = RESULTS_DIR / pr / "selected_tests.txt"
    if not sel_file.exists():
        return set()
    with open(sel_file) as f:
        return {normalize_test_path(l.strip()) for l in f if l.strip()}


def get_covered_files_from_xml(xml_path: str) -> set:
    """
    Parse coverage XML and return set of source files
    that have at least one covered line (hits > 0).
    """
    covered = set()
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for cls in root.iter("class"):
            filename = cls.get("filename", "")
            # Check if any line has hits > 0
            for line in cls.iter("line"):
                if int(line.get("hits", 0)) > 0:
                    covered.add(filename)
                    break
    except Exception as e:
        print(f"  WARNING: could not parse XML: {e}")
    return covered


def compute_matrix(pr_info: dict, graph: dict, covered_xml: set) -> dict:
    pr = pr_info["pr"]
    changed_files = pr_info["changed_files"]

    # Ground truth from graph
    oracle = get_oracle_tests(changed_files, graph)

    # tselect selection
    selected = load_selected_tests(pr)

    # Normalize oracle paths
    oracle_norm = {normalize_test_path(t) for t in oracle}

    # Check changed files are in XML coverage (reachability check)
    reachable = []
    unreachable = []
    for cf in changed_files:
        fname = cf.split("/")[-1].replace(".py", "")
        hits = any(fname in f for f in covered_xml)
        (reachable if hits else unreachable).append(cf)

    # Compute matrix
    tp = len(selected & oracle_norm)
    fp = len(selected - oracle_norm)
    fn = len(oracle_norm - selected)
    tn = 0  # not meaningful at test-file level

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall    = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and (precision + recall) > 0 else None)

    return {
        "pr":           pr,
        "desc":         pr_info["desc"],
        "category":     pr_info["category"],
        "changed_files": changed_files,
        "oracle_count": len(oracle_norm),
        "selected_count": len(selected),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "reachable_files":   reachable,
        "unreachable_files": unreachable,
        "fn_tests":  sorted(oracle_norm - selected),   # what tselect missed
        "fp_tests":  sorted(selected - oracle_norm),   # what tselect over-selected
    }


def print_report(results: list):
    SEP = "═" * 75

    print(f"\n{SEP}")
    print("  tselect Multi-PR Confusion Matrix (graph-oracle method)")
    print(SEP)

    for r in results:
        print(f"\n{'─'*75}")
        print(f"  PR #{r['pr']}  [{r['category']}]  {r['desc']}")
        print(f"  Changed : {', '.join(r['changed_files'])}")
        print(f"  Oracle  : {r['oracle_count']} tests (graph ground truth)")
        print(f"  Selected: {r['selected_count']} tests (tselect)")
        print(f"  TP={r['tp']}  FP={r['fp']}  FN={r['fn']}")

        p   = f"{r['precision']:.3f}" if r['precision'] is not None else "N/A"
        rec = f"{r['recall']:.3f}"    if r['recall']    is not None else "N/A"
        f1  = f"{r['f1']:.3f}"        if r['f1']        is not None else "N/A"
        print(f"  Precision={p}  Recall={rec}  F1={f1}")

        if r['unreachable_files']:
            print(f"  ⚠️  Not in coverage XML (GPU/hardware gate): {r['unreachable_files']}")

        if r['fn_tests']:
            print(f"  FN tests (missed by tselect):")
            for t in r['fn_tests'][:5]:
                print(f"    ✗ {t}")
            if len(r['fn_tests']) > 5:
                print(f"    ... and {len(r['fn_tests'])-5} more")

    # Aggregate
    valid = [r for r in results if r['oracle_count'] > 0]
    print(f"\n{SEP}")
    print("  Aggregate (function-proxy = test-file level)")
    print(f"{'─'*75}")

    total_tp = sum(r['tp'] for r in valid)
    total_fp = sum(r['fp'] for r in valid)
    total_fn = sum(r['fn'] for r in valid)
    agg_recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else None
    agg_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else None

    print(f"  PRs evaluated : {len(valid)}")
    print(f"  Total TP      : {total_tp}")
    print(f"  Total FP      : {total_fp}")
    print(f"  Total FN      : {total_fn}")
    print(f"  Agg Recall    : {agg_recall:.3f}" if agg_recall else "  Agg Recall    : N/A")
    print(f"  Agg Precision : {agg_precision:.3f}" if agg_precision else "  Agg Precision : N/A")

    # By category
    from collections import defaultdict
    cat_tp = defaultdict(int); cat_fn = defaultdict(int)
    for r in valid:
        cat_tp[r['category']] += r['tp']
        cat_fn[r['category']] += r['fn']

    print(f"\n  Recall by category:")
    for cat in sorted(cat_tp):
        tp, fn = cat_tp[cat], cat_fn[cat]
        rec = tp / (tp + fn) if (tp + fn) > 0 else None
        print(f"    {cat:20s}: {'N/A' if rec is None else f'{rec:.3f}'}")

    print(f"\n{SEP}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", help="Single PR to evaluate")
    args = parser.parse_args()

    graph = load_graph()

    print(f"Parsing coverage XML: {COVERAGE_XML}")
    covered_xml = get_covered_files_from_xml(COVERAGE_XML)
    print(f"  {len(covered_xml)} source files with coverage > 0")

    prs = [p for p in PRS if p["pr"] == args.pr] if args.pr else PRS

    results = []
    for pr_info in prs:
        r = compute_matrix(pr_info, graph, covered_xml)
        results.append(r)

    print_report(results)

    # Save JSON
    out = RESULTS_DIR / "matrix_report.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
