"""
run_tselect_for_prs.py
-----------------------
Runs tselect for each PR using the actual Python API:
    select_tests_from_graph(changed_files, graph, repo_root, config)

Usage:
    python3 run_tselect_for_prs.py
    python3 run_tselect_for_prs.py --pr 152993
    python3 run_tselect_for_prs.py --all
"""

import json
import sys
import os
import argparse
import urllib.request
from pathlib import Path

sys.path.insert(0, '/Users/nihalkumar/Desktop/nbaworks/tselect')

from tselect.core.graph_selector import select_tests_from_graph

# ── Config ────────────────────────────────────────────────────
GRAPH_FILE  = '/Users/nihalkumar/pytorch/.graph/tselect/dependency_graph.json'
REPO_ROOT   = Path('/Users/nihalkumar/pytorch')
RESULTS_DIR = Path('/Users/nihalkumar/Desktop/nbaworks/tselect/tselect_test')

CONFIG = {
    "graph": {"fanout_threshold": 30, "transitive_depth": 1}
}

# ── PR slate with known changed files ─────────────────────────
# Changed files fetched from GitHub API or filled manually
PRS = [
    {
        "pr": "145690",
        "desc": "Add typing to simd.py",
        "category": "types-only",
        "changed_files": [
            "torch/_inductor/codegen/simd.py",
        ],
    },
    {
        "pr": "145691",
        "desc": "Add typing to common.py",
        "category": "types-only",
        "changed_files": [
            "torch/_inductor/codegen/common.py",
        ],
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
        "changed_files": [
            "torch/_inductor/fx_passes/mkldnn_fusion.py",
        ],
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


def fetch_changed_files_from_github(pr_number: str) -> list[str]:
    """Try to fetch actual changed files from GitHub API."""
    try:
        url = f"https://api.github.com/repos/pytorch/pytorch/pulls/{pr_number}/files"
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github.v3+json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            files = json.loads(resp.read())
            return [f["filename"] for f in files]
    except Exception as e:
        print(f"  GitHub fetch failed ({e}), using hardcoded files")
        return []


def run_pr(pr_info: dict, graph: dict, use_github: bool = True) -> dict:
    pr = pr_info["pr"]
    print(f"\n{'─'*60}")
    print(f"PR #{pr} — {pr_info['desc']} [{pr_info['category']}]")

    # Try GitHub first, fall back to hardcoded
    changed_files = []
    if use_github:
        changed_files = fetch_changed_files_from_github(pr)

    if not changed_files:
        changed_files = pr_info["changed_files"]
        print(f"  Using hardcoded changed files")
    else:
        print(f"  Fetched {len(changed_files)} changed files from GitHub")

    print(f"  Changed files:")
    for f in changed_files:
        print(f"    {f}")

    # Run tselect
    try:
        selected, total_methods = select_tests_from_graph(
            changed_files=changed_files,
            graph=graph,
            repo_root=REPO_ROOT,
            config=CONFIG,
        )
        error = None
    except Exception as e:
        import traceback
        selected = {}
        total_methods = 0
        error = traceback.format_exc()
        print(f"  ERROR: {e}")

    print(f"  Selected tests: {len(selected)}")

    # Save results
    out_dir = RESULTS_DIR / pr
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write selected_tests.txt
    selected_file = out_dir / "selected_tests.txt"
    with open(selected_file, "w") as f:
        for test_path in sorted(selected.keys()):
            f.write(test_path + "\n")

    # Write changed_files.txt
    with open(out_dir / "changed_files.txt", "w") as f:
        for cf in changed_files:
            f.write(cf + "\n")

    # Write metadata
    with open(out_dir / "metadata.json", "w") as f:
        json.dump({
            "pr": pr,
            "desc": pr_info["desc"],
            "category": pr_info["category"],
            "changed_files": changed_files,
            "selected_count": len(selected),
            "total_methods_traversed": total_methods,
            "error": error,
            "config": CONFIG,
        }, f, indent=2)

    # Print selected tests
    if selected:
        for test_path in sorted(selected.keys()):
            print(f"    → {test_path}")
    else:
        print(f"    (no tests selected)")

    return {
        "pr": pr,
        "selected": selected,
        "changed_files": changed_files,
        "error": error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", help="Single PR number")
    parser.add_argument("--all", action="store_true", default=True)
    parser.add_argument("--no-github", action="store_true",
                        help="Skip GitHub API, use hardcoded files only")
    args = parser.parse_args()

    # Load graph
    print(f"Loading dependency graph from {GRAPH_FILE}...")
    if not Path(GRAPH_FILE).exists():
        print(f"ERROR: Graph file not found: {GRAPH_FILE}")
        print("Run tselect graph build first.")
        sys.exit(1)

    with open(GRAPH_FILE) as f:
        graph = json.load(f)
    print(f"  Graph loaded. Keys: {list(graph.keys())[:5]}...")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    prs_to_run = PRS
    if args.pr:
        prs_to_run = [p for p in PRS if p["pr"] == args.pr]
        if not prs_to_run:
            print(f"PR #{args.pr} not in slate")
            sys.exit(1)

    results = []
    for pr_info in prs_to_run:
        result = run_pr(pr_info, graph, use_github=not args.no_github)
        results.append(result)

    # Summary
    print(f"\n{'═'*60}")
    print("SUMMARY")
    print(f"{'─'*60}")
    print(f"{'PR':<10} {'Category':<16} {'Selected':>10} {'Changed files':>15}")
    print(f"{'─'*60}")
    for r in results:
        pr_info = next(p for p in PRS if p["pr"] == r["pr"])
        print(f"#{r['pr']:<9} {pr_info['category']:<16} "
              f"{len(r['selected']):>10} {len(r['changed_files']):>15}")

    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
