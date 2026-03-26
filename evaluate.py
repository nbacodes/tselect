"""
evaluate.py
-----------
Evaluates tselect against real PyTorch PRs.

Usage:
    python3 evaluate.py

Output:
    For each PR:
      - what tselect selected
      - how many tests
      - which files triggered which tests
    
    Summary table across all PRs.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, '/Users/nihalkumar/Desktop/nbaworks/tselect')

from tselect.core.graph_selector import select_tests_from_graph

# ── config ────────────────────────────────────────────────────────────────────

GRAPH_FILE  = '/Users/nihalkumar/pytorch/.graph/tselect/dependency_graph.json'
REPO_ROOT   = Path('/Users/nihalkumar/pytorch')

# The 2 PRs we collected
PRS = [
    {
        "pr_number":     178239,
        "run_id":        23475293782,
        "artifact_id":   6074130554,
        "td_file":       "/Users/nihalkumar/pytorch/td_6074130554/td_results.json",
        "changed_files": [
            "test/higher_order_ops/test_invoke_subgraph.py",
            "torch/_dynamo/variables/invoke_subgraph.py",
            "torch/_guards.py",
            "torch/_higher_order_ops/invoke_subgraph.py",
            "torch/compiler/__init__.py",
        ],
        "description": "dynamo + higher_order_ops + compiler (5 files)",
    },
    {
        "pr_number":     178199,
        "run_id":        23474882722,
        "artifact_id":   6073989664,
        "td_file":       "/Users/nihalkumar/pytorch/td_6073989664/td_results.json",
        "changed_files": [
            "torch/distributed/tensor/experimental/_context_parallel/_load_balancer.py",
        ],
        "description": "distributed/tensor (1 file)",
    },
]

# Stopping conditions to compare
CONDITIONS = {
    "current (fanout=30, depth=1)": {
        "graph": {"fanout_threshold": 30,  "transitive_depth": 1}
    },
    "fanout=50, depth=1": {
        "graph": {"fanout_threshold": 50,  "transitive_depth": 1}
    },
    "fanout=116, depth=1": {
        "graph": {"fanout_threshold": 116, "transitive_depth": 1}
    },
    "fanout=30, depth=2": {
        "graph": {"fanout_threshold": 30,  "transitive_depth": 2}
    },
    "fanout=30, depth=unlimited": {
        "graph": {"fanout_threshold": 30,  "transitive_depth": 999}
    },
}

# ── helpers ───────────────────────────────────────────────────────────────────

def load_graph():
    print(f"Loading graph from {GRAPH_FILE}...")
    with open(GRAPH_FILE) as f:
        graph = json.load(f)
    print(f"  Keys: {list(graph.keys())}")
    return graph

def load_td_baseline(td_file):
    with open(td_file) as f:
        data = json.load(f)
    selected = {item[0]["test_file"] for item in data["_test_scores"]}
    original = set(data["_original_tests"])
    return selected, original

def run_condition(pr, graph, condition_name, config):
    try:
        selected, total_methods = select_tests_from_graph(
            changed_files = pr["changed_files"],
            graph         = graph,
            repo_root     = REPO_ROOT,
            config        = config,
        )
        return selected, total_methods, None
    except Exception as e:
        import traceback; return {}, 0, traceback.format_exc()

def compare_with_baseline(selected_files, baseline_selected):
    our_set      = set(selected_files.keys())
    
    # normalize baseline — td_results uses short paths, tselect uses full paths
    # e.g. baseline: "inductor/test_scheduler"
    #      tselect:  "test/inductor/test_scheduler.py"
    def normalize(p):
        p = p.replace("test/", "").replace(".py", "")
        return p
    
    our_normalized       = {normalize(f) for f in our_set}
    baseline_normalized  = {normalize(f) for f in baseline_selected}
    
    only_in_ours      = our_normalized - baseline_normalized
    only_in_baseline  = baseline_normalized - our_normalized
    in_both           = our_normalized & baseline_normalized
    
    return {
        "our_count":        len(our_set),
        "baseline_count":   len(baseline_selected),
        "overlap":          len(in_both),
        "only_ours":        len(only_in_ours),
        "only_baseline":    len(only_in_baseline),
        "overlap_pct":      len(in_both) / len(baseline_normalized) * 100 if baseline_normalized else 0,
    }

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    graph = load_graph()
    
    print("\n" + "="*80)
    print("TSELECT EVALUATION — 2 PRs")
    print("="*80)
    
    all_results = []
    
    for pr in PRS:
        print(f"\n{'─'*80}")
        print(f"PR #{pr['pr_number']} — {pr['description']}")
        print(f"Changed files:")
        for f in pr["changed_files"]:
            print(f"  {f}")
        
        # load baseline from td_results.json
        baseline_selected, baseline_original = load_td_baseline(pr["td_file"])
        print(f"\nBaseline (td_results.json): {len(baseline_selected)}/{len(baseline_original)} selected")
        
        print(f"\n{'CONDITION':<35} {'SELECTED':>10} {'OVERLAP%':>10} {'ONLY OURS':>10} {'ONLY BASE':>10}")
        print("-"*75)
        
        pr_results = {"pr": pr["pr_number"], "conditions": {}}
        
        for condition_name, config in CONDITIONS.items():
            selected, total_methods, error = run_condition(pr, graph, condition_name, config)
            
            if error:
                print(f"{condition_name:<35} ERROR: {error}")
                continue
            
            cmp = compare_with_baseline(selected, baseline_selected)
            
            print(f"{condition_name:<35} {cmp['our_count']:>10} {cmp['overlap_pct']:>9.1f}% "
                  f"{cmp['only_ours']:>10} {cmp['only_baseline']:>10}")
            
            pr_results["conditions"][condition_name] = {
                "selected":      cmp["our_count"],
                "overlap_pct":   cmp["overlap_pct"],
                "only_ours":     cmp["only_ours"],
                "only_baseline": cmp["only_baseline"],
            }
        
        all_results.append(pr_results)
    
    # save results
    with open("evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("Results saved to evaluation_results.json")
    print("="*80)

if __name__ == "__main__":
    main()
