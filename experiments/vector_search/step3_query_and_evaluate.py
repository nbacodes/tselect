# step3_query_and_evaluate.py
import json
import numpy as np
import faiss
from pathlib import Path
import sys

TSELECT_ROOT = Path("/Users/nihalkumar/Desktop/nbaworks/tselect")
sys.path.insert(0, str(TSELECT_ROOT))

# ── 1. LOAD EVERYTHING ───────────────────────────────────────────────────────

print("Loading model, index, metadata...")
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "Qodo/Qodo-Embed-1-1.5B",
    trust_remote_code=True
)

index = faiss.read_index("test_index.faiss")
with open("test_metadata.json") as f:
    all_tests = json.load(f)
with open("changed_functions.json") as f:
    changed_functions = json.load(f)

print(f"Index: {index.ntotal} vectors")
print(f"Changed functions: {len(changed_functions)}")

# ── 2. GROUND TRUTH ───────────────────────────────────────────────────────────
# Tests that actually caught PR #176888 — fill this from your coverage XMLs.
# Format: just the test function names (without class prefix)

GROUND_TRUTH = {
    "test_max_pool2d_with_indices_backward",
    "test_max_pool2d_with_indices_backward2",
    "test_max_pool2d_with_indices_backward3",
    "test_max_pool2d_with_indices_backward4",
    "test_max_pool2d_with_indices_backward5",
    "test_max_pool2d_with_indices_backward6",
    "test_max_pool2d_with_indices_backward_fails",
    "test_max_pool2d1",
    "test_max_pool2d2",
    "test_max_pool2d3",
    "test_max_pool2d4",
    "test_max_pool2d5",
    "test_max_pool2d6",
    "test_max_pool2d7",
    "test_max_pool2d8",
    "test_adaptive_max_pool2d1",
    "test_adaptive_max_pool2d2",
    "test_adaptive_max_pool2d3",
    "test_fractional_max_pool2d1",
    "test_fractional_max_pool2d2",
    "test_fractional_max_pool2d3",
    "test_fractional_max_pool2d4",
    "test_fractional_max_pool2d5",
}

# ── 3. EMBED QUERY VECTORS ────────────────────────────────────────────────────

print("\nEmbedding changed function bodies as query vectors...")
query_texts = [f["body"][:2000] for f in changed_functions]
query_names = [f["name"] for f in changed_functions]

query_embeddings = model.encode(
    query_texts,
    normalize_embeddings=True,
    show_progress_bar=True
)

# ── 4. BATCH SEARCH ───────────────────────────────────────────────────────────

K = 50  # retrieve top 50 per query vector
print(f"\nSearching FAISS index (top {K} per query)...")

scores, indices = index.search(
    np.array(query_embeddings, dtype=np.float32),
    K
)

# ── 5. MERGE RESULTS ─────────────────────────────────────────────────────────
# Union across all query vectors, keep highest score per test

best_score = {}  # test_name -> highest similarity score

for q_idx, (q_name, row_scores, row_indices) in enumerate(
    zip(query_names, scores, indices)
):
    print(f"\nTop 10 for changed function '{q_name}':")
    for rank, (score, idx) in enumerate(zip(row_scores[:10], row_indices[:10])):
        test = all_tests[idx]
        marker = "✓ GT" if test["name"] in GROUND_TRUTH else ""
        print(f"  {rank+1:2d}. [{score:.3f}] {test['file']}::{test['name']} {marker}")

    for score, idx in zip(row_scores, row_indices):
        test_name = all_tests[idx]["name"]
        if test_name not in best_score or score > best_score[test_name]:
            best_score[test_name] = float(score)

# Sort by score
ranked = sorted(best_score.items(), key=lambda x: x[1], reverse=True)

# ── 6. COMPUTE RECALL@K ───────────────────────────────────────────────────────

if GROUND_TRUTH:
    print("\n── Recall Evaluation ──────────────────────")
    for k in [10, 20, 50, 100]:
        top_k_names = {name for name, _ in ranked[:k]}
        hits = GROUND_TRUTH & top_k_names
        recall = len(hits) / len(GROUND_TRUTH)
        print(f"Recall@{k:3d}:  {recall:.1%}  ({len(hits)}/{len(GROUND_TRUTH)} GT tests found)")
    print()
    print("Ground truth tests found in top 50:")
    for name, score in ranked[:50]:
        if name in GROUND_TRUTH:
            print(f"  ✓  [{score:.3f}]  {name}")
    print("Ground truth tests NOT found in top 50:")
    top50_names = {name for name, _ in ranked[:50]}
    for gt in GROUND_TRUTH:
        if gt not in top50_names:
            print(f"  ✗  {gt}")
else:
    print("\nNOTE: GROUND_TRUTH is empty — fill it from your coverage XMLs")
    print("      to get recall numbers. Showing top 20 selected tests:\n")
    for name, score in ranked[:20]:
        test_obj = next(t for t in all_tests if t["name"] == name)
        print(f"  [{score:.3f}]  {test_obj['file']}::{name}")

# ── 7. SAVE RESULTS ───────────────────────────────────────────────────────────

results = {
    "pr": "176888",
    "model": "Qodo-Embed-1-1.5B",
    "corpus_size": len(all_tests),
    "changed_functions": query_names,
    "top_50": [{"name": n, "score": s} for n, s in ranked[:50]]
}
with open("vector_search_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: vector_search_results.json")
