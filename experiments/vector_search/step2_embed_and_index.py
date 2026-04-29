"""
step2_embed_and_index.py
-------------------------
Extracts all test functions from inductor test files,
embeds them with Qodo-Embed-1-1.5B, builds FAISS index.
"""
import json
import numpy as np
import faiss
import time
import sys
from pathlib import Path

PYTORCH_ROOT = Path("/Users/nihalkumar/pytorch-pr-176888")

# ── 1. LOAD MODEL ─────────────────────────────────────────────────────────────
print("Loading Qodo-Embed-1-1.5B... (downloads ~3GB on first run)")
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "nomic-ai/nomic-embed-code",
    trust_remote_code=True
)
print("Model loaded.\n")

# ── 2. FIND INDUCTOR TEST FILES ───────────────────────────────────────────────
test_root = PYTORCH_ROOT / "test/inductor"
test_files = sorted(test_root.glob("test_*.py"))
print(f"Found {len(test_files)} inductor test files:")
for f in test_files:
    print(f"  {f.name}")

# ── 3. EXTRACT TEST FUNCTIONS VIA TREE-SITTER ─────────────────────────────────
print("\nExtracting test functions...")

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tspython.language())
ts_parser = Parser(PY_LANGUAGE)

def extract_test_functions(filepath: Path) -> list:
    """Extract all test_* functions from a Python file using tree-sitter."""
    try:
        source_bytes = filepath.read_bytes()
        source_str   = source_bytes.decode("utf-8", errors="replace")
        tree = ts_parser.parse(source_bytes)
    except Exception as e:
        print(f"  WARNING: could not parse {filepath.name}: {e}")
        return []

    results = []

    def walk(node):
        if node.type in ("function_definition", "decorated_definition"):
            # get the actual function_definition node
            fn_node = node
            if node.type == "decorated_definition":
                for child in node.children:
                    if child.type == "function_definition":
                        fn_node = child
                        break

            name_node = fn_node.child_by_field_name("name")
            if name_node:
                name = source_str[name_node.start_byte:name_node.end_byte]
                if name.startswith("test_"):
                    body = source_str[node.start_byte:node.end_byte]
                    results.append({
                        "name": name,
                        "body": body[:2000],   # cap at 2000 chars
                        "file": filepath.name,
                        "full_path": str(filepath),
                    })
                    return  # don't recurse into test body

        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return results


all_tests = []
for tf in test_files:
    funcs = extract_test_functions(tf)
    all_tests.extend(funcs)
    print(f"  {tf.name}: {len(funcs)} test functions")

print(f"\nTotal test functions extracted: {len(all_tests)}")

if len(all_tests) == 0:
    print("ERROR: No test functions found. Check PYTORCH_ROOT path.")
    sys.exit(1)

# ── 4. EMBED ──────────────────────────────────────────────────────────────────
print(f"\nEmbedding {len(all_tests)} functions (this takes 10-20 mins on CPU)...")

bodies = [t["body"] for t in all_tests]

start = time.time()
embeddings = model.encode(
    bodies,
    batch_size=8,
    show_progress_bar=True,
    normalize_embeddings=True,   # cosine similarity via inner product
)
elapsed = time.time() - start
print(f"\nEmbedding done in {elapsed:.1f}s")
print(f"Embedding shape: {embeddings.shape}")

# ── 5. BUILD FAISS INDEX ──────────────────────────────────────────────────────
dim = embeddings.shape[1]
print(f"\nBuilding FAISS index (dim={dim})...")

index = faiss.IndexFlatIP(dim)   # inner product on normalized = cosine
index.add(np.array(embeddings, dtype=np.float32))
print(f"FAISS index built. Total vectors: {index.ntotal}")

# ── 6. SAVE ───────────────────────────────────────────────────────────────────
faiss.write_index(index, "test_index.faiss")
print("Saved → test_index.faiss")

with open("test_metadata.json", "w") as f:
    json.dump(all_tests, f, indent=2)
print("Saved → test_metadata.json")

print("\nDone. Run step3 next.")
