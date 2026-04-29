"""
step1_extract_changed_functions.py
-----------------------------------
Extracts changed function bodies from PR #176888 using tselect's
existing diff_parser + tree-sitter infrastructure.
"""
import json
import sys
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
PYTORCH_ROOT = Path("/Users/nihalkumar/pytorch-pr-176888")
TSELECT_ROOT = Path("/Users/nihalkumar/Desktop/nbaworks/tselect")
sys.path.insert(0, str(TSELECT_ROOT))

from tselect.core.diff_parser import get_changed_functions

# ── 1. get changed function NAMES via tselect ────────────────────────────────
changed_file = "torch/_inductor/lowering.py"

print(f"Getting changed functions from {changed_file}...")
changed = get_changed_functions(
    repo_root=PYTORCH_ROOT,
    changed_files=[changed_file],
    base="upstream/main"
)

func_names = changed.get(changed_file, set())
print(f"Changed functions: {func_names}")

# ── 2. extract full bodies using tree-sitter ─────────────────────────────────
try:
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser

    PY_LANGUAGE = Language(tspython.language())
    parser = Parser(PY_LANGUAGE)

    source = (PYTORCH_ROOT / changed_file).read_bytes()
    tree = parser.parse(source)
    source_str = source.decode("utf-8", errors="replace")

    def get_all_functions(node, results=None):
        if results is None:
            results = []
        if node.type in ("function_definition", "decorated_definition"):
            # get name
            for child in node.children:
                if child.type == "function_definition":
                    name_node = child.child_by_field_name("name")
                    if name_node:
                        name = source_str[name_node.start_byte:name_node.end_byte]
                        body = source_str[node.start_byte:node.end_byte]
                        results.append({"name": name, "body": body})
                    break
            else:
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = source_str[name_node.start_byte:name_node.end_byte]
                    body = source_str[node.start_byte:node.end_byte]
                    results.append({"name": name, "body": body})
        for child in node.children:
            get_all_functions(child, results)
        return results

    all_funcs = get_all_functions(tree.root_node)
    print(f"Total functions in file: {len(all_funcs)}")

    # filter to only changed ones
    changed_bodies = [
        {
            "name": f["name"],
            "body": f["body"][:3000],   # cap at 3000 chars
            "file": changed_file,
            "body_length": len(f["body"])
        }
        for f in all_funcs
        if f["name"] in func_names
    ]

except Exception as e:
    print(f"tree-sitter extraction failed: {e}")
    print("Falling back to raw diff text per function...")

    # fallback: use the diff lines themselves as the "body"
    import subprocess
    result = subprocess.run(
        ["git", "diff", "--unified=10", "upstream/main...HEAD",
         "--", changed_file],
        capture_output=True, text=True, cwd=str(PYTORCH_ROOT)
    )
    diff_text = result.stdout

    # split diff into per-function chunks based on @@ headers
    chunks = []
    current_func = None
    current_lines = []
    for line in diff_text.splitlines():
        if "@@" in line and "def " in line:
            if current_func and current_lines:
                chunks.append((current_func, "\n".join(current_lines)))
            current_func = line.split("def ")[-1].split("(")[0].strip()
            current_lines = [line]
        elif current_func:
            current_lines.append(line)
    if current_func and current_lines:
        chunks.append((current_func, "\n".join(current_lines)))

    changed_bodies = [
        {"name": name, "body": body[:3000], "file": changed_file}
        for name, body in chunks
        if name in func_names
    ]

# ── 3. report + save ─────────────────────────────────────────────────────────
print(f"\nExtracted {len(changed_bodies)} changed function bodies:")
for f in changed_bodies:
    print(f"  {f['name']}  ({len(f['body'])} chars)")
    print(f"  Preview: {f['body'][:120].strip()!r}")
    print()

out = Path("changed_functions.json")
out.write_text(json.dumps(changed_bodies, indent=2))
print(f"Saved → {out}")
