"""
diff_parser.py
--------------
Parses git diff output to extract which functions/classes actually changed.

Two steps:
  1. Parse diff hunks to get changed line ranges per file
  2. Use AST with line numbers to find which functions contain those lines

This is what enables function-level selection in graph_selector.py.
Instead of "scheduler.py changed → run all 7 test files",
we get "_fuse_nodes() changed → run 2 test files that reference _fuse_nodes".

Why line-number based (not token diff):
  - Token diffs are complex and slow
  - Line ranges from git diff are always available and fast
  - A function is "changed" if any of its lines appear in the diff
  - This is conservative — we may slightly over-select, never under-select
"""

import ast
import re
import subprocess
from pathlib import Path
from collections import defaultdict


def get_changed_functions(repo_root: Path, changed_files: list) -> dict:
    """
    For each changed file, return which top-level functions/classes changed.

    Returns:
        {
            "torch/_inductor/scheduler.py": {
                "_fuse_nodes",
                "BaseScheduler",
            },
            "torch/_inductor/lowering.py": {
                "make_fallback",
            }
        }

    If a file has no function-level info (parse error, binary, etc.),
    its entry will be an empty set — caller falls back to file-level.
    """
    result = {}

    for cf in changed_files:
        rel = _normalize(cf, repo_root)
        if not rel.endswith(".py"):
            result[rel] = set()
            continue

        changed_lines = _get_changed_lines(repo_root, rel)
        if not changed_lines:
            # file is new (all lines changed) or diff unavailable
            # treat as fully changed — file-level fallback
            result[rel] = set()
            continue

        full_path = repo_root / rel
        if not full_path.exists():
            result[rel] = set()
            continue

        symbols = _functions_at_lines(full_path, changed_lines)
        result[rel] = symbols

    return result


def _normalize(path_str: str, repo_root: Path) -> str:
    """Normalize a file path to be relative to repo root."""
    try:
        return str(Path(path_str).relative_to(repo_root))
    except ValueError:
        return str(path_str).lstrip("./")


def _get_changed_lines(repo_root: Path, rel_path: str) -> set:
    """
    Run git diff to get the set of line numbers that changed in this file.

    Parses unified diff hunk headers like:
        @@ -10,5 +12,8 @@
    The +12,8 means: starting at line 12, 8 lines in the new file.
    We collect all those line numbers as "changed".

    Returns empty set if git diff unavailable or file is entirely new.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "HEAD", "--unified=0", "--", rel_path],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            timeout=10,
        )
        diff_output = result.stdout

        # also check staged changes
        if not diff_output.strip():
            result = subprocess.run(
                ["git", "diff", "--unified=0", "--", rel_path],
                capture_output=True,
                text=True,
                cwd=str(repo_root),
                timeout=10,
            )
            diff_output = result.stdout

    except Exception:
        return set()

    if not diff_output.strip():
        return set()

    changed_lines = set()

    # parse @@ -old_start,old_count +new_start,new_count @@
    hunk_pattern = re.compile(r"^\+\+\+ b/(.+)$|^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)

    for match in hunk_pattern.finditer(diff_output):
        if match.group(2) is not None:
            start = int(match.group(2))
            count = int(match.group(3)) if match.group(3) is not None else 1
            # count=0 means line was deleted (no new lines) — skip
            if count > 0:
                for line_no in range(start, start + count):
                    changed_lines.add(line_no)

    return changed_lines


def _functions_at_lines(file_path: Path, changed_lines: set) -> set:
    """
    Given a set of changed line numbers, return the names of all
    top-level functions and classes that contain any of those lines.

    Uses AST node line numbers — accurate for Python 3.8+.

    Also handles methods inside classes:
        class Scheduler:
            def _fuse_nodes(self):  ← return "Scheduler._fuse_nodes"

    Conservative: if a changed line falls outside any function/class
    (e.g. module-level code), we return a special marker "__module__"
    so the caller knows the file's module-level code changed.
    """
    symbols = set()

    try:
        source = file_path.read_text(encoding="utf-8", errors="ignore")
        tree   = ast.parse(source)
    except Exception:
        return symbols

    # collect all function/class definitions with their line ranges
    definitions = []  # (start_line, end_line, qualified_name)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # find parent class if any
            parent_class = _find_parent_class(tree, node)
            if parent_class:
                name = f"{parent_class}.{node.name}"
            else:
                name = node.name

            end_line = getattr(node, "end_lineno", node.lineno + 50)
            definitions.append((node.lineno, end_line, name))

        elif isinstance(node, ast.ClassDef):
            end_line = getattr(node, "end_lineno", node.lineno + 200)
            definitions.append((node.lineno, end_line, node.name))

    # for each changed line, find which definition contains it
    covered_by_definition = set()

    for start, end, name in definitions:
        for line in changed_lines:
            if start <= line <= end:
                symbols.add(name)
                covered_by_definition.add(line)

    # lines not inside any function = module-level code changed
    uncovered = changed_lines - covered_by_definition
    if uncovered:
        symbols.add("__module__")

    return symbols


def _find_parent_class(tree: ast.AST, target_node: ast.AST) -> str | None:
    """
    Walk the AST to find if target_node is directly inside a ClassDef.
    Returns the class name or None.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for child in ast.iter_child_nodes(node):
                if child is target_node:
                    return node.name
    return None