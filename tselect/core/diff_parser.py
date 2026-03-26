
from typing import Optional
"""
diff_parser.py
--------------
Parses git diff output to extract which functions/classes actually changed.

Supports PR mode (base...HEAD) and avoids incorrect fallbacks.
"""

import ast
import re
import subprocess
from pathlib import Path


def get_changed_functions(repo_root: Path, changed_files: list, base="upstream/main") -> dict:
    """
    For each changed file, return which top-level functions/classes changed.

    Returns:
        {
            "torch/_inductor/scheduler.py": {
                "_fuse_nodes",
                "BaseScheduler",
            },
        }

    If a file has no function-level info, returns {"__unknown__"}.
    """
    result = {}

    for cf in changed_files:
        rel = _normalize(cf, repo_root)

        if not rel.endswith(".py"):
            result[rel] = set()
            continue

        changed_lines = _get_changed_lines(repo_root, rel, base)

        if not changed_lines:
            print(f"[WARN] No changed lines detected for {rel}")
            result[rel] = {"__unknown__"}
            continue

        full_path = repo_root / rel
        if not full_path.exists():
            result[rel] = {"__unknown__"}
            continue

        symbols = _functions_at_lines(full_path, changed_lines)

        if not symbols:
            symbols = {"__unknown__"}

        result[rel] = symbols

    return result


def _normalize(path_str: str, repo_root: Path) -> str:
    try:
        return str(Path(path_str).relative_to(repo_root))
    except ValueError:
        return str(path_str).lstrip("./")


def _get_changed_lines(repo_root: Path, rel_path: str, base: str) -> set:
    """
    Extract changed line numbers using git diff base...HEAD
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--unified=0", f"{base}...HEAD", "--", rel_path],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            timeout=10,
        )
        diff_output = result.stdout

    except Exception as e:
        print(f"[ERROR] git diff failed for {rel_path}: {e}")
        return set()

    if not diff_output.strip():
        return set()

    changed_lines = set()

    hunk_pattern = re.compile(
        r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@",
        re.MULTILINE
    )

    for match in hunk_pattern.finditer(diff_output):
        start = int(match.group(1))
        count = int(match.group(2)) if match.group(2) else 1

        if count > 0:
            for line_no in range(start, start + count):
                changed_lines.add(line_no)

    return changed_lines


def _functions_at_lines(file_path: Path, changed_lines: set) -> set:
    symbols = set()

    try:
        source = file_path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except Exception:
        return symbols

    definitions = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parent_class = _find_parent_class(tree, node)

            if parent_class:
                name = f"{parent_class}.{node.name}"
            else:
                name = node.name

            end_line = getattr(node, "end_lineno", None)
            if end_line is None:
                # fallback: assume large function, not small
                end_line = node.lineno + 1000
            definitions.append((node.lineno, end_line, name))

        elif isinstance(node, ast.ClassDef):
            end_line = getattr(node, "end_lineno", node.lineno + 200)
            definitions.append((node.lineno, end_line, node.name))

    covered_lines = set()

    for start, end, name in definitions:
        for line in changed_lines:
            if start <= line <= end:
                symbols.add(name)
                covered_lines.add(line)

    uncovered = changed_lines - covered_lines
    if uncovered:
        symbols.add("__module__")

    return symbols


def _find_parent_class(tree: ast.AST, target_node: ast.AST) -> Optional[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for child in ast.iter_child_nodes(node):
                if child is target_node:
                    return node.name
    return None
