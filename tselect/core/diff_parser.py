"""
diff_parser.py
--------------
Parses git diff output to extract which functions/classes actually changed.

Supports PR mode (base...HEAD) and avoids incorrect fallbacks.

Now delegates symbol extraction to fn_diff.py (tree-sitter based),
which supports .py, .cpp, .cu, .cuh, .h, .hpp files.
"""

from typing import Optional
import re
import subprocess
from pathlib import Path

from tselect.core.fn_diff import extract_symbols_at_lines

# Extensions we attempt function-level extraction for
SUPPORTED_EXTENSIONS = {'.py', '.cpp', '.cu', '.cuh', '.h', '.hpp', '.cc', '.c'}


def get_changed_functions(repo_root: Path, changed_files: list, base="upstream/main") -> dict:
    """
    For each changed file, return which top-level functions/classes changed.

    Returns:
        {
            "torch/_inductor/scheduler.py": {"_fuse_nodes", "BaseScheduler"},
            "torch/csrc/jit/runtime/interpreter.cpp": {"InterpreterState::run"},
        }

    If a file has no function-level info, returns {"__unknown__"}.
    Non-code / unsupported files return set() so the caller skips them.
    """
    result = {}

    for cf in changed_files:
        rel = _normalize(cf, repo_root)

        suffix = Path(rel).suffix.lower()

        # Unsupported extension → return empty set (caller handles gracefully)
        if suffix not in SUPPORTED_EXTENSIONS:
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

        # Delegate to fn_diff (tree-sitter based, with ast fallback for .py)
        symbols = extract_symbols_at_lines(full_path, changed_lines)

        result[rel] = symbols if symbols else {"__unknown__"}

    return result


def _normalize(path_str: str, repo_root: Path) -> str:
    try:
        return str(Path(path_str).relative_to(repo_root))
    except ValueError:
        return str(path_str).lstrip("./")


def _get_changed_lines(repo_root: Path, rel_path: str, base: str) -> set:
    """
    Extract changed line numbers using git diff base...HEAD
    Returns 1-based line numbers matching what tree-sitter/ast produce.
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
        count = int(match.group(2)) if match.group(2) is not None else 1

        if count > 0:
            for line_no in range(start, start + count):
                changed_lines.add(line_no)

    return changed_lines