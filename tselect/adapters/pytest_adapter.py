"""
pytest_adapter.py
-----------------
tselect's job is TEST SELECTION. Execution is just pytest doing its thing.

Strategy:
  - Pass TEST FILES to pytest (not individual node IDs)
  - 20 files fit easily in subprocess args — no OS limit
  - --continue-on-collection-errors — skip broken files, run the rest
  - Failed tests are visible in live output + parsed for summary
  - If a test file has a broken import, pytest skips it and moves on
"""

import re
import subprocess
import sys
from pathlib import Path


def build_pytest_command(node_ids: list[str], extra_args: list[str] = None) -> list[str]:
    """
    Extract unique test files from node IDs and build a file-level pytest command.

    Why file-level not method-level:
      - 3592 node IDs → OSError (arg list too long)
      - 20 unique test files → works fine
      - pytest collects the right tests from those files anyway
      - --continue-on-collection-errors skips broken files cleanly
    """
    if not node_ids:
        return []

    # extract unique test files from node_ids
    test_files = sorted(set(nid.split("::")[0] for nid in node_ids))

    args = (
        [sys.executable, "-m", "pytest"]
        + test_files
        + ["--continue-on-collection-errors", "--tb=short", "--no-header"]
    )

    if extra_args:
        args += extra_args

    return args


def build_pytest_command_from_classes(selected_classes: list[str]) -> list[str]:
    """Legacy builder for ownership.yaml + manual JSON fallback mode."""
    if not selected_classes:
        return []

    test_files = sorted(set(c.split("::")[0] for c in selected_classes))
    return (
        [sys.executable, "-m", "pytest"]
        + test_files
        + ["--continue-on-collection-errors", "--tb=short", "--no-header"]
    )


def execute_command(cmd: list[str]) -> tuple[int, int, int, int]:
    """
    Execute pytest via subprocess with live output.

    --continue-on-collection-errors means:
      - broken import in test_inplacing_pass.py → skip it, run the rest
      - distributed tests that need CUDA → skip, run the rest
      - user sees exactly which files were skipped and why

    Returns: (return_code, passed, failed, skipped)
    """
    if not cmd:
        print("No tests to run.")
        return 0, 0, 0, 0

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(Path.cwd()),
    )

    output_lines = []
    for line in process.stdout:
        print(line, end="", flush=True)
        output_lines.append(line)

    process.wait()
    output = "".join(output_lines)

    passed  = _parse_count(output, r"(\d+) passed")
    failed  = _parse_count(output, r"(\d+) failed")
    skipped = _parse_count(output, r"(\d+) skipped")

    return process.returncode, passed, failed, skipped


def _parse_count(text: str, pattern: str) -> int:
    match = re.search(pattern, text)
    return int(match.group(1)) if match else 0