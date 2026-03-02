"""
pytest_adapter.py
-----------------
Builds and executes pytest commands for tselect.

Key design decisions:
  - Uses direct node IDs (file::Class::method) — no -k expression
    which breaks with 100+ classes due to shell length limits
  - Live output via Popen so user sees test results in real time
  - Simultaneously captures output for pass/fail/skip parsing
"""

import re
import subprocess
import sys
from pathlib import Path


def build_pytest_command(node_ids: list[str], extra_args: list[str] = None) -> list[str]:
    """
    Build pytest command from a list of node IDs.

    node_ids are exact pytest node IDs like:
        "test/inductor/test_scheduler.py::TestSchedulerCPU::test_foo"

    These come from pytest --collect-only during build-graph,
    so they are guaranteed to be valid and runnable.
    """
    if not node_ids:
        return []

    cmd = [sys.executable, "-m", "pytest"] + node_ids + ["--no-header", "-rN", "--tb=short"]

    if extra_args:
        cmd += extra_args

    return cmd


def build_pytest_command_from_classes(selected_classes: list[str]) -> list[str]:
    """
    Legacy builder: used when falling back to ownership.yaml + manual JSON.
    selected_classes are "file::ClassName" strings (no method level).
    """
    if not selected_classes:
        return []

    # detect if these are file::class (graph) or just class names (legacy)
    if all("::" in c for c in selected_classes):
        # file::class format — pass directly as node IDs
        return [sys.executable, "-m", "pytest"] + selected_classes + ["--no-header", "-rN", "--tb=short"]
    else:
        # plain class names — need -k expression
        k_expr = " or ".join(selected_classes)
        return [sys.executable, "-m", "pytest", "-k", k_expr, "--no-header", "-rN", "--tb=short"]


def execute_command(cmd: list[str]) -> tuple[int, int, int, int]:
    """
    Execute a pytest command with:
      - Live output (user sees results in real time)
      - Captured output (for pass/fail/skip parsing)

    Returns: (return_code, passed, failed, skipped)
    """
    if not cmd:
        print("No tests to run.")
        return 0, 0, 0, 0

    # Popen gives live output + capture simultaneously
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge stderr into stdout
        text=True,
        cwd=str(Path.cwd()),
    )

    output_lines = []

    # stream live to terminal and capture simultaneously
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