import re
import subprocess
from pathlib import Path
import sys


def build_pytest_command(selected_classes):
    """
    Build pytest command using -k filter instead of ::Class.
    """

    files = set()
    class_names = []

    for item in selected_classes:
        file_part, class_part = item.split("::")
        files.add(file_part)
        class_names.append(class_part)

    # Build OR filter
    k_expr = " or ".join(class_names)

    cmd = [sys.executable, "-m", "pytest"]   # ✅ safer than "pytest"

    cmd += list(files)

    cmd += ["-k", f"({k_expr})"]

    cmd += ["-vv"]          # live progress + detailed output
    cmd += ["--color=yes"]  # nice colored logs

    return cmd


def execute_command(cmd):
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    full_output = ""

    # 🔥 STREAM OUTPUT LIVE
    for line in process.stdout:
        print(line, end="")     # realtime logs
        full_output += line

    process.wait()

    passed = failed = skipped = 0

    passed_match = re.search(r"(\d+)\s+passed", full_output, re.IGNORECASE)
    failed_match = re.search(r"(\d+)\s+failed", full_output, re.IGNORECASE)
    skipped_match = re.search(r"(\d+)\s+skipped", full_output, re.IGNORECASE)

    if passed_match:
        passed = int(passed_match.group(1))

    if failed_match:
        failed = int(failed_match.group(1))

    if skipped_match:
        skipped = int(skipped_match.group(1))

    return process.returncode, passed, failed, skipped
