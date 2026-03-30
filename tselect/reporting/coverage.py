"""
reporting/coverage.py
---------------------
Handles coverage measurement and diff_cover integration.

Flow:
    1. prepare_coverage() writes .tselect_coveragerc (no plugins)
       and sets COVERAGE_RCFILE env var before pytest runs
    2. pytest runs with --cov flags → generates coverage.xml
    3. run_diff_cover() runs diff-cover → JSON report
    4. format_coverage_section() formats output for summary
"""

import json
import os
import subprocess
from pathlib import Path


COVERAGE_XML     = "coverage.xml"
COVERAGE_JSON    = ".tselect_coverage_report.json"
COVERAGE_RC_FILE = ".tselect_coveragerc"


def prepare_coverage(repo_root: Path, source_dir: str = ".") -> list:
    """
    Writes a minimal .coveragerc with no plugins.
    Sets COVERAGE_RCFILE env var so pytest-cov uses it instead of
    the repo's own coverage config (e.g. PyTorch's coverage_plugins).

    Returns list of args to append to the pytest command.
    """
    coveragerc_path = repo_root / COVERAGE_RC_FILE

    rc_lines = [
        "[run]",
        "source = " + source_dir,
        "branch = true",
        "omit =",
        "    */test*",
        "    */__pycache__/*",
        "",
    ]
    coveragerc_path.write_text("\n".join(rc_lines))

    # set BEFORE pytest subprocess starts — overrides repo coverage config
    os.environ["COVERAGE_RCFILE"] = str(coveragerc_path)

    return [
        "--cov=" + source_dir,
        "--cov-report=xml",
        "--cov-branch",
        "--cov-config=" + str(coveragerc_path),
    ]


def run_diff_cover(
    repo_root: Path,
    compare_branch: str = "origin/main",
) -> dict | None:
    """
    Run diff-cover against coverage.xml.

    Returns:
        {
            "total_percent_covered": 87,
            "src_stats": {
                "torch/_inductor/lowering.py": {
                    "percent_covered": 91,
                    "covered_lines":   [7790, 7791, 7802],
                    "missing_lines":   [7822],
                }
            }
        }
    Returns None if diff-cover not installed or coverage.xml missing.
    """
    coverage_xml  = repo_root / COVERAGE_XML
    coverage_json = repo_root / COVERAGE_JSON

    if not coverage_xml.exists():
        print("  coverage.xml not found — skipping diff-cover")
        return None

    try:
        result = subprocess.run(
            [
                "diff-cover",
                str(coverage_xml),
                "--compare-branch=" + compare_branch,
                "--format=json:" + str(coverage_json),
                "--quiet",
            ],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            timeout=60,
        )

        # returncode 1 = below threshold, still valid
        if result.returncode not in (0, 1):
            print("  diff-cover error: " + result.stderr.strip()[:200])
            return None

    except FileNotFoundError:
        print("  diff-cover not installed — run: pip install diff-cover")
        return None
    except subprocess.TimeoutExpired:
        print("  diff-cover timed out")
        return None
    except Exception as e:
        print("  diff-cover failed: " + str(e))
        return None

    if not coverage_json.exists():
        return None

    try:
        with open(coverage_json) as f:
            data = json.load(f)
        coverage_json.unlink(missing_ok=True)
        return data
    except Exception as e:
        print("  Could not parse diff-cover output: " + str(e))
        return None


def format_coverage_section(coverage_data: dict, W: int = 70) -> list:
    """
    Returns lines to print in the summary coverage section.
    Called by generate_summary — does not print directly.
    """
    lines = []
    total = coverage_data.get("total_percent_covered", None)
    if total is None:
        return lines

    lines.append("  " + "─" * (W - 4))
    lines.append("  Coverage of Changed Lines (diff-cover)")
    lines.append("  " + "─" * (W - 4))

    bar_len = 30
    filled  = int(bar_len * total / 100)
    bar     = "█" * filled + "░" * (bar_len - filled)
    icon    = "✅" if total >= 80 else "⚠️ " if total >= 60 else "❌"
    lines.append(f"  {icon}  {total}%  [{bar}]")
    lines.append("")

    src_stats = coverage_data.get("src_stats", {})
    if src_stats:
        lines.append("  Per-file breakdown:")
        for filepath, stats in sorted(src_stats.items()):
            pct     = stats.get("percent_covered", 0)
            missing = stats.get("missing_lines", [])
            f_icon  = "✅" if pct == 100 else "⚠️ " if pct >= 60 else "❌"
            fname   = filepath if len(filepath) <= 50 else "..." + filepath[-47:]
            if missing:
                miss_str = ", ".join(str(l) for l in missing[:5])
                if len(missing) > 5:
                    miss_str += f" +{len(missing)-5} more"
                lines.append(f"  {f_icon}  {fname:<50}  {pct:>3}%  (uncovered: {miss_str})")
            else:
                lines.append(f"  {f_icon}  {fname:<50}  {pct:>3}%")

    lines.append("")
    return lines