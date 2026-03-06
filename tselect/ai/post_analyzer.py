"""
tselect/ai/post_analyzer.py
---------------------------
LLM-based post-analysis that runs AFTER pytest, only when tests FAIL.

Takes failed test names + changed files + tracebacks and asks:
  - What is the root cause?
  - Which changed file/symbol caused it?
  - What is the suggested fix?

This is the high-value layer — saves developer time when something breaks.
"""

import json
import re
from tselect.ai.llm_client import LLMClient
from tselect.utils.logger import setup_logger

logger = setup_logger()

SYSTEM_PROMPT = """You are a test failure analyst for a Python codebase.
Given changed files and failed test names, identify the root cause.
Respond ONLY in valid JSON. No text outside the JSON object."""

ANALYSIS_PROMPT_TEMPLATE = """
CHANGED FILES AND SYMBOLS:
{changed_summary}

TEST RESULTS:
  Passed : {passed}
  Failed : {failed}
  Skipped: {skipped}

FAILED TESTS:
{failed_tests}

{traceback_section}

Given the above, identify the most likely root cause of these failures.

Respond ONLY in this exact JSON format:
{{
  "root_cause_file":   "most likely changed file that caused failures",
  "root_cause_symbol": "specific function or class that broke",
  "failure_pattern":   "one line description of what broke",
  "explanation":       "2-3 sentence explanation linking the change to the failures",
  "suggested_fix":     "concrete, actionable suggestion for the developer",
  "confidence":        0.0 to 1.0
}}"""


def _build_changed_summary(changed_files: list, changed_symbols: dict) -> str:
    lines = []
    for f in changed_files:
        syms = changed_symbols.get(f, set())
        clean_syms = {s for s in syms if s != "__module__"}
        if clean_syms:
            sym_str = ", ".join(sorted(clean_syms)[:8])
            lines.append(f"  - {f}  (changed: {sym_str})")
        else:
            lines.append(f"  - {f}")
    return "\n".join(lines)


def _build_traceback_section(tracebacks: dict) -> str:
    """
    tracebacks: {test_node_id: traceback_string}
    Truncates each to 400 chars to stay within token budget.
    """
    if not tracebacks:
        return ""

    lines = ["FAILURE TRACEBACKS (truncated):"]
    for test_id, tb in list(tracebacks.items())[:5]:  # max 5 tracebacks
        short_tb = tb.strip()[:400].replace("\n", "\n  ")
        lines.append(f"\n[{test_id}]\n  {short_tb}")

    return "\n".join(lines)


def _parse_analysis(raw: str) -> dict | None:
    if raw is None:
        return None

    raw = re.sub(r"```(?:json)?", "", raw).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return None


class PostAnalyzer:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def analyze(
        self,
        failed_tests:    list,
        changed_files:   list,
        changed_symbols: dict,
        passed:          int = 0,
        failed:          int = 0,
        skipped:         int = 0,
        tracebacks:      dict = None,
    ) -> dict | None:
        """
        Analyze test failures and return root cause + fix suggestion.

        Args:
            failed_tests:    list of failed pytest node IDs
            changed_files:   list of changed source file paths
            changed_symbols: {file: set(symbols)} from diff_parser
            passed/failed/skipped: test counts
            tracebacks:      {node_id: traceback_text} optional

        Returns:
            dict with root_cause_file, explanation, suggested_fix etc.
            or None if analysis fails
        """
        if not failed_tests:
            return None

        changed_summary   = _build_changed_summary(changed_files, changed_symbols)
        failed_tests_str  = "\n".join(f"  - {t}" for t in failed_tests[:20])
        traceback_section = _build_traceback_section(tracebacks or {})

        prompt = SYSTEM_PROMPT + ANALYSIS_PROMPT_TEMPLATE.format(
            changed_summary   = changed_summary,
            passed            = passed,
            failed            = failed,
            skipped           = skipped,
            failed_tests      = failed_tests_str,
            traceback_section = traceback_section,
        )

        raw      = self.llm.safe_complete(prompt)
        analysis = _parse_analysis(raw)

        if analysis is None:
            logger.debug("Post-analysis parse failed — skipping")
            return None

        return analysis
