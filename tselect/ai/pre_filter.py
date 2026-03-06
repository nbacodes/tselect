"""
tselect/ai/pre_filter.py
------------------------
LLM-based pre-filter that runs AFTER rule-based graph selection.

Takes the graph selector's output (candidate test files + their triggered_by
info) and asks Groq/Llama: "does this test actually need to run?"

Design principles:
  - Never under-selects: if confidence < threshold, keep the test
  - If LLM fails entirely, keep ALL candidates (fail open)
  - Only sends file names + class names (no raw code, no full diff)
    Paper finding: class name alone is the best feature, raw code = noise

Input:  selected dict from graph_selector.select_tests_from_graph()
Output: filtered selected dict (same structure, subset of input)
"""

import json
import re
from tselect.ai.llm_client import LLMClient, LLMClientError
from tselect.utils.logger import setup_logger

logger = setup_logger()

SYSTEM_PROMPT = """You are a precise test selection assistant for a Python codebase.
Your job is to determine if a test file DIRECTLY tests the changed functionality.

RULES:
- Only return should_run: true if test classes DIRECTLY test the changed symbols
- If the connection is indirect, coincidental, or just a shared import, return should_run: false
- Be precise, not conservative — false positives waste CI time
- Always give a confidence above 0.7 if you have a clear reason
Respond ONLY in valid JSON. No text outside the JSON object."""

CANDIDATE_PROMPT_TEMPLATE = """
CHANGED SOURCE FILES AND SYMBOLS:
{changed_summary}

CANDIDATE TEST FILE: {test_file}
TRIGGERED BY: {triggered_by}
TEST CLASSES:
{class_summary}

QUESTION:
Do any of these test classes DIRECTLY test the changed symbols listed above?
Or do they test something else that is unrelated to those specific changes?

Respond ONLY in this exact JSON format:
{{
  "should_run": true or false,
  "confidence": 0.0 to 1.0,
  "reason": "one sentence max",
  "relevant_classes": ["ClassName"] or []
}}"""


def _build_changed_summary(changed_files: list, changed_symbols: dict) -> str:
    """
    Build a concise summary of what changed, including module context.
    changed_symbols: {rel_path: set(symbol_names)} from diff_parser
    """
    lines = []
    for f in changed_files:
        syms  = changed_symbols.get(f, set())
        desc  = _module_description(f)
        clean = {s for s in syms if s != "__module__"}
        if clean:
            sym_str = ", ".join(sorted(clean)[:8])
            lines.append(f"  - {f}\n    changed: {sym_str}\n    ({desc})")
        else:
            lines.append(f"  - {f}\n    ({desc})")
    return "\n".join(lines)


def _module_description(file_path: str) -> str:
    """
    Return a brief description of what a module does.
    LLM uses this to reason about relevance more accurately.
    """
    KNOWN_MODULES = {
        "distributed":   "distributed collective communication (NCCL/Gloo ops)",
        "autograd":      "automatic differentiation engine",
        "optim":         "optimization algorithms (SGD, Adam, etc.)",
        "nn/modules":    "neural network layer implementations",
        "nn/functional": "neural network functional operations",
        "cuda":          "CUDA device management and streams",
        "jit":           "TorchScript JIT compiler",
        "quantization":  "model quantization utilities",
        "sparse":        "sparse tensor operations",
        "linalg":        "linear algebra operations",
        "fft":           "fast Fourier transform operations",
        "testing":       "test utilities and helpers",
    }
    for key, desc in KNOWN_MODULES.items():
        if key in file_path:
            return desc
    # fallback: derive from file name
    name = file_path.split("/")[-1].replace(".py", "").replace("_", " ")
    return f"{name} module"


def _build_class_summary(classes: dict) -> str:
    lines = []
    for cls_name, cls_data in classes.items():
        count = cls_data.get("test_count", 0)
        lines.append(f"  - {cls_name} ({count} tests)")
    return "\n".join(lines)


def _parse_decision(raw: str) -> dict | None:
    """
    Parse LLM JSON response. Returns None if unparseable.
    Handles LLMs that sometimes wrap JSON in markdown code blocks.
    """
    if raw is None:
        return None

    # strip markdown code fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    try:
        data = json.loads(raw)
        # validate required fields
        if "should_run" not in data or "confidence" not in data:
            return None
        return data
    except json.JSONDecodeError:
        # try to extract JSON object from messy response
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return None


class PreFilter:
    def __init__(self, llm: LLMClient, config: dict):
        self.llm       = llm
        ai_cfg         = config.get("ai", {})
        self.threshold = ai_cfg.get("confidence_threshold", 0.75)

    def filter(
        self,
        selected: dict,
        changed_files: list,
        changed_symbols: dict,
    ) -> tuple[dict, list]:
        """
        Filter candidate test files using LLM.

        Args:
            selected:        output of select_tests_from_graph()
            changed_files:   list of changed source file paths
            changed_symbols: {file: set(symbols)} from diff_parser

        Returns:
            (filtered_selected, ai_decisions)
            filtered_selected: same structure as selected, subset of input
            ai_decisions:      list of decision dicts for summary reporting
        """
        if not selected:
            return selected, []

        changed_summary = _build_changed_summary(changed_files, changed_symbols)
        filtered        = {}
        ai_decisions    = []

        print(f"\n  🤖 AI pre-filtering {len(selected)} candidate test files...")

        for test_file, data in selected.items():
            triggered_by  = data.get("triggered_by", [])
            triggered_str = ", ".join(triggered_by) if isinstance(triggered_by, list) else str(triggered_by)
            class_summary = _build_class_summary(data.get("classes", {}))

            prompt = SYSTEM_PROMPT + CANDIDATE_PROMPT_TEMPLATE.format(
                changed_summary = changed_summary,
                test_file       = test_file,
                triggered_by    = triggered_str,
                class_summary   = class_summary,
            )

            raw      = self.llm.safe_complete(prompt)
            decision = _parse_decision(raw)

            if decision is None:
                # LLM failed or returned garbage → keep test (fail open)
                logger.debug(f"  AI parse failed for {test_file} — keeping (fail open)")
                filtered[test_file] = data
                ai_decisions.append({
                    "test_file":   test_file,
                    "should_run":  True,
                    "confidence":  0.0,
                    "reason":      "AI unavailable — kept by default",
                    "ai_filtered": False,
                })
                continue

            should_run = decision.get("should_run", True)
            confidence = float(decision.get("confidence", 0.0))
            reason     = decision.get("reason", "")
            rel_classes = decision.get("relevant_classes", [])

            # safety rule: low confidence → keep test regardless
            if not should_run and confidence < self.threshold:
                logger.debug(
                    f"  AI says skip {test_file} but confidence {confidence:.2f} "
                    f"< threshold {self.threshold} — keeping"
                )
                should_run = True

            if should_run:
                # if LLM identified specific relevant classes, filter to those
                if rel_classes:
                    filtered_classes = {
                        cls: cls_data
                        for cls, cls_data in data.get("classes", {}).items()
                        if cls in rel_classes
                    }
                    if filtered_classes:
                        data = {**data, "classes": filtered_classes}

                filtered[test_file] = data
                print(f"  ✅ {test_file}")
                print(f"     kept   (confidence: {confidence:.2f})")
                print(f"     \"{reason}\"")
            else:
                print(f"  ❌ {test_file}")
                print(f"     removed (confidence: {confidence:.2f})")
                print(f"     \"{reason}\"")

            ai_decisions.append({
                "test_file":    test_file,
                "should_run":   should_run,
                "confidence":   confidence,
                "reason":       reason,
                "ai_filtered":  not should_run,
            })

        removed = len(selected) - len(filtered)
        if removed > 0:
            print(f"\n  AI removed {removed} test file(s)")

        return filtered, ai_decisions