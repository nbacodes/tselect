"""
tselect/ai/pre_filter.py
------------------------
LLM-based pre-filter that runs AFTER rule-based graph selection.

Takes the graph selector's output (candidate test files + their triggered_by
info) and asks Groq/Llama: "does this test actually need to run?"

Design principles:
  - Never under-selects: if confidence < threshold, keep the test
  - If LLM fails entirely, keep ALL candidates (fail open)
  - Sends 3 signals per candidate:
      1. git diff of changed files (truncated) — what actually changed
      2. test method names (first 10 per class) — what the test actually tests
      3. file names + class names — structural context
    Together these give the LLM grounded evidence, not just names.

Input:  selected dict from graph_selector.select_tests_from_graph()
Output: filtered selected dict (same structure, subset of input)
"""

import ast
import json
import re
import subprocess
from pathlib import Path
from tselect.ai.llm_client import LLMClient, LLMClientError
from tselect.utils.logger import setup_logger

logger = setup_logger()

# ── token budget constants ──────────────────────────────────────────────────
MAX_DIFF_CHARS    = 1500   # ~375 tokens — enough to see what changed
MAX_METHODS_CLASS = 10     # first 10 method names per class

SYSTEM_PROMPT = """You are a precise test selection assistant for a Python codebase.
Your job is to determine if a test file DIRECTLY tests the changed functionality.

RULES:
- Only return should_run: true if test classes DIRECTLY test the changed symbols
- If the connection is indirect, coincidental, or just a shared import, return should_run: false
- Use the git diff to understand WHAT changed, use method names to understand WHAT the test tests
- Be precise, not conservative — false positives waste CI time
Respond ONLY in valid JSON. No text outside the JSON object."""

CANDIDATE_PROMPT_TEMPLATE = """
CHANGED SOURCE FILES AND SYMBOLS:
{changed_summary}

CANDIDATE TEST FILE: {test_file}
TRIGGERED BY: {triggered_by}
TEST CLASSES AND METHODS:
{class_summary}

QUESTION:
Do any of these test classes DIRECTLY test the changed symbols listed above?
Or do they test something else that is unrelated to those specific changes?

Respond ONLY in this exact JSON format:
{{
  "should_run": true or false,
  "confidence": true probability — use full range (0.95=almost certain, 0.7-0.9=reasonably confident, 0.5-0.7=uncertain, below 0.5=unlikely but not sure),
  "reason": "one sentence max",
  "relevant_classes": ["ClassName"] or []
}}"""


# ── diff extraction ─────────────────────────────────────────────────────────

def _get_diff_summary(changed_files: list, repo_root: Path) -> dict:
    """
    Run git diff for each changed file and return truncated diffs.
    Returns {file_path: diff_string}
    """
    diffs = {}
    for f in changed_files:
        try:
            result = subprocess.run(
                ["git", "diff", "HEAD", "--", f],
                cwd            = repo_root,
                capture_output = True,
                text           = True,
                timeout        = 5,
            )
            diff = result.stdout.strip()
            if not diff:
                # try staged diff
                result = subprocess.run(
                    ["git", "diff", "--cached", "--", f],
                    cwd            = repo_root,
                    capture_output = True,
                    text           = True,
                    timeout        = 5,
                )
                diff = result.stdout.strip()

            if diff:
                lines      = diff.splitlines()
                kept       = []
                char_count = 0
                for line in lines:
                    if char_count + len(line) > MAX_DIFF_CHARS:
                        kept.append("  ... (truncated)")
                        break
                    kept.append(line)
                    char_count += len(line)
                diffs[f] = "\n".join(kept)
        except Exception as e:
            logger.debug(f"git diff failed for {f}: {e}")
    return diffs


# ── test method extraction ──────────────────────────────────────────────────

# Device suffixes added by instantiate_device_type_tests()
# e.g. TestOptimRenewedCPU → TestOptimRenewed
_DEVICE_SUFFIXES = ["CPU", "MPS", "CUDA", "XLA", "GPU", "HPU"]

def _strip_device_suffix(cls_name: str) -> str:
    """
    Strip device suffix from dynamically generated test class names.
    TestOptimRenewedCPU → TestOptimRenewed
    TestAutograd        → TestAutograd  (unchanged, no suffix)
    """
    for suffix in _DEVICE_SUFFIXES:
        if cls_name.endswith(suffix):
            return cls_name[:-len(suffix)]
    return cls_name


def _get_test_methods(test_file: str, repo_root: Path, classes: dict) -> dict:
    """
    Parse test file and extract first MAX_METHODS_CLASS method names per class.
    Returns {class_name: [method_names]}

    Handles PyTorch's dynamic test generation pattern:
      instantiate_device_type_tests(TestOptimRenewed, globals())
      → generates TestOptimRenewedCPU, TestOptimRenewedMPS at runtime

    Since TestOptimRenewedCPU doesn't exist in source, we look for
    the base class TestOptimRenewed and map its methods back.
    """
    methods = {}
    try:
        src  = (repo_root / test_file).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(src)
    except Exception:
        return methods

    # build reverse map: base_name → [original_cls_names]
    # e.g. "TestOptimRenewed" → ["TestOptimRenewedCPU", "TestOptimRenewedMPS"]
    base_to_original: dict[str, list[str]] = {}
    for cls_name in classes:
        base = _strip_device_suffix(cls_name)
        base_to_original.setdefault(base, []).append(cls_name)

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        # match either exact name or base name
        matched_originals = []
        if node.name in classes:
            matched_originals.append(node.name)          # exact match
        if node.name in base_to_original:
            matched_originals.extend(base_to_original[node.name])  # base match

        if not matched_originals:
            continue

        test_methods = [
            child.name
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            and child.name.startswith("test")
        ][:MAX_METHODS_CLASS]

        if test_methods:
            # assign same methods to all matched original class names
            for orig in matched_originals:
                methods[orig] = test_methods

    return methods


# ── prompt builders ─────────────────────────────────────────────────────────

def _build_changed_summary(
    changed_files: list,
    changed_symbols: dict,
    diffs: dict,
) -> str:
    lines = []
    for f in changed_files:
        syms  = changed_symbols.get(f, set())
        desc  = _module_description(f)
        clean = {s for s in syms if s != "__module__"}

        header = f"  - {f}  ({desc})"
        if clean:
            header += f"\n    changed symbols: {', '.join(sorted(clean)[:8])}"
        lines.append(header)

        diff = diffs.get(f)
        if diff:
            lines.append(f"    git diff:\n{_indent(diff, '      ')}")

    return "\n".join(lines)


def _build_class_summary(classes: dict, methods: dict) -> str:
    lines = []
    for cls_name, cls_data in classes.items():
        count       = cls_data.get("test_count", 0)
        cls_methods = methods.get(cls_name, [])

        if cls_methods:
            method_str = ", ".join(cls_methods)
            if count > MAX_METHODS_CLASS:
                method_str += f", ... ({count - MAX_METHODS_CLASS} more)"
            lines.append(f"  - {cls_name} ({count} tests)\n    methods: {method_str}")
        else:
            lines.append(f"  - {cls_name} ({count} tests)")

    return "\n".join(lines)


def _module_description(file_path: str) -> str:
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
        "masked":        "masked tensor operations",
        "distributions": "probability distributions",
        "inductor":      "torch.compile / inductor backend",
        "dynamo":        "TorchDynamo graph capture",
        "export":        "torch.export / graph serialization",
        "profiler":      "performance profiling utilities",
    }
    for key, desc in KNOWN_MODULES.items():
        if key in file_path:
            return desc
    name = file_path.split("/")[-1].replace(".py", "").replace("_", " ")
    return f"{name} module"


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())


# ── response parser ─────────────────────────────────────────────────────────

def _parse_decision(raw: str) -> dict | None:
    if raw is None:
        return None
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    try:
        data = json.loads(raw)
        if "should_run" not in data or "confidence" not in data:
            return None
        return data
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return None


# ── main class ───────────────────────────────────────────────────────────────

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
        repo_root: Path | None = None,
    ) -> tuple[dict, list]:
        """
        Filter candidate test files using LLM.

        Args:
            selected:        output of select_tests_from_graph()
            changed_files:   list of changed source file paths
            changed_symbols: {file: set(symbols)} from diff_parser
            repo_root:       Path to repo root (for git diff + test parsing)

        Returns:
            (filtered_selected, ai_decisions)
        """
        if not selected:
            return selected, []

        # ── enrich context ──
        diffs = {}
        if repo_root:
            diffs = _get_diff_summary(changed_files, repo_root)
            if diffs:
                logger.debug(f"  Got diffs for {len(diffs)}/{len(changed_files)} files")

        changed_summary = _build_changed_summary(changed_files, changed_symbols, diffs)
        filtered        = {}
        ai_decisions    = []

        print(f"\n  🤖 AI pre-filtering {len(selected)} candidate test files...")

        for test_file, data in selected.items():
            triggered_by  = data.get("triggered_by", [])
            triggered_str = ", ".join(triggered_by) if isinstance(triggered_by, list) else str(triggered_by)

            # extract test method names for this candidate
            methods = {}
            if repo_root:
                methods = _get_test_methods(test_file, repo_root, data.get("classes", {}))

            class_summary = _build_class_summary(data.get("classes", {}), methods)

            prompt = SYSTEM_PROMPT + CANDIDATE_PROMPT_TEMPLATE.format(
                changed_summary = changed_summary,
                test_file       = test_file,
                triggered_by    = triggered_str,
                class_summary   = class_summary,
            )

            raw      = self.llm.safe_complete(prompt)
            decision = _parse_decision(raw)

            if decision is None:
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

            should_run  = decision.get("should_run", True)
            confidence  = float(decision.get("confidence", 0.0))
            reason      = decision.get("reason", "")
            rel_classes = decision.get("relevant_classes", [])

            # safety rule: low confidence → keep test regardless
            if not should_run and confidence < self.threshold:
                logger.debug(
                    f"  AI says skip {test_file} but confidence {confidence:.2f} "
                    f"< threshold {self.threshold} — keeping"
                )
                should_run = True

            if should_run:
                # filter to only relevant classes if LLM identified them
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