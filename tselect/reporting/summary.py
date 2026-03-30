def _short_reason(reason: str, max_len: int = 60) -> str:
    reason = reason.strip().rstrip(".")
    if len(reason) <= max_len:
        return reason
    return reason[:max_len - 1].rsplit(" ", 1)[0] + "…"


def _build_audit(ai_decisions: list) -> dict:
    correct_keeps   = []
    correct_removes = []
    safety_keeps    = []
    contradictions  = []

    negative_phrases = [
        "does not directly", "unrelated", "not directly test",
        "no direct", "indirect", "not related"
    ]

    for d in ai_decisions:
        conf       = d.get("confidence", 0.0)
        should_run = d.get("should_run", True)
        reason     = d.get("reason", "").lower()
        reason_sounds_negative = any(p in reason for p in negative_phrases)

        if should_run:
            if conf >= 0.75:
                if reason_sounds_negative:
                    contradictions.append(d)
                else:
                    correct_keeps.append(d)
            else:
                safety_keeps.append(d)
        else:
            correct_removes.append(d)

    return {
        "correct_keeps":   correct_keeps,
        "correct_removes": correct_removes,
        "safety_keeps":    safety_keeps,
        "contradictions":  contradictions,
    }


def _print_audit(ai_decisions: list, W: int = 70) -> None:
    """Print the Selection Audit block. Shared by both report functions."""
    audit = _build_audit(ai_decisions)

    print("  " + "─" * (W - 4))
    print("  Selection Audit")
    print("  " + "─" * (W - 4))

    if audit["correct_keeps"]:
        print(f"  ✅ HIGH CONFIDENCE SELECTIONS  ({len(audit['correct_keeps'])} files)")
        for d in audit["correct_keeps"]:
            fname  = d["test_file"].split("/")[-1]
            reason = _short_reason(d.get("reason", ""), 55)
            conf   = d.get("confidence", 0.0)
            print(f"     {fname:<40}  ({conf:.2f})  {reason}")
        print()

    if audit["correct_removes"]:
        print(f"  ❌ HIGH CONFIDENCE REMOVALS  ({len(audit['correct_removes'])} files)")
        for d in audit["correct_removes"]:
            fname  = d["test_file"].split("/")[-1]
            reason = _short_reason(d.get("reason", ""), 55)
            conf   = d.get("confidence", 0.0)
            print(f"     {fname:<40}  ({conf:.2f})  {reason}")
        print()

    if audit["safety_keeps"]:
        print(f"  ⚠️  KEPT BY SAFETY RULE  ({len(audit['safety_keeps'])} files)")
        print(f"     (LLM was uncertain → kept to avoid missing bugs)")
        for d in audit["safety_keeps"]:
            fname  = d["test_file"].split("/")[-1]
            conf   = d.get("confidence", 0.0)
            print(f"     {fname:<40}  ({conf:.2f})  reason unclear")
        print()

    if audit["contradictions"]:
        print(f"  ⚠️  POTENTIAL FALSE POSITIVES  ({len(audit['contradictions'])} files)")
        print(f"     (kept at high confidence but reason suggests unrelated)")
        for d in audit["contradictions"]:
            fname  = d["test_file"].split("/")[-1]
            reason = _short_reason(d.get("reason", ""), 55)
            conf   = d.get("confidence", 0.0)
            print(f"     {fname:<40}  ({conf:.2f})  {reason}")
        print()

    total     = len(ai_decisions)
    confident = len(audit["correct_keeps"]) + len(audit["correct_removes"])
    uncertain = len(audit["safety_keeps"]) + len(audit["contradictions"])
    score     = confident / max(total, 1)
    print(f"  Decision quality : {confident}/{total} high-confidence  ({score:.0%})")
    if uncertain > 0:
        print(f"  Uncertain        : {uncertain} files kept conservatively")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY 1 — AI SELECTION REPORT
# ─────────────────────────────────────────────────────────────────────────────

def generate_ai_summary(selected: dict, ai_decisions: list) -> None:
    if not ai_decisions:
        return

    W = 70
    kept    = [d for d in ai_decisions if d["should_run"]]
    removed = [d for d in ai_decisions if not d["should_run"]]

    print()
    print("=" * W)
    print("  tselect — AI Selection Report")
    print("=" * W)
    print()
    print(f"  Candidates evaluated : {len(ai_decisions)}")
    print(f"  Selected             : {len(kept)}")
    print(f"  Removed by AI        : {len(removed)}")
    print()

    print("  " + "─" * (W - 4))
    print("  Per-File Decisions")
    print("  " + "─" * (W - 4))
    for d in ai_decisions:
        icon   = "✅" if d["should_run"] else "❌"
        action = "kept   " if d["should_run"] else "removed"
        conf   = d.get("confidence", 0.0)
        reason = _short_reason(d.get("reason", ""))
        fname  = d["test_file"].split("/")[-1]
        print(f"  {icon} {action}  {fname:<45}  ({conf:.2f})")
        print(f"            {reason}")
    print()

    _print_audit(ai_decisions, W)

    print("=" * W)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY 2 — EXECUTION REPORT
# ─────────────────────────────────────────────────────────────────────────────

def generate_summary(
    components,
    total_tests,
    duration,
    status,
    baseline,
    passed,
    failed,
    skipped,
    ai_decisions  = None,
    ai_removed    = 0,
    ai_confidence = None,
    ai_analysis   = None,
    coverage_data = None,
) -> None:
    """
    Printed after pytest finishes.
    All existing sections unchanged.
    Coverage section added at the bottom — only printed when --coverage used.
    """

    if failed > 0 and passed > 0:
        status_icon = "⚠  PARTIAL FAIL"
    elif failed > 0:
        status_icon = "✖  FAILED"
    else:
        status_icon = "✔  PASSED"

    time_saved    = 0
    percent_saved = 0
    if baseline:
        time_saved    = max(0, baseline - duration)
        percent_saved = (time_saved / baseline) * 100

    executed     = passed + failed + skipped
    ai_decisions = ai_decisions or []
    ai_percent   = (
        (ai_removed / max(len(ai_decisions), 1)) * 100
        if ai_removed else 0
    )

    n_components = len(components) if components else 0
    complexity   = "Low" if n_components <= 2 else "Medium" if n_components <= 5 else "High"
    comp_str     = ", ".join(sorted(components)) if components else "—"

    W = 70

    # ── header ──────────────────────────────────────────────────────────────
    print()
    print("=" * W)
    print("  tselect — Automated Test Impact Analysis")
    print("=" * W)
    print()
    print(f"  Status       : {status_icon}")
    print(f"  Duration     : {duration:.2f}s")
    print(f"  Components   : {comp_str}")
    print()

    # ── test counts ─────────────────────────────────────────────────────────
    print("  " + "─" * (W - 4))
    print("  Tests")
    print("  " + "─" * (W - 4))
    if ai_decisions:
        print(f"  Before AI filter : {len(ai_decisions)} candidate files")
        print(f"  After  AI filter : {len(ai_decisions) - ai_removed} files  "
              f"({ai_removed} removed, {ai_percent:.0f}% filtered out)")
    print(f"  Executed : {executed}")
    print(f"  Passed   : {passed}   Failed : {failed}   Skipped : {skipped}")
    print()

    # ── selection audit ─────────────────────────────────────────────────────
    if ai_decisions:
        _print_audit(ai_decisions, W)

    # ── AI failure analysis ─────────────────────────────────────────────────
    if ai_analysis and failed > 0:
        print("  " + "─" * (W - 4))
        print("  AI Failure Analysis")
        print("  " + "─" * (W - 4))
        print(f"  Root cause : {ai_analysis.get('root_cause_file', '—')}"
              f"  →  {ai_analysis.get('root_cause_symbol', '—')}")
        print(f"  Pattern    : {ai_analysis.get('failure_pattern', '—')}")
        print(f"  Explanation: {ai_analysis.get('explanation', '—')}")
        print(f"  Fix        : {ai_analysis.get('suggested_fix', '—')}")
        print()

    # ── insights ────────────────────────────────────────────────────────────
    print("  " + "─" * (W - 4))
    print("  Insights")
    print("  " + "─" * (W - 4))
    conf_str = f"{ai_confidence:.2f}" if ai_confidence is not None else "N/A"
    print(f"  Confidence Score : {conf_str}")
    print(f"  PR Complexity    : {complexity}  ({n_components} components changed)")
    if baseline:
        print(f"  Baseline         : {baseline:.2f}s  →  {duration:.2f}s"
              f"  ({percent_saved:.1f}% faster)")
    print()

    # ── coverage section — only shown when --coverage flag used ─────────────
    if coverage_data:
        from tselect.reporting.coverage import format_coverage_section
        for line in format_coverage_section(coverage_data, W):
            print(line)

    print("=" * W)
    print()