def _short_reason(reason: str, max_len: int = 60) -> str:
    """Truncate reason to a short single line."""
    reason = reason.strip().rstrip(".")
    if len(reason) <= max_len:
        return reason
    return reason[:max_len - 1].rsplit(" ", 1)[0] + "…"


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
):
    """
    Pretty CI-style report output for tselect.
    """

    # -------------------------------------------------
    # DERIVED VALUES
    # -------------------------------------------------
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

    executed      = passed + failed + skipped   # real executed count
    ai_decisions  = ai_decisions or []
    tests_before  = executed + sum(            # reconstruct pre-filter count
        d.get("test_count", 0)
        for d in ai_decisions if not d["should_run"]
    )
    ai_percent    = (
        (ai_removed / max(len(ai_decisions), 1)) * 100
        if ai_removed else 0
    )

    n_components = len(components) if components else 0
    complexity   = "Low" if n_components <= 2 else "Medium" if n_components <= 5 else "High"
    comp_str     = ", ".join(sorted(components)) if components else "—"

    W = 70  # width

    # -------------------------------------------------
    # HEADER
    # -------------------------------------------------
    print()
    print("=" * W)
    print("  tselect — Automated Test Impact Analysis")
    print("=" * W)
    print()
    print(f"  Status       : {status_icon}")
    print(f"  Duration     : {duration:.2f}s"
          + (f"  (saved {time_saved:.0f}s — {percent_saved:.1f}% faster)" if baseline else ""))
    print(f"  Components   : {comp_str}")
    print()

    # -------------------------------------------------
    # TEST COUNTS  (before → after)
    # -------------------------------------------------
    print("  " + "─" * (W - 4))
    print("  Tests")
    print("  " + "─" * (W - 4))
    if ai_decisions:
        print(f"  Before AI filter : {tests_before}")
        print(f"  After  AI filter : {executed}  (-{tests_before - executed} tests, {ai_percent:.0f}% of files removed)")
    else:
        print(f"  Executed : {executed}")
    print(f"  Passed   : {passed}   Failed : {failed}   Skipped : {skipped}")
    print()

    # -------------------------------------------------
    # AI PRE-FILTER DECISIONS
    # -------------------------------------------------
    if ai_decisions:
        print("  " + "─" * (W - 4))
        print("  AI Pre-filter")
        print("  " + "─" * (W - 4))
        for d in ai_decisions:
            icon   = "✅" if d["should_run"] else "❌"
            action = "kept   " if d["should_run"] else "removed"
            conf   = d.get("confidence", 0.0)
            reason = _short_reason(d.get("reason", ""))
            fname  = d["test_file"].split("/")[-1]   # just filename, not full path
            print(f"  {icon} {action}  {fname:<45}  ({conf:.2f})")
            print(f"            {reason}")
        print()

    # -------------------------------------------------
    # AI FAILURE ANALYSIS
    # -------------------------------------------------
    if ai_analysis and failed > 0:
        print("  " + "─" * (W - 4))
        print("  AI Failure Analysis")
        print("  " + "─" * (W - 4))
        print(f"  Root cause : {ai_analysis.get('root_cause_file', '—')}  →  {ai_analysis.get('root_cause_symbol', '—')}")
        print(f"  Pattern    : {ai_analysis.get('failure_pattern', '—')}")
        print(f"  Explanation: {ai_analysis.get('explanation', '—')}")
        print(f"  Fix        : {ai_analysis.get('suggested_fix', '—')}")
        print()

    # -------------------------------------------------
    # INSIGHTS
    # -------------------------------------------------
    print("  " + "─" * (W - 4))
    print("  Insights")
    print("  " + "─" * (W - 4))
    conf_str = f"{ai_confidence:.2f}" if ai_confidence is not None else "N/A"
    print(f"  Confidence Score : {conf_str}")
    print(f"  PR Complexity    : {complexity}  ({n_components} components changed)")
    if baseline:
        print(f"  Baseline         : {baseline:.2f}s  →  {duration:.2f}s  ({percent_saved:.1f}% faster)")
    print()
    print("=" * W)
    print()