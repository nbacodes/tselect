def generate_summary(
    components,
    total_tests,
    duration,
    status,
    baseline,
    passed,
    failed,
    skipped,
):
    """
    Pretty CI-style report output for tselect.
    """

    # -------------------------------------------------
    # STATUS LOGIC
    # -------------------------------------------------
    if failed > 0 and passed > 0:
        status_icon = "⚠ PARTIAL_FAIL"
    elif failed > 0:
        status_icon = "✖ FAILED"
    else:
        status_icon = "✔ PASSED"

    # -------------------------------------------------
    # TIME SAVINGS
    # -------------------------------------------------
    time_saved = 0
    percent_saved = 0

    if baseline:
        time_saved = max(0, baseline - duration)
        percent_saved = (time_saved / baseline) * 100 if baseline else 0

    comp_str = ", ".join(sorted(components)) if components else "None"

    # -------------------------------------------------
    # HEADER
    # -------------------------------------------------
    print("\n" + "=" * 70)
    print(" Automated Test Impact Analysis (tselect)")
    print("=" * 70)

    print(f"\nStatus            : {status_icon}")
    print(f"Components        : {comp_str}")
    print(f"Tests Executed    : {total_tests}")

    # -------------------------------------------------
    # TEST RESULTS SECTION
    # -------------------------------------------------
    print("\n" + "-" * 70)
    print(" Test Results")
    print("-" * 70)

    print(f"  Passed  : {passed}")
    print(f"  Failed  : {failed}")
    print(f"  Skipped : {skipped}")

    # -------------------------------------------------
    # EXECUTION METRICS
    # -------------------------------------------------
    print("\n" + "-" * 70)
    print(" Execution Metrics")
    print("-" * 70)

    print(f"  Execution Time : {duration:.2f}s")
    print(f"  Baseline Time  : {baseline:.2f}s" if baseline else "  Baseline Time  : N/A")
    print(f"  Time Saved     : {time_saved:.2f}s ({percent_saved:.1f}% faster)")

    # -------------------------------------------------
    # IMPACT SUMMARY
    # -------------------------------------------------
    print("\n" + "-" * 70)
    print(" Impact Summary")
    print("-" * 70)

    for c in components:
        print(f"  - {c}")

    # -------------------------------------------------
    # INSIGHTS
    # -------------------------------------------------
    print("\n" + "-" * 70)
    print(" Insights")
    print("-" * 70)

    print("  Confidence Score : N/A (AI layer coming soon)")
    print("  PR Complexity    : N/A (dependency graph pending)")

    print("\n" + "=" * 70 + "\n")
