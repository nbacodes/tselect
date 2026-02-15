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
    print("\n" + "=" * 70)
    print(" Automated Test Impact Analysis (tselect)")
    print("=" * 70)

    icon = "✔" if status == "PASSED" else "⚠"
    print(f"\nStatus            : {icon} {status}")
    print(f"Components        : {', '.join(components) if components else 'None'}")
    print(f"Tests Executed    : {total_tests}")

    print("\nTest Results:")
    print(f"  Passed  : {passed}")
    print(f"  Failed  : {failed}")
    print(f"  Skipped : {skipped}")

    print("\nExecution Metrics:")
    print(f"  Execution Time : {duration:.2f}s")

    if baseline:
        saved = baseline - duration
        percent = (saved / baseline) * 100 if baseline else 0
        print(f"  Baseline Time  : {baseline:.2f}s")
        print(f"  Time Saved     : {saved:.2f}s ({percent:.1f}% faster)")

    print("\nImpact Summary:")
    for c in components:
        print(f"  - {c}")

    print("\nInsights:")
    print("  Confidence Score : N/A (AI layer coming soon)")
    print("  PR Complexity    : N/A (dependency graph pending)")

    print("=" * 70 + "\n")
