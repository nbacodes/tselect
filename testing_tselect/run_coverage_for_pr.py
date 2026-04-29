#!/usr/bin/env bash
# ============================================================
# run_coverage_for_pr.sh
# Runs tselect-selected tests for a single PR with coverage.
#
# Usage (open a separate terminal for each PR):
#   ./run_coverage_for_pr.sh 121953
#   ./run_coverage_for_pr.sh 181666
#   ./run_coverage_for_pr.sh 145690
#   ./run_coverage_for_pr.sh 152993
# ============================================================

set -euo pipefail

PR="${1:-}"
if [[ -z "$PR" ]]; then
    echo "Usage: $0 <PR_NUMBER>"
    echo "Example: $0 121953"
    exit 1
fi

# ── Config ────────────────────────────────────────────────────
PYTORCH_DIR="/Users/nihalkumar/pytorch-pr-176888"
TSELECT_TEST_DIR="/Users/nihalkumar/Desktop/nbaworks/tselect/tselect_test"
VENV="/Users/nihalkumar/envs/pytorch-dev"
TIMEOUT=600  # seconds per test file

SELECTED_FILE="$TSELECT_TEST_DIR/$PR/selected_tests.txt"
OUT_DIR="$TSELECT_TEST_DIR/$PR"
COV_XML="$OUT_DIR/coverage_selected.xml"
LOG_FILE="$OUT_DIR/coverage_run.log"

# ── Validate ──────────────────────────────────────────────────
if [[ ! -f "$SELECTED_FILE" ]]; then
    echo "ERROR: No selected_tests.txt for PR #$PR"
    echo "  Expected: $SELECTED_FILE"
    exit 1
fi

source "$VENV/bin/activate"

n_tests=$(grep -c '.' "$SELECTED_FILE" 2>/dev/null || echo 0)
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  PR #$PR — Coverage Run"
echo "║  Tests to run : $n_tests"
echo "║  Output       : $COV_XML"
echo "╚══════════════════════════════════════════════╝"
echo ""

cd "$PYTORCH_DIR"

# ── Run each selected test with coverage ──────────────────────
mkdir -p "$OUT_DIR/cov_parts"
FAILED=()
PASSED=()
SKIPPED=()

while IFS= read -r test_path; do
    [[ -z "$test_path" ]] && continue

    # Normalize path
    test_path=$(echo "$test_path" | xargs)
    test_path="${test_path#test/}"
    full_path="$PYTORCH_DIR/test/$test_path"

    if [[ ! -f "$full_path" ]]; then
        echo "  MISSING: $test_path"
        SKIPPED+=("$test_path")
        continue
    fi

    safe_name=$(echo "$test_path" | tr '/' '_' | sed 's/.py$//')
    cov_part="$OUT_DIR/cov_parts/.coverage.$safe_name"
    test_log="$OUT_DIR/cov_parts/${safe_name}.log"

    echo "  ► $test_path"

    if timeout "$TIMEOUT" python3 -m pytest \
        --no-header -q \
        --tb=no \
        --cov=torch._inductor \
        --cov-report="" \
        --cov-data-file="$cov_part" \
        "$full_path" \
        > "$test_log" 2>&1; then
        echo "    ✓ passed"
        PASSED+=("$test_path")
    else
        exit_code=$?
        if [[ $exit_code -eq 124 ]]; then
            echo "    ⏱ timeout"
        else
            echo "    ✗ failed (exit $exit_code)"
        fi
        FAILED+=("$test_path")
    fi

done < "$SELECTED_FILE"

# ── Combine coverage files ────────────────────────────────────
echo ""
echo "Combining coverage files..."

cov_files=("$OUT_DIR/cov_parts"/.coverage.*)
if [[ ${#cov_files[@]} -eq 0 ]] || [[ ! -f "${cov_files[0]}" ]]; then
    echo "  No coverage data collected."
    exit 1
fi

# Write .coveragerc
cat > "$OUT_DIR/.coveragerc" <<EOF
[run]
data_file = $OUT_DIR/.coverage.combined
source = torch/_inductor

[xml]
output = $COV_XML
EOF

python3 -m coverage combine \
    --rcfile="$OUT_DIR/.coveragerc" \
    "${cov_files[@]}"

python3 -m coverage xml \
    --rcfile="$OUT_DIR/.coveragerc" \
    -o "$COV_XML"

echo "  ✓ Coverage XML saved: $COV_XML"

# ── Summary ───────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  PR #$PR — Done"
printf "║  Passed  : %d\n" "${#PASSED[@]}"
printf "║  Failed  : %d\n" "${#FAILED[@]}"
printf "║  Skipped : %d\n" "${#SKIPPED[@]}"
echo "║  Coverage: $COV_XML"
echo "╚══════════════════════════════════════════════╝"
