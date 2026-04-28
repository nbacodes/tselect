#!/usr/bin/env bash
# ============================================================
# collect_ground_truth.sh
# For each PR in the slate, runs ALL relevant test files
# with coverage to produce the ground-truth coverage map.
#
# Usage:
#   ./collect_ground_truth.sh --all
#   ./collect_ground_truth.sh --pr 152993
#   ./collect_ground_truth.sh --module inductor   # run all inductor tests
#
# This is the expensive step. Ground truth only needs to be
# collected once per module group, not once per PR — PRs that
# touch the same module share the same ground truth.
# ============================================================

set -euo pipefail

PYTORCH_DIR="${PYTORCH_DIR:-$HOME/pytorch}"
RESULTS_DIR="${RESULTS_DIR:-$(pwd)/multi_pr_results}"
VENV="${VENV:-$HOME/.virtualenvs/pytorch-dev}"
COVERAGE_DIR="$RESULTS_DIR/ground_truth"
TIMEOUT="${TIMEOUT:-600}"            # seconds per test file
MAX_WORKERS="${MAX_WORKERS:-1}"      # parallel workers (keep 1 on Mac)

# ── Test suites by module ─────────────────────────────────────
# Each entry: "MODULE_KEY|TEST_FILES(space-sep)"
# Tests are relative to $PYTORCH_DIR/test/
declare -A TEST_SUITES

TEST_SUITES["inductor"]="
    inductor/test_torchinductor.py
    inductor/test_inductor_utils.py
    inductor/test_codegen_triton.py
    inductor/test_scheduler.py
    inductor/test_codecache.py
    inductor/test_mkldnn_pattern_matcher.py
    inductor/test_triton_kernels.py
"

TEST_SUITES["inductor_codegen"]="
    inductor/test_codegen_triton.py
    inductor/test_triton_kernels.py
"

TEST_SUITES["inductor_fx_passes"]="
    inductor/test_mkldnn_pattern_matcher.py
    inductor/test_pattern_matcher.py
"

# ── PR → module mapping ───────────────────────────────────────
# Which test suite is the right ground truth for each PR
declare -A PR_MODULE_MAP=(
    ["145690"]="inductor_codegen"
    ["145691"]="inductor_codegen"
    ["145692"]="inductor_codegen"
    ["152993"]="inductor"
    ["121953"]="inductor_fx_passes"
    ["181666"]="inductor"
    ["181700"]="inductor_codegen"
)

# ── Arg parsing ───────────────────────────────────────────────
RUN_ALL=false
SINGLE_PR=""
SINGLE_MODULE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)     RUN_ALL=true; shift ;;
        --pr)      SINGLE_PR="$2"; shift 2 ;;
        --module)  SINGLE_MODULE="$2"; shift 2 ;;
        --pytorch-dir) PYTORCH_DIR="$2"; shift 2 ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

log() { echo "[$(date '+%H:%M:%S')] $*"; }

activate_venv() {
    if [[ -f "$VENV/bin/activate" ]]; then
        source "$VENV/bin/activate"
    fi
}

run_test_with_coverage() {
    local test_file="$1"
    local cov_dir="$2"
    local full_path="$PYTORCH_DIR/test/$test_file"

    if [[ ! -f "$full_path" ]]; then
        log "  SKIP (not found): $test_file"
        return 0
    fi

    local safe_name
    safe_name=$(echo "$test_file" | tr '/' '_' | sed 's/.py$//')
    local cov_file="$cov_dir/.coverage.$safe_name"
    local log_file="$cov_dir/logs/${safe_name}.log"
    mkdir -p "$cov_dir/logs"

    log "  Running: $test_file"

    if timeout "$TIMEOUT" python3 -m pytest \
        --no-header -q \
        --ignore-glob="**/test_torchinductor_dynamic_shapes*" \
        --cov="torch._inductor" \
        --cov-report="" \
        --cov-data-file="$cov_file" \
        "$full_path" \
        > "$log_file" 2>&1; then
        log "    ✓ $safe_name"
    else
        local exit_code=$?
        if [[ $exit_code -eq 124 ]]; then
            log "    ⏱ TIMEOUT: $safe_name"
        else
            log "    ✗ FAILED (exit $exit_code): $safe_name — see $log_file"
        fi
    fi
}

run_ground_truth_for_module() {
    local module_key="$1"
    local cov_dir="$COVERAGE_DIR/$module_key"

    if [[ -z "${TEST_SUITES[$module_key]+_}" ]]; then
        log "ERROR: Unknown module '$module_key'"
        return 1
    fi

    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "Ground truth: module=$module_key"
    log "Coverage dir: $cov_dir"

    # Skip if already collected (remove to regenerate)
    if [[ -f "$cov_dir/coverage_combined.json" ]]; then
        log "  Already collected — skipping. Delete $cov_dir to redo."
        return 0
    fi

    mkdir -p "$cov_dir"
    activate_venv

    local tests
    tests=$(echo "${TEST_SUITES[$module_key]}" | xargs)

    for test_file in $tests; do
        test_file=$(echo "$test_file" | xargs)
        [[ -z "$test_file" ]] && continue
        run_test_with_coverage "$test_file" "$cov_dir"
    done

    # Combine coverage files
    log "  Combining coverage files..."
    python3 - "$cov_dir" <<'PYEOF'
import sys, os, json, subprocess

cov_dir = sys.argv[1]
cov_files = [f for f in os.listdir(cov_dir) if f.startswith('.coverage.')]
if not cov_files:
    print("  No coverage files found")
    sys.exit(0)

# Use coverage CLI to combine and export JSON
rc_file = os.path.join(cov_dir, '.coveragerc')
with open(rc_file, 'w') as f:
    f.write(f"[run]\ndata_file = {os.path.join(cov_dir, '.coverage.combined')}\n")

# Combine
subprocess.run(
    ['python3', '-m', 'coverage', 'combine',
     '--rcfile', rc_file] + [os.path.join(cov_dir, f) for f in cov_files],
    capture_output=True
)

# Export to JSON
result = subprocess.run(
    ['python3', '-m', 'coverage', 'json',
     '--rcfile', rc_file,
     '-o', os.path.join(cov_dir, 'coverage_combined.json'),
     '--pretty-print'],
    capture_output=True, text=True
)
if result.returncode == 0:
    print(f"  Combined {len(cov_files)} coverage file(s)")
else:
    print(f"  Coverage combine error: {result.stderr[:200]}")
PYEOF

    log "  Ground truth collected: $cov_dir/coverage_combined.json"
}

# ── Main ──────────────────────────────────────────────────────
mkdir -p "$COVERAGE_DIR"

log "Ground Truth Collector"
log "PyTorch dir : $PYTORCH_DIR"
log "Coverage dir: $COVERAGE_DIR"
log ""

if [[ -n "$SINGLE_MODULE" ]]; then
    run_ground_truth_for_module "$SINGLE_MODULE"
elif [[ -n "$SINGLE_PR" ]]; then
    module="${PR_MODULE_MAP[$SINGLE_PR]:-inductor}"
    log "PR #$SINGLE_PR → module: $module"
    run_ground_truth_for_module "$module"
elif [[ "$RUN_ALL" == "true" ]]; then
    # Collect unique modules across all PRs
    declare -A done_modules
    for pr in "${!PR_MODULE_MAP[@]}"; do
        mod="${PR_MODULE_MAP[$pr]}"
        if [[ -z "${done_modules[$mod]+_}" ]]; then
            done_modules[$mod]=1
            run_ground_truth_for_module "$mod"
        fi
    done
else
    echo "Usage: $0 --all | --pr <PR> | --module <key>"
    exit 1
fi

log ""
log "Done. Ground truth at: $COVERAGE_DIR"
