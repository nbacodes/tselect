#!/usr/bin/env bash
# ============================================================
# run_selected_tests.sh
# For each PR, run ONLY the tests that tselect selected,
# collecting coverage. This is Step 3 of the 3-phase pipeline.
#
# Prerequisite: run_tselect_multi_pr.sh must have run first
#               so selected_tests.txt exists for each PR.
#
# Usage:
#   ./run_selected_tests.sh --all
#   ./run_selected_tests.sh --pr 152993
# ============================================================

set -euo pipefail

PYTORCH_DIR="${PYTORCH_DIR:-$HOME/pytorch}"
RESULTS_DIR="${RESULTS_DIR:-$(pwd)/multi_pr_results}"
VENV="${VENV:-$HOME/.virtualenvs/pytorch-dev}"
TIMEOUT="${TIMEOUT:-600}"

declare -a ALL_PRS=(145690 145691 145692 152993 121953 181666 181700)

log() { echo "[$(date '+%H:%M:%S')] $*"; }

activate_venv() {
    [[ -f "$VENV/bin/activate" ]] && source "$VENV/bin/activate"
}

RUN_ALL=false
SINGLE_PR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all) RUN_ALL=true; shift ;;
        --pr)  SINGLE_PR="$2"; shift 2 ;;
        --pytorch-dir) PYTORCH_DIR="$2"; shift 2 ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

run_selected_for_pr() {
    local pr="$1"
    local pr_dir="$RESULTS_DIR/$pr"
    local selected_file="$pr_dir/selected_tests.txt"
    local cov_dir="$pr_dir/selected_coverage"

    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "PR #$pr — running selected tests"

    if [[ ! -f "$selected_file" ]]; then
        log "  SKIP: no selected_tests.txt (run run_tselect_multi_pr.sh first)"
        return 1
    fi

    local n_selected
    n_selected=$(grep -c '.' "$selected_file" 2>/dev/null || echo 0)
    log "  Selected test count: $n_selected"

    if [[ "$n_selected" -eq 0 ]]; then
        log "  No tests selected — recording empty coverage"
        mkdir -p "$cov_dir"
        echo '{"files": {}}' > "$cov_dir/coverage.json"
        return 0
    fi

    mkdir -p "$cov_dir/logs"
    activate_venv

    while IFS= read -r test_path; do
        [[ -z "$test_path" ]] && continue

        # Normalize: strip leading test/ if present, strip trailing whitespace
        test_path=$(echo "$test_path" | sed 's|^test/||' | xargs)
        local full_path="$PYTORCH_DIR/test/$test_path"

        if [[ ! -f "$full_path" ]]; then
            log "  MISSING: $full_path"
            continue
        fi

        local safe_name
        safe_name=$(echo "$test_path" | tr '/' '_' | sed 's/.py$//')
        local cov_file="$cov_dir/.coverage.$safe_name"
        local log_file="$cov_dir/logs/${safe_name}.log"

        log "  Running: $test_path"
        if timeout "$TIMEOUT" python3 -m pytest \
            --no-header -q \
            --cov="torch._inductor" \
            --cov-report="" \
            --cov-data-file="$cov_file" \
            "$full_path" \
            > "$log_file" 2>&1; then
            log "    ✓ $safe_name"
        else
            local exit_code=$?
            [[ $exit_code -eq 124 ]] && log "    ⏱ TIMEOUT: $safe_name" || \
                                        log "    ✗ FAILED: $safe_name"
        fi

    done < "$selected_file"

    # Combine coverage
    python3 - "$cov_dir" <<'PYEOF'
import sys, os, subprocess

cov_dir = sys.argv[1]
cov_files = [f for f in os.listdir(cov_dir) if f.startswith('.coverage.')]
if not cov_files:
    import json
    with open(os.path.join(cov_dir, 'coverage.json'), 'w') as f:
        json.dump({"files": {}}, f)
    print("  No coverage files — writing empty JSON")
    sys.exit(0)

rc = os.path.join(cov_dir, '.coveragerc')
with open(rc, 'w') as f:
    f.write(f"[run]\ndata_file = {os.path.join(cov_dir, '.coverage')}\n")

subprocess.run(['python3', '-m', 'coverage', 'combine',
                '--rcfile', rc] + [os.path.join(cov_dir, f) for f in cov_files],
               capture_output=True)

subprocess.run(['python3', '-m', 'coverage', 'json',
                '--rcfile', rc,
                '-o', os.path.join(cov_dir, 'coverage.json'),
                '--pretty-print'],
               capture_output=True)

print(f"  Combined {len(cov_files)} coverage file(s) → coverage.json")
PYEOF

    log "  Coverage saved: $cov_dir/coverage.json"
}

# ── Main ──────────────────────────────────────────────────────
for pr in "${ALL_PRS[@]}"; do
    if [[ "$RUN_ALL" == "true" ]] || [[ "$SINGLE_PR" == "$pr" ]]; then
        run_selected_for_pr "$pr"
    fi
done
