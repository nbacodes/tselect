#!/usr/bin/env bash
# ============================================================
# run_tselect_multi_pr.sh
# Runs tselect for each PR in the benchmark slate and records
# selected test files. Output goes to results/<PR>/ per PR.
#
# Usage: ./run_tselect_multi_pr.sh [--pr 145692] [--all]
#
# Prerequisites:
#   - PyTorch repo at $PYTORCH_DIR (default: ~/pytorch)
#   - tselect at $TSELECT_DIR (default: ~/Desktop/nbaworks/tselect)
#   - virtualenv pytorch-dev active, OR pass --venv <path>
# ============================================================

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────
PYTORCH_DIR="${PYTORCH_DIR:-$HOME/pytorch}"
TSELECT_DIR="${TSELECT_DIR:-$HOME/Desktop/nbaworks/tselect}"
RESULTS_DIR="${RESULTS_DIR:-$(pwd)/multi_pr_results}"
VENV="${VENV:-$HOME/.virtualenvs/pytorch-dev}"

# ── PR slate ─────────────────────────────────────────────────
# Format: "PR_NUMBER|DESCRIPTION|CHANGED_FILES(comma-sep)|CATEGORY"
declare -a PR_SLATE=(
    "145690|Add typing to simd.py|torch/_inductor/codegen/simd.py|types-only"
    "145691|Add typing to common.py|torch/_inductor/codegen/common.py|types-only"
    "145692|get_backend_features to OrderedSet|torch/_inductor/codegen/common.py,torch/_inductor/scheduler.py|types-runtime"
    "152993|Fix ModularIndexing assumptions|torch/_inductor/scheduler.py,torch/_inductor/ir.py|bug-fix"
    "121953|Fix addmm fusion check|torch/_inductor/fx_passes/mkldnn_fusion.py|bug-fix"
    "181666|Add SDK lib fallback cpp-wrapper|torch/_inductor/codecache.py,torch/_inductor/cpp_builder.py|feature"
    "181700|TMA load in fused pointwise epilogues|torch/_inductor/codegen/triton.py,torch/_inductor/ir.py|gpu-codegen"
)

# ── Argument parsing ──────────────────────────────────────────
RUN_ALL=false
SINGLE_PR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)   RUN_ALL=true; shift ;;
        --pr)    SINGLE_PR="$2"; shift 2 ;;
        --pytorch-dir) PYTORCH_DIR="$2"; shift 2 ;;
        --tselect-dir) TSELECT_DIR="$2"; shift 2 ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        --venv)  VENV="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ "$RUN_ALL" == "false" && -z "$SINGLE_PR" ]]; then
    echo "Usage: $0 --all | --pr <PR_NUMBER>"
    echo ""
    echo "Available PRs:"
    for entry in "${PR_SLATE[@]}"; do
        IFS='|' read -r pr desc files cat <<< "$entry"
        printf "  #%-8s [%-14s] %s\n" "$pr" "$cat" "$desc"
    done
    exit 0
fi

# ── Helpers ───────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

activate_venv() {
    if [[ -f "$VENV/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "$VENV/bin/activate"
    else
        log "WARNING: venv not found at $VENV — using system Python"
    fi
}

fetch_pr_diff() {
    local pr_number="$1"
    local output_file="$2"

    log "Fetching diff for PR #$pr_number from GitHub..."

    # Try GitHub API first (no auth needed for public repos, rate-limited)
    local api_url="https://api.github.com/repos/pytorch/pytorch/pulls/${pr_number}"
    local pr_data
    pr_data=$(curl -sf "${api_url}" 2>/dev/null || echo "")

    if [[ -n "$pr_data" ]]; then
        local base_sha head_sha
        base_sha=$(echo "$pr_data" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['base']['sha'])" 2>/dev/null || echo "")
        head_sha=$(echo "$pr_data" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['head']['sha'])" 2>/dev/null || echo "")

        if [[ -n "$base_sha" && -n "$head_sha" ]]; then
            log "  Base: $base_sha, Head: $head_sha"
            # Try to get diff from local git repo first (faster)
            if git -C "$PYTORCH_DIR" cat-file -e "$head_sha" 2>/dev/null; then
                git -C "$PYTORCH_DIR" diff "${base_sha}...${head_sha}" > "$output_file"
                log "  Diff saved from local git ($( wc -l < "$output_file") lines)"
                return 0
            fi
        fi
    fi

    # Fallback: use local git with PR branch fetch
    log "  Trying local git fetch of PR #$pr_number..."
    git -C "$PYTORCH_DIR" fetch origin "pull/${pr_number}/head:pr/${pr_number}" 2>/dev/null || true

    if git -C "$PYTORCH_DIR" rev-parse "pr/${pr_number}" &>/dev/null; then
        local merge_base
        merge_base=$(git -C "$PYTORCH_DIR" merge-base main "pr/${pr_number}" 2>/dev/null || \
                     git -C "$PYTORCH_DIR" merge-base origin/main "pr/${pr_number}" 2>/dev/null || echo "")
        if [[ -n "$merge_base" ]]; then
            git -C "$PYTORCH_DIR" diff "${merge_base}...pr/${pr_number}" > "$output_file"
            log "  Diff saved from local fetch ($( wc -l < "$output_file") lines)"
            return 0
        fi
    fi

    log "ERROR: Could not fetch diff for PR #$pr_number"
    return 1
}

run_tselect_for_pr() {
    local pr_number="$1"
    local description="$2"
    local expected_files="$3"
    local category="$4"

    local out_dir="$RESULTS_DIR/$pr_number"
    mkdir -p "$out_dir"

    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "PR #$pr_number — $description"
    log "Category: $category"
    log "Expected changed files: $expected_files"
    log "Output dir: $out_dir"

    # Write metadata
    cat > "$out_dir/metadata.json" <<EOF
{
  "pr": "$pr_number",
  "description": "$description",
  "expected_changed_files": $(echo "$expected_files" | python3 -c "import sys; files=sys.stdin.read().strip().split(','); print(__import__('json').dumps(files))"),
  "category": "$category",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

    # 1. Fetch diff
    local diff_file="$out_dir/pr.diff"
    if ! fetch_pr_diff "$pr_number" "$diff_file"; then
        echo "FAILED: diff fetch" > "$out_dir/status.txt"
        return 1
    fi

    # Summarize what files changed
    python3 - "$diff_file" <<'PYEOF' > "$out_dir/changed_files.txt"
import sys, re
with open(sys.argv[1]) as f:
    content = f.read()
files = re.findall(r'^diff --git a/(\S+)', content, re.MULTILINE)
print('\n'.join(files))
PYEOF

    log "  Files changed in diff:"
    while IFS= read -r f; do log "    $f"; done < "$out_dir/changed_files.txt"

    # 2. Run tselect
    log "  Running tselect..."
    activate_venv

    local tselect_log="$out_dir/tselect_run.log"
    local tselect_out="$out_dir/selected_tests.txt"

    # Try multiple ways tselect might accept input
    if python3 "$TSELECT_DIR/tselect.py" \
            --diff "$diff_file" \
            --pytorch-dir "$PYTORCH_DIR" \
            --output "$tselect_out" \
            > "$tselect_log" 2>&1; then
        log "  tselect succeeded (--diff mode)"
    elif python3 "$TSELECT_DIR/tselect.py" \
            --pr "$pr_number" \
            --pytorch-dir "$PYTORCH_DIR" \
            --output "$tselect_out" \
            >> "$tselect_log" 2>&1; then
        log "  tselect succeeded (--pr mode)"
    else
        log "  WARNING: tselect failed or used fallback. Check $tselect_log"
        # Create empty output so pipeline continues
        touch "$tselect_out"
    fi

    # Count selected tests
    local n_selected=0
    if [[ -f "$tselect_out" ]]; then
        n_selected=$(grep -c '.' "$tselect_out" 2>/dev/null || echo 0)
    fi
    log "  Selected tests: $n_selected"

    # 3. Write summary for this PR
    cat > "$out_dir/summary.txt" <<EOF
PR: #$pr_number
Description: $description
Category: $category
Changed files (from diff): $(cat "$out_dir/changed_files.txt" | wc -l | tr -d ' ')
Selected tests (tselect): $n_selected
EOF

    echo "OK" > "$out_dir/status.txt"
    log "  Done. Summary: $out_dir/summary.txt"
}

# ── Main ──────────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR"

log "tselect Multi-PR Evaluation Runner"
log "PyTorch dir : $PYTORCH_DIR"
log "tselect dir : $TSELECT_DIR"
log "Results dir : $RESULTS_DIR"
log ""

processed=0
failed=0

for entry in "${PR_SLATE[@]}"; do
    IFS='|' read -r pr desc files cat <<< "$entry"

    if [[ "$RUN_ALL" == "true" ]] || [[ "$SINGLE_PR" == "$pr" ]]; then
        if run_tselect_for_pr "$pr" "$desc" "$files" "$cat"; then
            ((processed++))
        else
            ((failed++))
            log "  FAILED for PR #$pr"
        fi
        echo ""
    fi
done

if [[ $processed -eq 0 && -z "$SINGLE_PR" ]]; then
    log "No PRs matched. Use --all or --pr <number>"
    exit 1
fi

log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Done. Processed: $processed, Failed: $failed"
log "Results at: $RESULTS_DIR"
