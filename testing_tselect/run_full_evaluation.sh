#!/usr/bin/env bash
# ============================================================
# run_full_evaluation.sh
# Master orchestrator for the 3-phase tselect evaluation.
#
#   Phase A: Collect ground truth coverage (run ALL tests)
#   Phase B: Run tselect on each PR diff
#   Phase C: Run only selected tests, collect coverage
#   Phase D: Compute confusion matrices + report
#
# Usage:
#   ./run_full_evaluation.sh                    # all 7 PRs
#   ./run_full_evaluation.sh --pr 152993        # single PR
#   ./run_full_evaluation.sh --skip-ground-truth  # if A already done
#   ./run_full_evaluation.sh --phase b,c,d      # specific phases
#
# Env vars (all optional):
#   PYTORCH_DIR   - path to pytorch checkout
#   TSELECT_DIR   - path to tselect source
#   RESULTS_DIR   - output directory
#   VENV          - path to virtualenv
# ============================================================

set -euo pipefail

export PYTORCH_DIR=/Users/nihalkumar/pytorch
export TSELECT_DIR=/Users/nihalkumar/Desktop/nbaworks/tselect
export RESULTS_DIR=/Users/nihalkumar/Desktop/nbaworks/tselect/tselect_test
export VENV=/Users/nihalkumar/envs/pytorch-dev

PYTORCH_DIR="${PYTORCH_DIR:-$HOME/pytorch}"
TSELECT_DIR="${TSELECT_DIR:-$HOME/Desktop/nbaworks/tselect}"
RESULTS_DIR="${RESULTS_DIR:-$(pwd)/multi_pr_results}"
VENV="${VENV:-$HOME/.virtualenvs/pytorch-dev}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log()  { echo ""; echo "╔══ [$(date '+%H:%M:%S')] $*"; }
step() { echo "║  $*"; }
finish() { echo "╚══ Done: $*"; echo ""; }

# ── Parse args ────────────────────────────────────────────────
SINGLE_PR=""
PHASES="a,b,c,d"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pr)                SINGLE_PR="$2"; shift 2 ;;
        --phases)            PHASES="$2"; shift 2 ;;
        --skip-ground-truth) PHASES=$(echo "$PHASES" | sed 's/a,//;s/,a//;s/^a$//'); shift ;;
        --pytorch-dir)       PYTORCH_DIR="$2"; shift 2 ;;
        --tselect-dir)       TSELECT_DIR="$2"; shift 2 ;;
        --results-dir)       RESULTS_DIR="$2"; shift 2 ;;
        --venv)              VENV="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

PR_ARG="${SINGLE_PR:+--pr $SINGLE_PR}"
MODE="${SINGLE_PR:+single PR #$SINGLE_PR}"
MODE="${MODE:-all 7 PRs}"

COMMON_ENV="PYTORCH_DIR=$PYTORCH_DIR TSELECT_DIR=$TSELECT_DIR RESULTS_DIR=$RESULTS_DIR VENV=$VENV"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   tselect Multi-PR Evaluation — $MODE"
echo "╠══════════════════════════════════════════════════╣"
echo "║  PyTorch dir : $PYTORCH_DIR"
echo "║  tselect dir : $TSELECT_DIR"
echo "║  Results dir : $RESULTS_DIR"
echo "║  Phases      : $PHASES"
echo "╚══════════════════════════════════════════════════╝"

mkdir -p "$RESULTS_DIR"

# ── Phase A: Ground truth ─────────────────────────────────────
if echo "$PHASES" | grep -q 'a'; then
    log "Phase A — Ground Truth Collection"
    step "Running all relevant inductor test suites"
    step "This is the expensive phase — can take 30–60 min depending on hardware"

    env $COMMON_ENV bash "$SCRIPT_DIR/collect_ground_truth.sh" \
        ${SINGLE_PR:+--pr $SINGLE_PR} \
        ${SINGLE_PR:-"--all"}

    finish "Ground truth collected at $RESULTS_DIR/ground_truth/"
fi

# ── Phase B: tselect ─────────────────────────────────────────
if echo "$PHASES" | grep -q 'b'; then
    log "Phase B — tselect Execution"
    step "Running tselect for each PR to get selected test lists"

    env $COMMON_ENV bash "$SCRIPT_DIR/run_tselect_multi_pr.sh" \
        ${SINGLE_PR:+--pr $SINGLE_PR} \
        ${SINGLE_PR:-"--all"}

    finish "tselect outputs at $RESULTS_DIR/<PR>/selected_tests.txt"

    # Print a quick summary
    echo "  Selected test counts:"
    for pr_dir in "$RESULTS_DIR"/*/; do
        pr=$(basename "$pr_dir")
        [[ "$pr" == "ground_truth" ]] && continue
        sel_file="$pr_dir/selected_tests.txt"
        n=0
        [[ -f "$sel_file" ]] && n=$(grep -c '.' "$sel_file" 2>/dev/null || echo 0)
        printf "    PR #%-8s : %3s tests selected\n" "$pr" "$n"
    done
fi

# ── Phase C: Run selected tests ───────────────────────────────
if echo "$PHASES" | grep -q 'c'; then
    log "Phase C — Selected Test Execution"
    step "Running only tselect-selected tests for each PR"

    env $COMMON_ENV bash "$SCRIPT_DIR/run_selected_tests.sh" \
        ${SINGLE_PR:+--pr $SINGLE_PR} \
        ${SINGLE_PR:-"--all"}

    finish "Coverage at $RESULTS_DIR/<PR>/selected_coverage/coverage.json"
fi

# ── Phase D: Confusion matrices ───────────────────────────────
if echo "$PHASES" | grep -q 'd'; then
    log "Phase D — Confusion Matrix Computation"
    step "Comparing Phase A vs Phase C coverage"

    python3 "$SCRIPT_DIR/compute_confusion_matrix.py" \
        --results-dir "$RESULTS_DIR" \
        ${SINGLE_PR:+--pr $SINGLE_PR} \
        --save-json "$RESULTS_DIR/matrix_report.json"

    # Also save CSV for spreadsheet use
    python3 "$SCRIPT_DIR/compute_confusion_matrix.py" \
        --results-dir "$RESULTS_DIR" \
        ${SINGLE_PR:+--pr $SINGLE_PR} \
        --format csv \
        > "$RESULTS_DIR/matrix_report.csv"

    finish "Reports at $RESULTS_DIR/matrix_report.{json,csv}"
fi

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   Evaluation complete.                           ║"
echo "║   Results: $RESULTS_DIR"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "Key output files:"
echo "  matrix_report.json  — machine-readable full report"
echo "  matrix_report.csv   — spreadsheet-ready"
echo "  <PR>/selected_tests.txt          — what tselect picked"
echo "  <PR>/selected_coverage/          — Phase C coverage"
echo "  ground_truth/<module>/           — Phase A coverage"
