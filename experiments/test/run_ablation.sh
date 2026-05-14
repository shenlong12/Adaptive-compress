#!/bin/bash
# ============================================================
# SADS Threshold Ablation — run_ablation.sh
# ============================================================
# Usage:
#   bash run_ablation.sh              # full eval (all samples)
#   bash run_ablation.sh --quick      # 200 samples/dataset for smoke test
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_PY="$SCRIPT_DIR/ablation_tau_sweep.py"
RESULTS_DIR="$SCRIPT_DIR/results"
LOG_FILE="$RESULTS_DIR/ablation_run.log"

mkdir -p "$RESULTS_DIR"

# Default: full evaluation
LIMIT=-1

if [[ "$1" == "--quick" ]]; then
    echo "[INFO] Quick mode: 200 samples per dataset"
    LIMIT=200
fi

echo "============================================"
echo " SADS Threshold Ablation"
echo " Sparsity : 40%"
echo " Taus     : inf 6.0 5.0 4.0 0.0"
echo " Limit    : $LIMIT"
echo " Log      : $LOG_FILE"
echo "============================================"

python "$SWEEP_PY" \
    --taus inf 6.0 5.0 4.0 0.0 \
    --sparsity 40 \
    --limit "$LIMIT" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================"
echo " Done. Results:"
echo "   CSV  → $RESULTS_DIR/ablation_tau_results.csv"
echo "   JSON → $RESULTS_DIR/ablation_tau_results.json"
echo "   Log  → $LOG_FILE"
echo "============================================"