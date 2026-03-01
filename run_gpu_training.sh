#!/bin/bash
# DGRCL v1.6 — Full Pipeline: Backtest + Confidence Calibration
# Runs on AMD GPU (RX 6600) with required ROCm environment variables.
#
# Usage:
#   bash run_gpu_training.sh --real-data                      # backtest only
#   bash run_gpu_training.sh --real-data --save-calibration-data  # backtest + calibration
#
# All train.py arguments are forwarded. Common flags:
#   --start-fold N        Resume from fold N (1-based)
#   --end-fold N          Stop at fold N (inclusive)
#   --mag-weight 0.05     Override base λ
#   --epochs 100          Max epochs per fold
#   --output-dir DIR      Results directory (default: ./backtest_results)
#   --save-calibration-data   Save val-fold logits/labels for calibration step

# ─── AMD GPU Environment ─────────────────────────────────────────────
# HSA override for Navi 23 (RX 6600) compatibility with ROCm
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# PyTorch HIP memory allocator improvements
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

echo "=== AMD GPU Environment Setup ==="
echo "HSA_OVERRIDE_GFX_VERSION: $HSA_OVERRIDE_GFX_VERSION"
echo "PYTORCH_HIP_ALLOC_CONF:  $PYTORCH_HIP_ALLOC_CONF"
echo "=================================="
echo ""

# ─── Activate venv ────────────────────────────────────────────────────
source venv/bin/activate

# ─── Determine output directory from args (for calibration step) ──────
OUTPUT_DIR="./backtest_results"
SAVE_CALIBRATION=false
for arg in "$@"; do
    if [[ "$prev_arg" == "--output-dir" ]]; then
        OUTPUT_DIR="$arg"
    fi
    if [[ "$arg" == "--save-calibration-data" ]]; then
        SAVE_CALIBRATION=true
    fi
    prev_arg="$arg"
done

# ─── Step 1: Walk-Forward Backtest ────────────────────────────────────
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Step 1/2: Walk-Forward Backtest (train.py)         ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

python train.py "$@"
BACKTEST_EXIT=$?

if [ $BACKTEST_EXIT -ne 0 ]; then
    echo ""
    echo "❌ Backtest failed with exit code $BACKTEST_EXIT"
    exit $BACKTEST_EXIT
fi

echo ""
echo "✅ Backtest complete."

# ─── Step 2: Confidence Calibration (if calibration data was saved) ───
if [ "$SAVE_CALIBRATION" = true ]; then
    LOGITS_FILE="$OUTPUT_DIR/calibration_logits.json"
    LABELS_FILE="$OUTPUT_DIR/calibration_labels.json"

    if [ -f "$LOGITS_FILE" ] && [ -f "$LABELS_FILE" ]; then
        echo ""
        echo "╔══════════════════════════════════════════════════════╗"
        echo "║  Step 2/2: Confidence Calibration (Recs 2 & 6)      ║"
        echo "╚══════════════════════════════════════════════════════╝"
        echo ""

        python confidence_calibration.py --results-dir "$OUTPUT_DIR"
        CAL_EXIT=$?

        if [ $CAL_EXIT -ne 0 ]; then
            echo ""
            echo "⚠ Calibration failed (exit code $CAL_EXIT) — backtest results are still valid."
        else
            echo ""
            echo "✅ Calibration complete. Artifacts saved to: $OUTPUT_DIR"
        fi
    else
        echo ""
        echo "⚠ Calibration data files not found in $OUTPUT_DIR — skipping calibration."
        echo "  (Did the backtest save any fold data?)"
    fi
else
    echo ""
    echo "ℹ  Calibration skipped (add --save-calibration-data to enable)."
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Pipeline complete. Results in: $OUTPUT_DIR"
echo "═══════════════════════════════════════════════════════"
