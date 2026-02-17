#!/bin/bash
# DGRCL Feature Ablation Study Runner on AMD GPU
# Wraps run_experiments.py with necessary environment variables for ROCm/HIP

# HSA override for Navi 23 (RX 6600) compatibility with ROCm
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# PyTorch HIP memory allocator improvements
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

echo "=== DGRCL Ablation Study Environment (AMD GPU) ==="
echo "HSA_OVERRIDE_GFX_VERSION: $HSA_OVERRIDE_GFX_VERSION"
echo "PYTORCH_HIP_ALLOC_CONF: $PYTORCH_HIP_ALLOC_CONF"
echo "=================================================="

# Activate venv
source venv/bin/activate

# Run the experiment runner
# Passes all arguments through to run_experiments.py
# Example: ./run_feature_ablation_study.sh --epochs 50 --end-fold 5
python run_experiments.py "$@"
