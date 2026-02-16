#!/bin/bash
# Run script for DGRCL training on AMD GPU (RX 6600)
# Sets required environment variables for ROCm compatibility

# HSA override for Navi 23 (RX 6600) compatibility with ROCm
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# PyTorch HIP memory allocator improvements
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# Optional: Enable logging for debugging
# export TORCH_DISTRIBUTED_DEBUG=INFO

echo "=== AMD GPU Environment Setup ==="
echo "HSA_OVERRIDE_GFX_VERSION: $HSA_OVERRIDE_GFX_VERSION"
echo "PYTORCH_HIP_ALLOC_CONF: $PYTORCH_HIP_ALLOC_CONF"
echo "=================================="
echo ""

# Activate venv and run training
source venv/bin/activate

# Run the training script with all passed arguments
python train.py "$@"
