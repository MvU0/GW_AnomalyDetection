#!/bin/bash
source ~/.bashrc
conda activate your_env
cd /your_dir/to_mainfunction/


# Determine device
if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
    DEVICE="cpu"
else
    DEVICE="cuda"
fi


# Debug info
echo "====================================="
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Using device: $DEVICE"
echo "====================================="

# Run python
python3 main.py "$@" -device $DEVICE