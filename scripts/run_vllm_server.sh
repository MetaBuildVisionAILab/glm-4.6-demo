#!/bin/bash
# vLLM Server for GLM-4.6V-Flash with Video Support
# This script starts vLLM OpenAI-compatible server for multimodal inference
# including video input support (not supported by llama.cpp)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Model configuration
# Use local path if model is downloaded, otherwise use HuggingFace model ID
MODEL="${MODEL:-zai-org/GLM-4.6V-Flash}"

# Server configuration
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# GPU configuration
# Set to 2 for multi-GPU (tensor parallelism across 2 V100s)
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"

# Context size (GLM-4.6V supports up to 128K, but limited by VRAM)
# Increase for longer video processing
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

# GPU memory utilization (0.0-1.0)
# Reduce if OOM errors occur
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

# Served model name (matches llama.cpp alias for GUI compatibility)
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm-4.6v-flash}"

echo "=========================================="
echo "GLM-4.6V-Flash vLLM Server"
echo "=========================================="
echo "Model: $MODEL"
echo "Host: $HOST:$PORT"
echo "Tensor Parallel: $TENSOR_PARALLEL GPU(s)"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "=========================================="
echo ""

# Check if vllm is installed
if ! command -v vllm &> /dev/null; then
    echo "Error: vLLM is not installed."
    echo "Install with: pip install vllm>=0.6.0"
    exit 1
fi

# Start vLLM server
# Note: --enforce-eager disables CUDA graphs and FlashAttention for V100 compatibility
exec vllm serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --trust-remote-code \
    --served-model-name "$SERVED_MODEL_NAME" \
    --enforce-eager \
    "$@"
