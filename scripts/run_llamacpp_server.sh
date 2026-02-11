#!/bin/bash
# GLM-4.6V llama-server Startup Script
# Uses llama.cpp's native llama-server with vision support (mmproj)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# llama-server binary path
LLAMA_SERVER="${LLAMA_SERVER:-/home/radar01/sdb1/jw/llama.cpp/build/bin/llama-server}"

# Model paths
MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/GLM-4.6V-Flash-GGUF/GLM-4.6V-Flash-Q5_K_M.gguf}"
MMPROJ_PATH="${MMPROJ_PATH:-$PROJECT_DIR/models/GLM-4.6V-Flash-GGUF/mmproj-F16.gguf}"

# Server settings
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
CTX_SIZE=128000
N_GPU_LAYERS=-1

echo "=========================================="
echo "GLM-4.6V llama-server (Vision Enabled)"
echo "=========================================="
echo "Binary:  $LLAMA_SERVER"
echo "Model:   $MODEL_PATH"
echo "MMProj:  $MMPROJ_PATH"
echo "Host:    $HOST:$PORT"
echo "Context: $CTX_SIZE"
echo "GPU Layers: $N_GPU_LAYERS"
echo "=========================================="

# Check if binary exists
if [ ! -f "$LLAMA_SERVER" ]; then
    echo "❌ Error: llama-server not found at $LLAMA_SERVER"
    echo "   Build llama.cpp first or set LLAMA_SERVER environment variable"
    exit 1
fi

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Error: Model not found at $MODEL_PATH"
    exit 1
fi

# Check if mmproj exists
if [ ! -f "$MMPROJ_PATH" ]; then
    echo "⚠️ Warning: mmproj not found at $MMPROJ_PATH"
    echo "   Running in text-only mode"
    MMPROJ_FLAG=""
else
    MMPROJ_FLAG="--mmproj $MMPROJ_PATH"
fi

echo ""
echo "🚀 Starting llama-server..."
echo ""

# Run llama-server
exec "$LLAMA_SERVER" \
    -m "$MODEL_PATH" \
    $MMPROJ_FLAG \
    --host "$HOST" \
    --port "$PORT" \
    -c "$CTX_SIZE" \
    -ngl "$N_GPU_LAYERS" \
    --mmproj-offload
