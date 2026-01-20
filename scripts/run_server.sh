#!/bin/bash
# GLM-4.6V-Flash Server Script
# This script starts llama-server with GLM-4.6V-Flash model

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LLAMA_CPP="$PROJECT_DIR/../llama.cpp"
MODEL_DIR="$PROJECT_DIR/models/GLM-4.6V-Flash-GGUF"

# Model paths
MODEL="$MODEL_DIR/GLM-4.6V-Flash-Q8_0.gguf"
MMPROJ="$MODEL_DIR/mmproj-F16.gguf"

# Check if models exist
if [ ! -f "$MODEL" ]; then
    echo "Error: Main model not found at $MODEL"
    echo "Please download the model first."
    exit 1
fi

if [ ! -f "$MMPROJ" ]; then
    echo "Warning: MMProj model not found at $MMPROJ"
    echo "Server will start in text-only mode."
fi

# Server configuration
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
ALIAS="${ALIAS:-glm-4.6v-flash}"

# Context size (can be adjusted based on VRAM)
CTX_SIZE="${CTX_SIZE:-16384}"

# Number of slots for parallel processing
N_PARALLEL="${N_PARALLEL:-4}"

# Run llama-server
if [ -f "$MMPROJ" ]; then
    # Multimodal mode
    echo "Starting GLM-4.6V-Flash server in multimodal mode..."
    "$LLAMA_CPP/llama-server" \
        --model "$MODEL" \
        --mmproj "$MMPROJ" \
        --alias "$ALIAS" \
        --host "$HOST" \
        --port "$PORT" \
        --ctx-size "$CTX_SIZE" \
        --parallel "$N_PARALLEL" \
        --jinja \
        "$@"
else
    # Text-only mode
    echo "Starting GLM-4.6V-Flash server in text-only mode..."
    "$LLAMA_CPP/llama-server" \
        --model "$MODEL" \
        --alias "$ALIAS" \
        --host "$HOST" \
        --port "$PORT" \
        --ctx-size "$CTX_SIZE" \
        --parallel "$N_PARALLEL" \
        --jinja \
        "$@"
fi
