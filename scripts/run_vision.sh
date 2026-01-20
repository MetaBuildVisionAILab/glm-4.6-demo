#!/bin/bash
# GLM-4.6V-Flash Multimodal (Vision) Inference Script
# This script runs GLM-4.6V-Flash with image support

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
    echo "Error: MMProj model not found at $MMPROJ"
    echo "Please download the mmproj model first."
    exit 1
fi

# Default parameters (General Conversation)
TEMP="${TEMP:-0.8}"
TOP_P="${TOP_P:-0.6}"
TOP_K="${TOP_K:-2}"
REPEAT_PENALTY="${REPEAT_PENALTY:-1.1}"

# Image file (required)
IMAGE="${IMAGE:-$1}"

if [ -z "$IMAGE" ] || [ ! -f "$IMAGE" ]; then
    echo "Error: Please provide a valid image file"
    echo "Usage: $0 <image_path> [prompt...]"
    echo "Or: IMAGE=path/to/image.jpg $0 [prompt...]"
    exit 1
fi

# Run llama-mtmd-cli (multimodal)
"$LLAMA_CPP/llama-mtmd-cli" \
    --model "$MODEL" \
    --mmproj "$MMPROJ" \
    --image "$IMAGE" \
    --jinja \
    --temp "$TEMP" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --repeat-penalty "$REPEAT_PENALTY" \
    "${@:2}"
