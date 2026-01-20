#!/bin/bash
# GLM-4.6V-Flash Text-only Inference Script
# This script runs GLM-4.6V-Flash for text-only conversations

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LLAMA_CPP="$PROJECT_DIR/../llama.cpp"
MODEL_DIR="$PROJECT_DIR/models/GLM-4.6V-Flash-GGUF"

# Model path
MODEL="$MODEL_DIR/GLM-4.6V-Flash-Q8_0.gguf"

# Check if model exists
if [ ! -f "$MODEL" ]; then
    echo "Error: Model not found at $MODEL"
    echo "Please download the model first."
    exit 1
fi

# Default parameters (General Conversation)
TEMP="${TEMP:-0.8}"
TOP_P="${TOP_P:-0.6}"
TOP_K="${TOP_K:-2}"
REPEAT_PENALTY="${REPEAT_PENALTY:-1.1}"

# For coding tasks, you can override:
# TEMP=1.0 TOP_P=0.95 TOP_K=40 ./scripts/run_text.sh

# Run llama-cli
"$LLAMA_CPP/llama-cli" \
    --model "$MODEL" \
    --jinja \
    --temp "$TEMP" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --repeat-penalty "$REPEAT_PENALTY" \
    "$@"
