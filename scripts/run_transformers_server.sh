#!/bin/bash
# GLM-4.6V-Flash Transformers Server
# FastAPI-based OpenAI-compatible server with video support

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Server configuration
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"

# Model path (HuggingFace model ID or local path)
export MODEL_PATH="${MODEL_PATH:-zai-org/GLM-4.6V-Flash}"

echo "=========================================="
echo "GLM-4.6V-Flash Transformers Server"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Host: $HOST:$PORT"
echo "Max Tokens: $MAX_NEW_TOKENS"
echo "=========================================="
echo ""

# Check if transformers 5.x is installed
TRANSFORMERS_VERSION=$("$PROJECT_DIR/.venv/bin/python" -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "not installed")
echo "Transformers version: $TRANSFORMERS_VERSION"

if [[ ! "$TRANSFORMERS_VERSION" =~ ^5\. ]]; then
    echo ""
    echo "WARNING: transformers>=5.0.0rc1 is required for video support."
    echo "Install with: pip install 'transformers>=5.0.0rc1'"
    echo ""
fi

# Run the server
exec "$PROJECT_DIR/.venv/bin/python" "$SCRIPT_DIR/run_transformers_server.py" "$@"
