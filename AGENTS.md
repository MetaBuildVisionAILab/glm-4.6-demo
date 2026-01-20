# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains a GLM-4.6V-Flash (9B multimodal vision model) inference setup. The project is designed for Tesla V100 32GB × 2 GPUs (64GB VRAM total).

**Two Backend Options:**
- **llama.cpp** - Fast inference with GGUF quantized models (image support only)
- **vLLM** - Full multimodal support including **video input** (requires HuggingFace model)

## Project Structure

```
glm-4.6v/
├── models/GLM-4.6V-Flash-GGUF/     # GGUF model weights (for llama.cpp)
│   ├── GLM-4.6V-Flash-Q8_0.gguf    # Main model (~10GB)
│   └── mmproj-F16.gguf              # Vision encoder (~1.8GB)
├── scripts/                         # Inference scripts
│   ├── run_text.sh                  # Text-only inference (llama.cpp)
│   ├── run_vision.sh                # Multimodal inference (llama.cpp)
│   ├── run_server.sh                # API server (llama.cpp, images only)
│   └── run_vllm_server.sh           # API server (vLLM, images + videos)
├── streamlit_gui.py                 # Streamlit web GUI
├── requirements_gui.txt             # Python dependencies for GUI
└── test_images/                     # Test images directory
```

**Important**: The llama.cpp binary directory is at `../llama.cpp` relative to this project (i.e., `/home/radar01/sdb1/jw/llama.cpp`). All scripts reference this location.

## Running Inference

### Text-Only Mode

```bash
./scripts/run_text.sh
./scripts/run_text.sh --prompt "Your question here"
```

### Multimodal Mode (Image + Text)

```bash
./scripts/run_vision.sh path/to/image.jpg "Describe this image"
IMAGE=path/to/image.jpg ./scripts/run_vision.sh "What's in this picture?"
```

### Server Mode - llama.cpp (Images Only)

```bash
./scripts/run_server.sh
```

The llama.cpp server starts on `http://0.0.0.0:8000` and provides OpenAI-compatible endpoints. **Note:** llama.cpp does not support video input.

### Server Mode - vLLM (Images + Videos)

```bash
./scripts/run_vllm_server.sh
```

The vLLM server provides **full video support** via the `video_url` content type. First-time run will download the model from HuggingFace (~18GB).

**Configuration via environment variables:**
```bash
TENSOR_PARALLEL=2 ./scripts/run_vllm_server.sh    # Use 2 GPUs
MAX_MODEL_LEN=32768 ./scripts/run_vllm_server.sh  # Larger context for videos
GPU_MEMORY_UTILIZATION=0.85 ./scripts/run_vllm_server.sh  # Reduce if OOM
```

**Requirements:**
- `pip install vllm>=0.6.0`
- HuggingFace model: `zai-org/GLM-4.6V-Flash` (auto-downloaded)

### Server Mode - Transformers (Images + Videos, Recommended)

```bash
./scripts/run_transformers_server.sh
```

The Transformers server provides **full video and image support** using transformers 5.x directly. This is the recommended method for video input.

**Configuration via environment variables:**
```bash
PORT=8000 ./scripts/run_transformers_server.sh           # Custom port
MAX_NEW_TOKENS=8192 ./scripts/run_transformers_server.sh # More output tokens
```

**Requirements:**
- `pip install "transformers>=5.0.0rc1" fastapi uvicorn accelerate`
- HuggingFace model: `zai-org/GLM-4.6V-Flash` (auto-downloaded ~18GB)

All servers provide OpenAI-compatible endpoints:
- `POST /v1/chat/completions` - Chat with multimodal support
- `GET /health` - Health check
- `GET /v1/models` - List available models

### Streamlit GUI

```bash
# Install dependencies first
pip install -r requirements_gui.txt

# Start the GUI
streamlit run streamlit_gui.py --server.port 8501
```

The GUI provides:
- Chat interface with history
- System and user prompt inputs
- Image upload (single/multiple)
- Video upload (direct processing via `video_url` type)
- Parameter controls (Temperature, Top P, Top K, Max Tokens)
- Streaming response display
- Server status monitoring

## Configuration

### Model Parameters

Default parameters (General Conversation) are set in the scripts:
- `TEMP=0.8`, `TOP_P=0.6`, `TOP_K=2`, `REPEAT_PENALTY=1.1`

For coding tasks, override with environment variables:
```bash
TEMP=1.0 TOP_P=0.95 TOP_K=40 ./scripts/run_text.sh
```

### Server Configuration

Configure via environment variables:
```bash
HOST=127.0.0.1 PORT=8080           # Custom host/port
CTX_SIZE=32768                      # Context size (default: 16384)
N_PARALLEL=8                        # Parallel slots (default: 4)
SYSTEM_PROMPT="..."                 # System prompt for behavior
```

### Language Response

The model may respond in Chinese by default. Force English responses:
```bash
./scripts/run_text.sh --prompt "Respond in English and reason in English. What is AI?"
SYSTEM_PROMPT="Respond in English and reason in English" ./scripts/run_server.sh
```

## Model Quantization

The current setup uses Q8_0 quantization (~10GB). Other quantizations are available from [Unsloth/GLM-4.6V-Flash-GGUF](https://huggingface.co/unsloth/GLM-4.6V-Flash-GGUF):

| Quantization | Size | VRAM Required |
|--------------|------|---------------|
| BF16 | ~18.8GB | ~24GB |
| Q8_0 | ~10GB | ~16GB |
| Q6_K | ~8.27GB | ~14GB |
| Q5_K_M | ~7.05GB | ~12GB |
| Q4_K_M | ~6.17GB | ~10GB |

To switch quantizations, download the model file and update the `MODEL` variable in `scripts/run_*.sh`.

## Binaries Used

The scripts use these llama.cpp binaries (located at `../llama.cpp/`):
- `llama-cli` - Text-only inference (run_text.sh)
- `llama-mtmd-cli` - Multimodal inference (run_vision.sh)
- `llama-server` - API server (run_server.sh)

All scripts use the `--jinja` flag for Jinja template support.

## GPU Memory Management

Monitor GPU usage with:
```bash
watch -n 1 nvidia-smi
```

For OOM issues, reduce context size:
```bash
CTX_SIZE=8192 ./scripts/run_text.sh
```

## Troubleshooting

- **Model not found**: Ensure model files are in `models/GLM-4.6V-Flash-GGUF/`
- **Slow inference**: Verify CUDA is working with `nvidia-smi`, check GPU utilization
- **Chinese responses**: Add `SYSTEM_PROMPT` or use `--prompt` with English instruction
- **Out of memory**: Reduce `CTX_SIZE` or use lower quantization (e.g., Q4_K_M)
- **GUI won't connect**: Ensure llama-server is running first with `./scripts/run_server.sh`

## Multimodal API Format

GLM-4.6V supports both images and videos through the `/v1/chat/completions` endpoint. Media is sent as base64 data URIs.

### Image Format
```json
{
  "type": "image_url",
  "image_url": {"url": "data:image/jpeg;base64,..."}
}
```

### Video Format
```json
{
  "type": "video_url",
  "video_url": {"url": "data:video/mp4;base64,..."}
}
```

The model processes videos directly - no manual frame extraction required. Supported formats: MP4, AVI, MOV, MKV, WebM.

## References

- [llama.cpp AGENTS.md](../llama.cpp/AGENTS.md) - Review before making changes
- [llama.cpp repository](https://github.com/ggml-org/llama.cpp)
- [GLM-4.6V-Flash-GGUF model](https://huggingface.co/unsloth/GLM-4.6V-Flash-GGUF)
