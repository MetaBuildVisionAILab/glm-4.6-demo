# GLM-4.6V-Flash with llama.cpp

This directory contains the setup for running GLM-4.6V-Flash (9B multimodal vision model) using llama.cpp.

## Environment

- **GPU**: Tesla V100 32GB × 2 (64GB VRAM total)
- **Model**: GLM-4.6V-Flash (9B parameters)
- **Backend**: llama.cpp with CUDA acceleration

## Directory Structure

```
glm-4.6v/
├── models/
│   └── GLM-4.6V-Flash-GGUF/
│       ├── GLM-4.6V-Flash-Q8_0.gguf       # Main model (~10GB)
│       └── mmproj-F16.gguf                # Vision encoder (~1.8GB)
├── scripts/
│   ├── run_text.sh                        # Text-only inference
│   ├── run_vision.sh                      # Multimodal (image+text) inference
│   └── run_server.sh                      # API server mode
├── test_images/                           # Test images folder
└── README.md                              # This file
```

## Quick Start

### 1. Text-Only Inference

```bash
./scripts/run_text.sh
```

Or provide a prompt directly:

```bash
./scripts/run_text.sh --prompt "What is the capital of France?"
```

### 2. Multimodal Inference (Image + Text)

```bash
./scripts/run_vision.sh path/to/image.jpg "Describe this image in detail."
```

Or use environment variable:

```bash
IMAGE=path/to/image.jpg ./scripts/run_vision.sh "What's in this picture?"
```

### 3. Server Mode (API)

Start the server:

```bash
./scripts/run_server.sh
```

The server will start on `http://0.0.0.0:8000` by default.

Then interact with it via curl or any HTTP client:

```bash
# Text-only completion
curl http://localhost:8000/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?"}'

# Multimodal with base64 image
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4.6v-flash",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What is in this image?"},
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]
      }
    ]
  }'
```

## Model Parameters

### Recommended Settings (GLM-4.6V-Flash Official)

| Use Case | top_p | top_k | temperature | repeat_penalty |
|----------|-------|-------|-------------|----------------|
| General Conversation | 0.6 | 2 | 0.8 | 1.1 |
| Coding | 0.95 | 40 | - | 1.1 |

### Overriding Parameters

You can override default parameters via environment variables:

```bash
# For coding tasks
TEMP=1.0 TOP_P=0.95 TOP_K=40 ./scripts/run_text.sh

# For more focused responses
TEMP=0.5 TOP_P=0.5 TOP_K=1 ./scripts/run_text.sh
```

## Server Configuration

Configure the server via environment variables:

```bash
# Custom host/port
HOST=127.0.0.1 PORT=8080 ./scripts/run_server.sh

# Larger context size (requires more VRAM)
CTX_SIZE=32768 ./scripts/run_server.sh

# More parallel slots
N_PARALLEL=8 ./scripts/run_server.sh
```

## Language Response

The model may respond in Chinese by default. To force English responses:

```bash
# Add system prompt via command
./scripts/run_text.sh --prompt "Respond in English and reason in English. What is AI?"

# Or use in server mode
SYSTEM_PROMPT="Respond in English and reason in English" ./scripts/run_server.sh
```

## Model Quantization Options

The scripts use Q8_0 quantization by default (~10GB). Other options from Unsloth:

| Quantization | Size | Quality | VRAM Required |
|--------------|------|---------|---------------|
| BF16 | ~18.8GB | Best | ~24GB |
| Q8_0 | ~10GB | Near-BF16 | ~16GB |
| Q6_K | ~8.27GB | Excellent | ~14GB |
| Q5_K_M | ~7.05GB | Very Good | ~12GB |
| Q5_K_S | ~6.7GB | Very Good | ~11GB |
| Q4_K_M | ~6.17GB | Good | ~10GB |
| Q4_K_S | ~5.76GB | Good | ~9GB |
| Q4_0 | ~5.48GB | Good | ~9GB |
| Q3_K_M | ~4.97GB | Fair | ~8GB |
| Q2_K | ~4.01GB | Fastest | ~7GB |

To use a different quantization, download the model files from [Unsloth/GLM-4.6V-Flash-GGUF](https://huggingface.co/unsloth/GLM-4.6V-Flash-GGUF) and update the `MODEL` variable in the scripts.

## GPU Memory Management

With 2x Tesla V100 (64GB total):

- **Q8_0**: Model fits entirely in GPU memory (~16GB VRAM)
- **Q4_K_M**: Requires ~10GB VRAM
- **BF16**: Requires ~24GB VRAM
- **Multiple concurrent requests**: Adjust `N_PARALLEL` based on your VRAM

Monitor GPU usage:

```bash
watch -n 1 nvidia-smi
```

## Troubleshooting

### Out of Memory

Reduce context size or model quantization:

```bash
CTX_SIZE=8192 ./scripts/run_text.sh
```

### Slow Inference

1. Ensure CUDA is working: Check that `nvidia-smi` shows GPU utilization
2. Reduce context size if KV cache is growing too large
3. Use lower quantization (Q4_K_M instead of Q8_K_XL)

### Model Not Found

Ensure the model files are in the correct location:

```bash
ls -lh models/GLM-4.6V-Flash-GGUF/
```

### Chinese Responses

Add the system prompt to force English:

```bash
./scripts/run_text.sh --system-prompt "Respond in English and reason in English."
```

## Performance Tips

1. **Batch requests**: Use server mode with `N_PARALLEL > 1` for concurrent processing
2. **Context size**: Use `CTX_SIZE` to balance quality vs memory
3. **Quantization**: Q8_K_XL offers near-BF16 quality at half the size
4. **KV cache**: llama.cpp automatically manages KV cache between GPU/CPU

## API Endpoints (Server Mode)

When running `./scripts/run_server.sh`:

- **Health**: `GET /health`
- **List Slots**: `GET /slots`
- **Completion**: `POST /completion`
- **Chat Completion**: `POST /v1/chat/completions`
- **Models**: `GET /v1/models`
- **Embeddings**: `POST /v1/embeddings`

See [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/tree/master/examples/server) for more details.

## References

- [llama.cpp](https://github.com/ggml-org/llama.cpp) - Main repository
- [GLM-4.6V-Flash-GGUF](https://huggingface.co/unsloth/GLM-4.6V-Flash-GGUF) - Model source
- [GLM-4V Official](https://github.com/THUDM/GLM-4) - Original model

## License

This setup uses:
- llama.cpp (MIT License)
- GLM-4.6V-Flash (Model-specific license - check HuggingFace model card)
