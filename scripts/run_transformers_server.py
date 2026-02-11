#!/usr/bin/env python3
"""
GLM-4.6V-Flash Transformers Inference Server

A FastAPI-based OpenAI-compatible server for GLM-4.6V multimodal inference
with video and image support.

Requires: transformers>=5.0.0rc1, torch, fastapi, uvicorn
"""

import base64
import io
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoProcessor, Glm4vForConditionalGeneration

# ============================================================================
# Configuration
# ============================================================================

MODEL_PATH = os.environ.get("MODEL_PATH", "zai-org/GLM-4.6V-Flash")

# Temp directory for saving uploaded media files
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
TEMP_DIR = PROJECT_DIR / "tmp"
TEMP_DIR.mkdir(exist_ok=True)
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "4096"))

# ============================================================================
# Pydantic Models (OpenAI Compatible)
# ============================================================================

class ImageUrl(BaseModel):
    url: str

class VideoUrl(BaseModel):
    url: str

class ContentPart(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None
    video_url: Optional[VideoUrl] = None

class Message(BaseModel):
    role: str
    content: Union[str, List[ContentPart]]

class ChatCompletionRequest(BaseModel):
    model: str = "glm-4.6v-flash"
    messages: List[Message]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.6
    top_k: Optional[int] = 2
    max_tokens: Optional[int] = 4096
    stream: Optional[bool] = False

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class Choice(BaseModel):
    index: int = 0
    message: Dict[str, Any]
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "zai-org"

class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]

# ============================================================================
# Model Loading
# ============================================================================

print(f"Loading model: {MODEL_PATH}")
print("This may take a few minutes on first run (downloading ~18GB)...")

# Limit video frames to prevent CUDA kernel overflow
# Reduce num_frames from 16 to 8 to prevent "too many resources requested for launch" error
processor = AutoProcessor.from_pretrained(
    MODEL_PATH, 
    trust_remote_code=True, 
    do_sample_frames=True, 
    num_frames=1,  # Reduced from 16 to prevent CUDA resource errors
)

# Use float16 for V100 compatibility (doesn't support bfloat16)
dtype = torch.float16
print(f"Using dtype: {dtype}")

# Use sdpa for memory efficiency (V100 doesn't support flash_attention_2)
model = Glm4vForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=dtype,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="sdpa",
)

print(f"Model loaded successfully on {model.device}")

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="GLM-4.6V-Flash Inference Server",
    description="OpenAI-compatible API for GLM-4.6V multimodal inference",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Helper Functions
# ============================================================================

def save_base64_to_temp_file(data_uri: str, prefix: str = "media_") -> str:
    """
    Convert base64 data URI to a temporary file for processing.
    
    Note: For images, we skip saving since the GUI already saved frames.
    Only videos are saved as files (required by transformers processor).
    
    Args:
        data_uri: "data:video/mp4;base64,..." format URI
        prefix: Prefix for temp file name
        
    Returns:
        - For images: Returns the original data URI (processed in-memory by transformers)
        - For videos: Absolute path to the saved temp file
    """
    # Parse data:video/mp4;base64,...
    header, base64_data = data_uri.split(",", 1)
    
    # Extract extension from MIME type
    mime_type = header.split(":")[1].split(";")[0]  # video/mp4
    ext_map = {
        "video/mp4": ".mp4",
        "video/avi": ".avi",
        "video/quicktime": ".mov",
        "video/x-msvideo": ".avi",
        "video/webm": ".webm",
        "video/x-matroska": ".mkv",
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
    }
    ext = ext_map.get(mime_type, ".bin")
    
    # For images: return original data URI (no file saving needed)
    # Transformers processor can handle base64 data URIs directly
    if mime_type.startswith("image/"):
        print(f"[Image] Using in-memory base64 data (no file save)")
        return data_uri
    
    # For videos: save to temp file (required for video processing)
    filename = f"{prefix}{uuid.uuid4().hex[:8]}{ext}"
    filepath = TEMP_DIR / filename
    filepath.write_bytes(base64.b64decode(base64_data))
    
    print(f"Saved video to temp file: {filepath}")
    return str(filepath)


def convert_openai_to_glm_messages(messages: List[Message]) -> List[Dict[str, Any]]:
    """
    Convert OpenAI-format messages to GLM-4.6V format.
    
    GLM requires ALL messages to have content as a list of dicts:
    - Text: {"type": "text", "text": "..."}
    - Image: {"type": "image", "url": "..."}
    - Video: {"type": "video", "url": "..."}
    """
    
    def wrap_text(text: str) -> List[Dict[str, Any]]:
        """Wrap text in GLM content format."""
        return [{"type": "text", "text": text}]
    
    def convert_content_part(part) -> Optional[Dict[str, Any]]:
        """Convert a single content part to GLM format."""
        if isinstance(part, dict):
            part_type = part.get("type", "")
            if part_type == "text":
                return {"type": "text", "text": part.get("text", "")}
            elif part_type == "image_url":
                image_data = part.get("image_url", {})
                url = image_data.get("url", "") if isinstance(image_data, dict) else ""
                # Convert base64 data URI to temp file path
                if url.startswith("data:"):
                    url = save_base64_to_temp_file(url, prefix="image_")
                return {"type": "image", "url": url}
            elif part_type == "video_url":
                video_data = part.get("video_url", {})
                url = video_data.get("url", "") if isinstance(video_data, dict) else ""
                # Convert base64 data URI to temp file path
                if url.startswith("data:"):
                    url = save_base64_to_temp_file(url, prefix="video_")
                return {"type": "video", "url": url}
        else:
            # Pydantic model
            if hasattr(part, 'type'):
                if part.type == "text":
                    return {"type": "text", "text": part.text or ""}
                elif part.type == "image_url" and part.image_url:
                    url = part.image_url.url
                    # Convert base64 data URI to temp file path
                    if url.startswith("data:"):
                        url = save_base64_to_temp_file(url, prefix="image_")
                    return {"type": "image", "url": url}
                elif part.type == "video_url" and part.video_url:
                    url = part.video_url.url
                    # Convert base64 data URI to temp file path
                    if url.startswith("data:"):
                        url = save_base64_to_temp_file(url, prefix="video_")
                    return {"type": "video", "url": url}
        return None
    
    glm_messages = []
    
    for msg in messages:
        role = msg.role
        content = msg.content
        
        # Debug: print content info
        print(f"[DEBUG] Role: {role}, Content type: {type(content).__name__}")
        if isinstance(content, list):
            print(f"[DEBUG] Content parts count: {len(content)}")
            for i, part in enumerate(content):
                if hasattr(part, 'type'):
                    print(f"[DEBUG]   Part {i}: type={part.type}, is Pydantic")
                elif isinstance(part, dict):
                    print(f"[DEBUG]   Part {i}: type={part.get('type')}, is dict")
        
        # All messages must have content as list of dicts
        if isinstance(content, str):
            # Plain text - wrap in list format
            glm_messages.append({
                "role": role,
                "content": wrap_text(content)
            })
        else:
            # Content is a list of parts
            glm_content = []
            for part in content:
                converted = convert_content_part(part)
                if converted:
                    glm_content.append(converted)
            
            # Ensure at least empty text if no content
            if not glm_content:
                glm_content = wrap_text("")
            
            glm_messages.append({
                "role": role,
                "content": glm_content
            })
    
    return glm_messages

def generate_response(
    messages: List[Dict[str, Any]],
    max_tokens: int = 4096,
    temperature: float = 0.8,
    top_p: float = 0.6,
    top_k: int = 2,
) -> str:
    """Generate response using the model."""
    
    # Apply chat template
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)
    
    # Remove token_type_ids if present (not used by GLM)
    inputs.pop("token_type_ids", None)
    
    # Generate
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=temperature > 0,
        )
    
    # Decode output (skip input tokens)
    input_len = inputs["input_ids"].shape[1]
    output_text = processor.decode(
        generated_ids[0][input_len:],
        skip_special_tokens=True
    )
    
    return output_text

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "model": MODEL_PATH}

@app.get("/v1/models")
async def list_models() -> ModelsResponse:
    """List available models."""
    return ModelsResponse(
        data=[
            ModelInfo(
                id="glm-4.6v-flash",
                created=int(time.time()),
            )
        ]
    )

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint with streaming support."""
    
    try:
        # Convert messages
        glm_messages = convert_openai_to_glm_messages(request.messages)
        
        # Debug: print converted messages
        print(f"Converted messages: {len(glm_messages)} messages")
        
        # Generate response
        start_time = time.time()
        output_text = generate_response(
            messages=glm_messages,
            max_tokens=request.max_tokens or MAX_NEW_TOKENS,
            temperature=request.temperature or 0.8,
            top_p=request.top_p or 0.6,
            top_k=request.top_k or 2,
        )
        gen_time = time.time() - start_time
        
        print(f"Generated {len(output_text)} chars in {gen_time:.2f}s")
        
        # Handle streaming mode
        if request.stream:
            async def generate_stream():
                import json
                response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                created = int(time.time())
                
                # Send content in chunks (simulate streaming)
                chunk_size = 20  # characters per chunk
                for i in range(0, len(output_text), chunk_size):
                    chunk = output_text[i:i+chunk_size]
                    data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                
                # Send final chunk
                final_data = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(final_data)}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream"
            )
        
        # Non-streaming response
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                Choice(
                    message={"role": "assistant", "content": output_text},
                    finish_reason="stop"
                )
            ],
            usage=Usage()  # Token counting not implemented
        )
        
    except Exception as e:
        import traceback
        print(f"Error during generation: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print(f"\nStarting server at http://{HOST}:{PORT}")
    print("Endpoints:")
    print(f"  - POST /v1/chat/completions")
    print(f"  - GET  /v1/models")
    print(f"  - GET  /health")
    print()
    
    uvicorn.run(app, host=HOST, port=PORT)
