#!/usr/bin/env python3
"""
GLM-4.6V Inference GUI (Gradio 6.0)
A Gradio web interface for GLM-4.6V multimodal model inference.
"""

import base64
import io
import json
import requests
from typing import Generator, List, Dict, Any, Tuple

import cv2
import gradio as gr
from PIL import Image

# ============================================================================
# Configuration
# ============================================================================

PAGE_TITLE = "GLM-4.6V Inference GUI"
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8000

# Global config (mutable via UI)
CONFIG = {
    "host": DEFAULT_HOST,
    "port": DEFAULT_PORT,
    "system_prompt": "You are a helpful AI assistant.",
    "temperature": 0.8,
    "top_p": 0.6,
    "top_k": 2,
    "max_tokens": 4096,
    "video_fps": 1.0,
    "max_frames": 8,
}


# ============================================================================
# API Functions
# ============================================================================

def get_server_url() -> str:
    return f"http://{CONFIG['host']}:{CONFIG['port']}"


def check_server_health() -> str:
    try:
        response = requests.get(f"{get_server_url()}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("model_loaded"):
                return "✅ Model loaded and ready"
            return "⏳ Model is loading..."
        return f"⚠️ Status: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to server"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def encode_file_base64(file_path: str) -> str:
    with open(file_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    
    ext = file_path.lower().split(".")[-1]
    mime_map = {
        "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
        "gif": "image/gif", "webp": "image/webp",
        "mp4": "video/mp4", "avi": "video/avi", "mov": "video/quicktime",
        "mkv": "video/x-matroska", "webm": "video/webm"
    }
    mime = mime_map.get(ext, "application/octet-stream")
    return f"data:{mime};base64,{data}"


def extract_video_frames(file_path: str, fps: float = 1.0, max_frames: int = 8, max_size: int = 512) -> List[str]:
    """
    비디오에서 프레임을 추출하여 base64 이미지 리스트로 반환

    Args:
        file_path: 비디오 파일 경로
        fps: 초당 추출할 프레임 수
        max_frames: 최대 프레임 수
        max_size: 이미지 최대 크기 (긴 변 기준)

    Returns:
        base64 인코딩된 이미지 data URI 리스트
    """
    cap = cv2.VideoCapture(file_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_fps <= 0:
        cap.release()
        return []

    # 추출 간격 계산
    frame_interval = max(1, int(video_fps / fps))

    frames = []
    frame_idx = 0

    while len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # BGR to RGB, then encode
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        # 이미지 리사이즈 (토큰 수 절감)
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=80)
        base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        frames.append(f"data:image/jpeg;base64,{base64_data}")

        frame_idx += frame_interval

    cap.release()
    return frames


def send_chat_request(messages: List[Dict], stream: bool = False) -> Generator[str, None, None]:
    """Send chat completion to server."""
    url = f"{get_server_url()}/v1/chat/completions"
    
    payload = {
        "model": "glm-4.6v-flash",
        "messages": messages,
        "temperature": CONFIG["temperature"],
        "top_p": CONFIG["top_p"],
        "top_k": CONFIG["top_k"],
        "max_tokens": CONFIG["max_tokens"],
        "stream": stream
    }
    
    try:
        response = requests.post(url, json=payload, stream=stream, timeout=300)
        response.raise_for_status()
        
        if stream:
            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8") if isinstance(line, bytes) else line
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            parsed = json.loads(data)
                            if "choices" in parsed and parsed["choices"]:
                                content = parsed["choices"][0].get("delta", {}).get("content", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            pass
        else:
            data = response.json()
            if "choices" in data and data["choices"]:
                yield data["choices"][0]["message"].get("content", "")
                
    except requests.exceptions.RequestException as e:
        yield f"❌ Error: {str(e)}"


# ============================================================================
# Chat Handler
# ============================================================================

def respond_multimodal(
    message: Dict[str, Any],
    history: List[Tuple[str, str]],
) -> str:
    """
    Multimodal chat handler (with images/videos).
    Returns complete response at once (non-streaming).
    """
    # Build API messages (히스토리 무시 - 현재 메시지만 전송)
    api_messages = [{"role": "system", "content": CONFIG["system_prompt"]}]
    # history 파라미터는 무시됨 - 토큰 수 일정하게 유지
    
    # Build multimodal content
    content_parts = []
    
    # Process files
    files = message.get("files", [])
    for file_item in files:
        if file_item:
            # Gradio 6.0: file can be dict with 'path' key or just a string path
            if isinstance(file_item, dict):
                file_path = file_item.get("path", "")
            else:
                file_path = file_item
            
            if not file_path:
                continue
                
            data_uri = encode_file_base64(file_path)
            ext = file_path.lower().split(".")[-1]
            
            if ext in ["jpg", "jpeg", "png", "gif", "webp"]:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": data_uri}
                })
            elif ext in ["mp4", "avi", "mov", "mkv", "webm"]:
                # 프레임 추출 (video_url 대신 image_url 배열로 전송)
                frames = extract_video_frames(
                    file_path,
                    fps=CONFIG["video_fps"],
                    max_frames=CONFIG["max_frames"]
                )
                for frame_uri in frames:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": frame_uri}
                    })
    
    # Add text
    text = message.get("text", "")
    if text:
        content_parts.append({"type": "text", "text": text})
    
    # Create message
    if len(content_parts) == 1 and content_parts[0].get("type") == "text":
        api_messages.append({"role": "user", "content": text})
    else:
        api_messages.append({"role": "user", "content": content_parts})
    
    # Get full response (non-streaming)
    full_response = ""
    for chunk in send_chat_request(api_messages, stream=False):
        full_response += chunk
    
    return full_response if full_response else "⏳ No response received"


# ============================================================================
# Settings Callbacks
# ============================================================================

def update_config(host, port, system_prompt, temperature, top_p, top_k, max_tokens, video_fps, max_frames):
    CONFIG["host"] = host
    CONFIG["port"] = int(port)
    CONFIG["system_prompt"] = system_prompt
    CONFIG["temperature"] = temperature
    CONFIG["top_p"] = top_p
    CONFIG["top_k"] = top_k
    CONFIG["max_tokens"] = int(max_tokens)
    CONFIG["video_fps"] = video_fps
    CONFIG["max_frames"] = int(max_frames)
    return check_server_health()


# ============================================================================
# Main UI
# ============================================================================

with gr.Blocks(title=PAGE_TITLE) as demo:
    gr.Markdown(f"# 🤖 {PAGE_TITLE}")
    
    with gr.Row():
        with gr.Column(scale=3):
            # Use ChatInterface for automatic message handling
            chat = gr.ChatInterface(
                fn=respond_multimodal,
                multimodal=True,
                chatbot=gr.Chatbot(height=500),
                textbox=gr.MultimodalTextbox(
                    placeholder="Type a message or upload images/videos...",
                    file_types=["image", "video"],
                    file_count="multiple",
                ),
            )
        
        with gr.Column(scale=1):
            gr.Markdown("## ⚙️ Settings")
            
            with gr.Accordion("Server", open=True):
                host_input = gr.Textbox(label="Host", value=DEFAULT_HOST)
                port_input = gr.Number(label="Port", value=DEFAULT_PORT, precision=0)
                status_text = gr.Textbox(label="Status", value="⚪ Unknown", interactive=False)
                check_btn = gr.Button("🔄 Check Status")
            
            with gr.Accordion("System Prompt", open=False):
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    value=CONFIG["system_prompt"],
                    lines=3
                )
            
            with gr.Accordion("Parameters", open=True):
                temperature = gr.Slider(0.0, 2.0, value=0.8, step=0.1, label="Temperature")
                top_p = gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="Top P")
                top_k = gr.Slider(1, 100, value=2, step=1, label="Top K")
                max_tokens = gr.Number(value=4096, label="Max Tokens", precision=0)

            with gr.Accordion("Video Settings", open=True):
                video_fps = gr.Slider(0.5, 2.0, value=1.0, step=0.5, label="Video FPS")
                max_frames = gr.Slider(4, 16, value=8, step=2, label="Max Frames")
            
            apply_btn = gr.Button("Apply Settings", variant="primary")
    
    # Event handlers
    check_btn.click(
        fn=lambda h, p: update_config(h, p, CONFIG["system_prompt"],
                                       CONFIG["temperature"], CONFIG["top_p"],
                                       CONFIG["top_k"], CONFIG["max_tokens"],
                                       CONFIG["video_fps"], CONFIG["max_frames"]),
        inputs=[host_input, port_input],
        outputs=[status_text]
    )

    apply_btn.click(
        fn=update_config,
        inputs=[host_input, port_input, system_prompt, temperature, top_p, top_k, max_tokens, video_fps, max_frames],
        outputs=[status_text]
    )
    
    demo.load(fn=check_server_health, outputs=[status_text])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
