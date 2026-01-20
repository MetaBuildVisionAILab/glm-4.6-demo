#!/usr/bin/env python3
"""
GLM-4.6V-Flash Inference GUI
A Streamlit web interface for GLM-4.6V multimodal model inference.
"""

import base64
import io
import time
from typing import List, Dict, Any, Optional, Generator

import streamlit as st
import requests
from PIL import Image

# ============================================================================
# Configuration & Constants
# ============================================================================

PAGE_TITLE = "GLM-4.6V-Flash Inference GUI"
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8000
DEFAULT_MODEL = "glm-4.6v-flash"

# Default parameters from CLAUDE.md
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.6
DEFAULT_TOP_K = 2
DEFAULT_MAX_TOKENS = 4096
DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."

# ============================================================================
# Session State Management
# ============================================================================

def init_session_state() -> None:
    """Initialize all session state variables."""
    # Server configuration
    if "server_host" not in st.session_state:
        st.session_state.server_host = DEFAULT_HOST
    if "server_port" not in st.session_state:
        st.session_state.server_port = DEFAULT_PORT

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # System prompt
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT

    # Media storage
    if "uploaded_images" not in st.session_state:
        st.session_state.uploaded_images = []
    if "uploaded_video" not in st.session_state:
        st.session_state.uploaded_video = None

    # Parameters
    if "temperature" not in st.session_state:
        st.session_state.temperature = DEFAULT_TEMPERATURE
    if "top_p" not in st.session_state:
        st.session_state.top_p = DEFAULT_TOP_P
    if "top_k" not in st.session_state:
        st.session_state.top_k = DEFAULT_TOP_K
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = DEFAULT_MAX_TOKENS
    if "use_streaming" not in st.session_state:
        st.session_state.use_streaming = True

    # Server status
    if "server_status" not in st.session_state:
        st.session_state.server_status = "unknown"
    if "slots_info" not in st.session_state:
        st.session_state.slots_info = []

    # Processing state
    if "is_generating" not in st.session_state:
        st.session_state.is_generating = False
    if "user_prompt" not in st.session_state:
        st.session_state.user_prompt = ""


# ============================================================================
# API Client Functions
# ============================================================================

def get_server_url() -> str:
    """Get the server URL from session state."""
    return f"http://{st.session_state.server_host}:{st.session_state.server_port}"


def check_server_health() -> tuple[bool, str]:
    """
    Check if the server is running and model is loaded.

    Returns:
        tuple: (is_ready, status_message)
    """
    try:
        response = requests.get(f"{get_server_url()}/health", timeout=5)
        if response.status_code == 200:
            return True, "Model loaded and ready"
        elif response.status_code == 503:
            return False, "Model is loading..."
        else:
            return False, f"Unexpected status: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to server"
    except requests.exceptions.Timeout:
        return False, "Connection timeout"
    except Exception as e:
        return False, f"Error: {str(e)}"


def get_server_slots() -> List[Dict[str, Any]]:
    """
    Get information about server processing slots.

    Returns:
        List of slot information dictionaries
    """
    try:
        response = requests.get(f"{get_server_url()}/slots", timeout=5)
        if response.status_code == 200:
            return response.json()
        return []
    except Exception:
        return []


def send_chat_completion(
    messages: List[Dict[str, Any]],
    stream: bool = True
) -> Generator[Dict[str, Any], None, None]:
    """
    Send chat completion request to the server.

    Args:
        messages: List of message dictionaries
        stream: Whether to use streaming mode

    Yields:
        Response chunks (if streaming) or full response
    """
    url = f"{get_server_url()}/v1/chat/completions"

    payload = {
        "model": DEFAULT_MODEL,
        "messages": messages,
        "temperature": st.session_state.temperature,
        "top_p": st.session_state.top_p,
        "top_k": st.session_state.top_k,
        "max_tokens": st.session_state.max_tokens,
        "stream": stream
    }

    try:
        # For non-streaming, don't use stream=True
        response = requests.post(
            url,
            json=payload,
            stream=stream,
            timeout=120
        )
        response.raise_for_status()

        if stream:
            line_count = 0
            for line in response.iter_lines(decode_unicode=False):
                if line:
                    line = line.decode('utf-8') if isinstance(line, bytes) else line
                    if line.startswith('data: '):
                        data = line[6:]  # Remove 'data: ' prefix
                        if data == '[DONE]':
                            break
                        line_count += 1
                        yield {"raw": data, "line_num": line_count}
        else:
            yield {"full": response.json()}

    except requests.exceptions.RequestException as e:
        yield {"error": str(e)}


# ============================================================================
# Media Processing Functions
# ============================================================================

def encode_image_base64(image_file) -> str:
    """
    Convert uploaded image file to base64 data URI.

    Args:
        image_file: Streamlit UploadedFile object

    Returns:
        str: Data URI in format "data:image/<type>;base64,..."
    """
    image_bytes = image_file.read()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    image_type = image_file.type.split('/')[1]  # e.g., "jpeg", "png"
    return f"data:image/{image_type};base64,{image_base64}"


def encode_video_base64(video_file) -> str:
    """
    Convert uploaded video file to base64 data URI.

    Args:
        video_file: Streamlit UploadedFile object

    Returns:
        str: Data URI in format "data:video/<type>;base64,..."
    """
    # Seek to start in case file was already read
    video_file.seek(0)
    video_bytes = video_file.read()
    video_base64 = base64.b64encode(video_bytes).decode('utf-8')
    video_type = video_file.type  # e.g., "video/mp4"
    print(f"[DEBUG GUI] Encoded video: {len(video_bytes)} bytes, type: {video_type}")
    return f"data:{video_type};base64,{video_base64}"


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


# ============================================================================
# Message Formatting
# ============================================================================

def build_message_payload(user_prompt: str) -> List[Dict[str, Any]]:
    """
    Build the message payload for API request.

    Args:
        user_prompt: User's text prompt

    Returns:
        List of message dictionaries
    """
    messages = [
        {"role": "system", "content": st.session_state.system_prompt}
    ]

    # Debug: print session state
    print(f"[DEBUG GUI] uploaded_video: {st.session_state.uploaded_video is not None}")
    if st.session_state.uploaded_video:
        print(f"[DEBUG GUI]   video name: {st.session_state.uploaded_video.get('name')}")
        data_uri = st.session_state.uploaded_video.get('data_uri', '')
        print(f"[DEBUG GUI]   data_uri length: {len(data_uri)}")

    # Build user message content
    content_parts = []

    # Add images first (if any)
    if st.session_state.uploaded_images:
        for img in st.session_state.uploaded_images:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": img["data_uri"]}
            })

    # Add video (if any)
    if st.session_state.uploaded_video:
        content_parts.append({
            "type": "video_url",
            "video_url": {"url": st.session_state.uploaded_video["data_uri"]}
        })

    # Add text prompt
    content_parts.append({
        "type": "text",
        "text": user_prompt
    })

    print(f"[DEBUG GUI] content_parts count: {len(content_parts)}")
    
    messages.append({
        "role": "user",
        "content": content_parts if len(content_parts) > 1 else user_prompt
    })

    return messages


# ============================================================================
# UI Components
# ============================================================================

def render_sidebar() -> None:
    """Render the sidebar with server config, parameters, and status."""
    st.sidebar.title("⚙️ Configuration")

    # Server Configuration
    st.sidebar.subheader("Server Connection")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.session_state.server_host = st.text_input(
            "Host", st.session_state.server_host, label_visibility="collapsed"
        )
    with col2:
        st.session_state.server_port = st.number_input(
            "Port", st.session_state.server_port, label_visibility="collapsed"
        )

    # Health check
    if st.sidebar.button("🔄 Check Status", use_container_width=True):
        is_ready, message = check_server_health()
        st.session_state.server_status = "ok" if is_ready else "error"
        if is_ready:
            st.sidebar.success(f"✅ {message}")
            st.session_state.slots_info = get_server_slots()
        else:
            st.sidebar.warning(f"⚠️ {message}")

    # Display server status
    if st.session_state.server_status == "ok":
        st.sidebar.info("🟢 Server Ready")
    elif st.session_state.server_status == "error":
        st.sidebar.error("🔴 Server Error")
    else:
        st.sidebar.info("⚪ Status Unknown")

    # Slot information
    if st.session_state.slots_info:
        st.sidebar.subheader("Processing Slots")
        active_slots = sum(1 for slot in st.session_state.slots_info if slot.get("is_processing", False))
        total_slots = len(st.session_state.slots_info)
        st.sidebar.metric("Active Slots", f"{active_slots}/{total_slots}")

    # Parameters
    st.sidebar.subheader("Generation Parameters")
    st.session_state.temperature = st.sidebar.slider(
        "Temperature", 0.0, 2.0, st.session_state.temperature, 0.1
    )
    st.session_state.top_p = st.sidebar.slider(
        "Top P", 0.0, 1.0, st.session_state.top_p, 0.05
    )
    st.session_state.top_k = st.sidebar.slider(
        "Top K", 1, 100, int(st.session_state.top_k), 1
    )
    st.session_state.max_tokens = st.sidebar.number_input(
        "Max Tokens", 1, 32768, int(st.session_state.max_tokens), 128
    )
    st.session_state.use_streaming = st.sidebar.checkbox(
        "Enable Streaming", st.session_state.use_streaming
    )


def render_system_prompt() -> None:
    """Render the system prompt input area."""
    st.session_state.system_prompt = st.text_area(
        "System Prompt",
        st.session_state.system_prompt,
        height=100,
        help="Sets the behavior and personality of the AI assistant"
    )


def render_media_upload() -> None:
    """Render media upload section for images and videos."""
    st.subheader("📎 Media Upload")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Images**")
        uploaded_images = st.file_uploader(
            "Upload images",
            type=['png', 'jpg', 'jpeg', 'gif', 'webp'],
            accept_multiple_files=True,
            key="image_uploader",
            label_visibility="collapsed"
        )

        # Update session state with new images
        if uploaded_images:
            st.session_state.uploaded_images = []
            for img in uploaded_images:
                st.session_state.uploaded_images.append({
                    "name": img.name,
                    "size": img.size,
                    "data_uri": encode_image_base64(img)
                })
            # Display previews
            for img in st.session_state.uploaded_images:
                st.caption(f"📷 {img['name']} ({format_file_size(img['size'])})")
        elif not st.session_state.uploaded_images:
            st.session_state.uploaded_images = []

    with col2:
        st.write("**Video**")
        uploaded_video = st.file_uploader(
            "Upload video",
            type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
            key="video_uploader",
            label_visibility="collapsed"
        )

        # Update session state ONLY when new video is uploaded
        if uploaded_video:
            # Check if this is a new upload (different name or no previous video)
            if (st.session_state.uploaded_video is None or 
                st.session_state.uploaded_video.get("name") != uploaded_video.name):
                st.session_state.uploaded_video = {
                    "name": uploaded_video.name,
                    "size": uploaded_video.size,
                    "data_uri": encode_video_base64(uploaded_video)
                }
        
        # Display video info from session state (persists across reruns)
        if st.session_state.uploaded_video:
            st.caption(f"🎥 {st.session_state.uploaded_video['name']} ({format_file_size(st.session_state.uploaded_video['size'])})")


def render_chat_interface() -> None:
    """Render the main chat interface."""
    st.subheader("💬 Chat")

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Display current media in chat (if any)
    if st.session_state.uploaded_images or st.session_state.uploaded_video:
        with st.chat_message("user"):
            for img in st.session_state.uploaded_images:
                st.image(f"data:image/png;base64,{img['data_uri'].split(',')[1]}", width=200)
            if st.session_state.uploaded_video:
                st.info(f"🎥 Video: {st.session_state.uploaded_video['name']}")

    # Chat input for user prompt
    if prompt := st.chat_input("Enter your message...", disabled=st.session_state.is_generating):
        st.session_state.user_prompt = prompt
        st.session_state.is_generating = True
        st.rerun()


def render_action_buttons() -> None:
    """Render action buttons for clearing media and resetting chat."""
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🗑️ Clear Media", use_container_width=True):
            st.session_state.uploaded_images = []
            st.session_state.uploaded_video = None
            st.rerun()

    with col2:
        if st.button("♻️ Reset Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# ============================================================================
# Generation Handler
# ============================================================================

def handle_generation() -> None:
    """Handle the generation process."""
    if not st.session_state.is_generating:
        return

    # Check server status first
    is_ready, status_msg = check_server_health()
    if not is_ready:
        st.error(f"❌ Server not ready: {status_msg}")
        st.info("💡 Make sure llama-server is running: `./scripts/run_server.sh`")
        st.session_state.is_generating = False
        st.rerun()
        return

    # Get user prompt from session state
    user_prompt = st.session_state.get("user_prompt", "")

    if not user_prompt:
        st.session_state.is_generating = False
        st.rerun()
        return

    # Build message payload
    messages = build_message_payload(user_prompt)

    # Debug log
    st.info(f"🔍 Sending request with {len(messages)} messages")

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_prompt)
        for img in st.session_state.uploaded_images:
            st.image(f"data:image/png;base64,{img['data_uri'].split(',')[1]}", width=200)
        if st.session_state.uploaded_video:
            st.info(f"🎥 Video: {st.session_state.uploaded_video['name']}")

    # Add to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Generate response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        with st.spinner("Generating..."):
            try:
                chunk_count = 0
                if st.session_state.use_streaming:
                    # Streaming mode
                    for chunk in send_chat_completion(messages, stream=True):
                        if "error" in chunk:
                            st.error(f"❌ Error: {chunk['error']}")
                            break

                        chunk_count += 1

                        try:
                            import json
                            data = json.loads(chunk["raw"])
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                # GLM-4.6V uses both reasoning_content and content
                                content = delta.get("content", "")
                                reasoning = delta.get("reasoning_content", "")
                                full_response += content + reasoning
                                response_placeholder.markdown(full_response + "▌")
                        except json.JSONDecodeError as e:
                            st.warning(f"JSON decode error at chunk {chunk_count}: {e}")
                            continue

                    response_placeholder.markdown(full_response)
                    st.info(f"✅ Received {chunk_count} chunks")
                else:
                    # Non-streaming mode
                    for chunk in send_chat_completion(messages, stream=False):
                        if "error" in chunk:
                            st.error(f"❌ Error: {chunk['error']}")
                            break

                        if "full" in chunk:
                            data = chunk["full"]
                            if "choices" in data and len(data["choices"]) > 0:
                                message = data["choices"][0]["message"]
                                # GLM-4.6V uses both reasoning_content and content
                                content = message.get("content", "")
                                reasoning = message.get("reasoning_content", "")
                                full_response = reasoning + content
                                response_placeholder.markdown(full_response)

                # Add to chat history
                if full_response:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": full_response
                    })

            except Exception as e:
                st.error(f"❌ Generation error: {str(e)}")

    # Reset state
    st.session_state.is_generating = False
    st.session_state.user_prompt = ""
    st.rerun()


# ============================================================================
# Main Application
# ============================================================================

def main() -> None:
    """Main application entry point."""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🤖 GLM-4.6V-Flash Inference GUI")

    # Initialize session state
    init_session_state()

    # Render sidebar
    render_sidebar()

    # Main content area
    col1, col2 = st.columns([3, 2])

    with col2:
        # Media upload - MUST RUN FIRST to populate session state
        render_media_upload()

    with col1:
        # System prompt
        render_system_prompt()

        # Chat interface
        render_chat_interface()

        # Action buttons
        render_action_buttons()

    # Handle generation if triggered
    if st.session_state.is_generating:
        handle_generation()


if __name__ == "__main__":
    main()
