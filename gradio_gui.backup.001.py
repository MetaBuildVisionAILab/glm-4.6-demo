#!/usr/bin/env python3
"""
GLM-4.6V Inference GUI (Gradio 6.0)
A Gradio web interface for GLM-4.6V multimodal model inference.
"""

import base64
from datetime import datetime
import io
import json
import os
import requests
import time
import uuid
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
    "system_prompt": """
    당신은 도로 교통 영상을 분석하여 사고 발생 여부를 판별하는 전문가입니다.

## 입력

- 제공되는 이미지들은 영상에서 1초 간격으로 추출한 프레임입니다.
- 이미지 순서는 영상 재생 순서와 동일합니다. (1번 이미지 = 0초, 2번 이미지 = 1초, ...)

## 분석 절차

다음 단계를 순서대로 수행하세요:

1. **전체 흐름 파악**: 모든 프레임을 순서대로 확인하여 교통 상황의 변화를 파악하세요.
2. **이상 징후 탐지**: 차량의 급정거, 비정상적 정지, 차량 간 급접근, 긴급차량 출동 등을 찾으세요.
3. **인과관계 추론**: 후반부에 긴급차량이나 정체가 있다면, 앞선 프레임에서 원인을 역추적하세요.
4. **최종 판정**: 수집한 증거를 바탕으로 사고 여부를 판정하세요.

## 사고 판별 기준

### 사고로 판별하는 경우

- 충돌 장면이 직접 보이는 경우
- 충돌 직후 상태: 차량이 비정상적으로 밀착되어 정지 (범퍼 접촉)
- 사고 결과 상태: 차량 파손, 도로 위 파편, 비정상적 차량 각도
- 사고 후속 상황: 긴급차량(구급차 (119 차량), 경찰차) 출동, 도로 위 비상 정지

### 사고로 판별하지 않는 경우

- 충돌 없이 급정거만 한 경우
- 단순히 차간 거리가 좁은 정상 주행
- 일반적인 교통 정체 (사고 증거 없음)

## 중요

- 1fps 샘플링으로 인해 충돌 "순간"이 누락될 수 있습니다.
- 충돌 순간이 없더라도, 충돌 전후 상태를 통해 사고를 추론하세요.
- 긴급차량 출동은 사고의 강력한 증거입니다.

## 사고 유형

type 필드는 반드시 다음 값 중 하나여야 합니다:

- "차량간 사고"
- "차량 단독 사고"
- "차량-보행자 사고"
- "차량-자전거 사고"

## 신뢰도 기준

- "high": 충돌 장면 또는 충돌 직후 상태가 명확히 보임
- "medium": 충돌이 발생한 것으로 보이나 일부 가려지거나 간접 증거로 추론
- "low": 정황상 사고로 추정되나 직접적 증거 부족

## 응답 언어

- reasoning 필드: 영어로 작성
- 그 외 필드: 한국어로 작성

## 응답 형식

응답은 반드시 JSON 형식만 출력하세요. 추가 설명을 포함하지 마세요.

### 사고 발생 시

{
"accident": true,
"incidents": [
{
"type": "차량간 사고",
"frame": 5,
"confidence": "high",
"reasoning": "White sedan rear-ends the vehicle ahead in frame 5"
}
]
}

### 다중 사고 시
{
"accident": true,
"incidents": [
{
"type": "차량간 사고",
"frame": 3,
"confidence": "high",
"reasoning": "Two vehicles collide at intersection in frame 3"
},
{
"type": "차량-보행자 사고",
"frame": 3,
"confidence": "medium",
"reasoning": "Pedestrian struck by vehicle debris, partially obscured"
},
{
"type": "차량 단독 사고",
"frame": 7,
"confidence": "low",
"reasoning": "Vehicle appears to hit guardrail after losing control, impact not clearly visible"
}
]
}
### 사고 미발생 시
{
"accident": false,
"reasoning": "No collision observed throughout the footage"
}
### 판별 불가 시
{
"accident": null,
"reason": "unanalyzable",
"detail": "Video quality too low to identify vehicles or collisions"
}
    """,
    "temperature": 0.3,
    "top_p": 0.6,
    "top_k": 2,
    "max_tokens": 8192,
    "seconds_interval": 1.0,
    "max_frames": 300,
    "frame_start": 0.0,
    "frame_end": 1.0,
    "uneven_sampling": False,
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


# 프레임 저장 기본 경로
FRAME_SAVE_DIR = "/home/radar01/sdb1/jw/glm-4.6v/tmp"
RESULTS_DIR = "/home/radar01/sdb1/jw/glm-4.6v/results"
VIDEO_EXTS = {"mp4", "avi", "mov", "mkv", "webm"}


def extract_video_frames(
    file_path: str,
    seconds_interval: float = 1.0,
    max_frames: int = 8,
    max_size: int = 512,
    start_ratio: float = 0.0,
    end_ratio: float = 1.0,
    uneven_sampling: bool = False,
    return_dir: bool = False,
    save_dir: str | None = None,
) -> List[str] | tuple[List[str], str]:
    """
    비디오에서 프레임을 추출하여 base64 이미지 리스트로 반환하고,
    각 프레임을 /home/radar01/sdb1/jw/glm-4.6v/tmp/{UUID}/{UUID}_N.png 형태로 저장

    Args:
        file_path: 비디오 파일 경로
        seconds_interval: 추출 간격 (초 단위, 비디오 FPS에 맞게 자동 변환)
        max_frames: 최대 프레임 수
        max_size: 이미지 최대 크기 (긴 변 기준)
        start_ratio: 추출 시작 지점 (전체 구간 비율, 0.0~1.0)
        end_ratio: 추출 종료 지점 (전체 구간 비율, 0.0~1.0)
        uneven_sampling: 중앙 구간을 더 촘촘히 샘플링
        return_dir: 프레임 저장 디렉토리 반환 여부
        save_dir: 프레임 저장 디렉토리 지정 (None이면 기본 경로 사용)

    Returns:
        base64 인코딩된 이미지 data URI 리스트 (또는 (리스트, 저장 디렉토리))
    """
    cap = cv2.VideoCapture(file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0:
        cap.release()
        return []

    # FPS가 유효하지 않으면 기본값 30 사용
    if fps <= 0:
        fps = 30.0

    start_ratio = max(0.0, min(1.0, float(start_ratio)))
    end_ratio = max(start_ratio, min(1.0, float(end_ratio)))
    start_idx = int(total_frames * start_ratio)
    end_idx = int(total_frames * end_ratio)
    if end_idx <= start_idx:
        end_idx = min(total_frames, start_idx + 1)

    # 초 단위 간격을 프레임 간격으로 변환
    frame_interval = max(1, int(fps * seconds_interval))
    print(f"[Video Info] FPS: {fps:.2f}, Total frames: {total_frames}, Frame interval: {frame_interval}")

    candidate_indices = []
    if uneven_sampling:
        range_len = max(1, end_idx - start_idx)
        focus_start = start_idx + int(range_len * 0.4)
        focus_end = start_idx + int(range_len * 0.6)
        outer_interval = max(1, frame_interval * 2)
        inner_interval = max(1, frame_interval // 2)
        candidate_indices.extend(range(start_idx, focus_start, outer_interval))
        candidate_indices.extend(range(focus_start, focus_end, inner_interval))
        candidate_indices.extend(range(focus_end, end_idx, outer_interval))
        candidate_indices = sorted(set(candidate_indices))
    else:
        candidate_indices = list(range(start_idx, end_idx, frame_interval))

    if len(candidate_indices) > max_frames:
        mid = len(candidate_indices) // 2
        half = max_frames // 2
        start = max(0, mid - half)
        end = start + max_frames
        if end > len(candidate_indices):
            end = len(candidate_indices)
            start = max(0, end - max_frames)
        candidate_indices = candidate_indices[start:end]

    # 날짜/시간 + 비디오명 기반 저장 디렉토리 생성 (또는 지정 경로)
    if save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_base = os.path.splitext(os.path.basename(file_path))[0]
        save_dir = os.path.join(FRAME_SAVE_DIR, f"{timestamp}_{video_base}")
    os.makedirs(save_dir, exist_ok=True)

    frames = []
    frame_count = 0
    for frame_idx in candidate_indices:
        if len(frames) >= max_frames:
            break
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

        # 프레임을 PNG 파일로 저장
        frame_count += 1
        frame_filename = f"frame_{frame_count}.png"
        frame_path = os.path.join(save_dir, frame_filename)
        img.save(frame_path, format="PNG")
        print(f"[Frame saved] {frame_path}")

        # Base64 인코딩 (API 전송용)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=80)
        base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        frames.append(f"data:image/jpeg;base64,{base64_data}")

    cap.release()
    print(f"[Frames extracted] Total {len(frames)} frames saved to {save_dir}")
    if return_dir:
        return frames, save_dir
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


def build_text_override(file_path: str) -> str | None:
    filename_lower = os.path.basename(file_path).lower()
    prompt_dir = os.path.join(os.path.dirname(__file__), "prompt")
    keywords = [
        "stop", "pedestrian", "wrongway", "fire",
        "slow", "trafficjam", "fast", "drop",
    ]
    matched_keyword = next((k for k in keywords if k in filename_lower), None)
    if not matched_keyword:
        return None
    json_path = os.path.join(prompt_dir, f"{matched_keyword}.json")
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        if isinstance(json_data, str):
            return json_data
        if isinstance(json_data, list):
            return "\n".join(str(item) for item in json_data)
        return json.dumps(json_data, ensure_ascii=True)
    except (OSError, json.JSONDecodeError):
        return None


def analyze_video_file(file_path: str, save_dir: str | None = None) -> str:
    api_messages = [{"role": "system", "content": CONFIG["system_prompt"]}]
    content_parts = []

    data_uri = encode_file_base64(file_path)
    ext = file_path.lower().split(".")[-1]

    if ext in ["jpg", "jpeg", "png", "gif", "webp"]:
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": data_uri}
        })
    elif ext in VIDEO_EXTS:
        frames = extract_video_frames(
            file_path,
            seconds_interval=CONFIG["seconds_interval"],
            max_frames=CONFIG["max_frames"],
            start_ratio=CONFIG["frame_start"],
            end_ratio=CONFIG["frame_end"],
            uneven_sampling=CONFIG["uneven_sampling"],
            save_dir=save_dir,
        )
        for frame_uri in frames:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": frame_uri}
            })

    text_override = build_text_override(file_path)
    if text_override:
        content_parts.append({"type": "text", "text": text_override})

    if not content_parts:
        return "❌ No content to send"

    if len(content_parts) == 1 and content_parts[0].get("type") == "text":
        api_messages.append({"role": "user", "content": content_parts[0]["text"]})
    else:
        api_messages.append({"role": "user", "content": content_parts})

    full_response = ""
    for chunk in send_chat_request(api_messages, stream=False):
        full_response += chunk

    return full_response if full_response else "⏳ No response received"


def batch_process_path(input_path: str) -> tuple[str, str]:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    input_path = (input_path or "").strip().strip('"').strip("'")
    if not input_path:
        return "", "❌ Path is empty"

    input_path = os.path.expanduser(input_path)

    try:
        if os.path.isfile(input_path):
            ext = os.path.splitext(input_path)[1].lower().lstrip(".")
            if ext not in VIDEO_EXTS:
                return "", "❌ Not a supported video file"
            result = analyze_video_file(input_path)
            prompt_text = build_text_override(input_path) or ""
            return save_result_txt([input_path], result, system_prompt=CONFIG["system_prompt"], prompt_text=prompt_text)

        if os.path.isdir(input_path):
            videos = [
                os.path.join(input_path, name)
                for name in sorted(os.listdir(input_path))
                if os.path.splitext(name)[1].lower().lstrip(".") in VIDEO_EXTS
            ]
            if not videos:
                return "", "❌ No video files found in folder"

            folder_name = os.path.basename(os.path.normpath(input_path))
            batch_dir = make_batch_output_dir(input_path)
            os.makedirs(batch_dir, exist_ok=True)

            summaries = []
            for video in videos:
                video_base = os.path.splitext(os.path.basename(video))[0]
                frames_dir = os.path.join(batch_dir, video_base)
                result = analyze_video_file(video, save_dir=frames_dir)
                prompt_text = build_text_override(video) or ""
                summaries.append(os.path.basename(video) + "\n" + result + "\n")
                save_result_txt([video], result, base_dir=batch_dir, system_prompt=CONFIG["system_prompt"], prompt_text=prompt_text)

            summary_text = "\n".join(summaries).strip() + "\n"
            save_result_txt(videos, summary_text, folder_name=folder_name, base_dir=batch_dir, system_prompt=CONFIG["system_prompt"], prompt_text="BATCH_SUMMARY")

            summary_path = os.path.join(batch_dir, f"{folder_name}.txt")
            return summary_path, f"✅ Saved: {summary_path}"

        return "", f"❌ Path not found: {input_path}"
    except OSError as e:
        return "", f"❌ File error: {e}"
    except Exception as e:
        return "", f"❌ Error: {e}"


def load_text_file(path_str: str) -> str:
    path_str = (path_str or "").strip()
    if not path_str:
        return "❌ Path is empty"
    if not os.path.exists(path_str):
        return "❌ File not found"
    try:
        with open(path_str, "r", encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        return f"❌ Error reading file: {e}"


def load_text_file_with_path(path_str: str) -> tuple[str, str]:
    return path_str, load_text_file(path_str)


def extract_path_from_file(file_obj) -> str:
    if not file_obj:
        return ""
    if isinstance(file_obj, dict):
        return file_obj.get("path", "")
    return str(file_obj)


def make_batch_output_dir(batch_folder: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = os.path.basename(os.path.normpath(batch_folder))
    return os.path.join(FRAME_SAVE_DIR, f"{timestamp}_{folder_name}")


def save_result_txt(
    video_paths: list[str],
    result_text: str,
    folder_name: str | None = None,
    base_dir: str | None = None,
    system_prompt: str | None = None,
    prompt_text: str | None = None,
) -> tuple[str, str]:
    if not video_paths:
        return "", "❌ No video paths"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    payload = {
        "systemPrompt": system_prompt or "",
        "prompt": prompt_text or "",
        "response": result_text,
    }
    content = json.dumps(payload, ensure_ascii=False)

    # Folder summary mode
    if folder_name is not None:
        target_dir = base_dir or os.path.dirname(video_paths[0])
        out_path = os.path.join(target_dir, f"{folder_name}.txt")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(content)
            return out_path, f"✅ Saved: {out_path}"
        except OSError:
            fallback = os.path.join(RESULTS_DIR, f"{folder_name}.txt")
            with open(fallback, "w", encoding="utf-8") as f:
                f.write(content)
            return fallback, f"✅ Saved (fallback): {fallback}"

    # Single/individual file mode
    last_path = ""
    for vp in video_paths:
        base_name = os.path.splitext(os.path.basename(vp))[0]
        target_dir = base_dir or os.path.dirname(vp)
        out_path = os.path.join(target_dir, f"{base_name}.txt")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(content)
            last_path = out_path
        except OSError:
            fallback = os.path.join(RESULTS_DIR, f"{base_name}.txt")
            with open(fallback, "w", encoding="utf-8") as f:
                f.write(content)
            last_path = fallback
    return last_path, f"✅ Saved: {last_path}" if last_path else "❌ Save failed"


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
    start_time = time.time()
    # Build API messages (히스토리 무시 - 현재 메시지만 전송)
    api_messages = [{"role": "system", "content": CONFIG["system_prompt"]}]
    # history 파라미터는 무시됨 - 토큰 수 일정하게 유지
    
    # Build multimodal content
    content_parts = []
    
    # Process files
    uploaded_filenames = []
    video_paths = []
    frame_dirs = []
    json_text_override = None
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

            uploaded_filenames.append(os.path.basename(file_path))
            if json_text_override is None:
                json_text_override = build_text_override(file_path)
                
            data_uri = encode_file_base64(file_path)
            ext = file_path.lower().split(".")[-1]
            
            if ext in ["jpg", "jpeg", "png", "gif", "webp"]:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": data_uri}
                })
            elif ext in ["mp4", "avi", "mov", "mkv", "webm"]:
                video_paths.append(file_path)
                # 프레임 추출 (video_url 대신 image_url 배열로 전송)
                frames, save_dir = extract_video_frames(
                    file_path,
                    seconds_interval=CONFIG["seconds_interval"],
                    max_frames=CONFIG["max_frames"],
                    start_ratio=CONFIG["frame_start"],
                    end_ratio=CONFIG["frame_end"],
                    uneven_sampling=CONFIG["uneven_sampling"],
                    return_dir=True,
                )
                if save_dir:
                    frame_dirs.append(save_dir)
                for frame_uri in frames:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": frame_uri}
                    })
    
    # Add text
    text = json_text_override if json_text_override is not None else message.get("text", "")
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

    elapsed = time.time() - start_time
    elapsed_text = f"Processing time: {elapsed:.2f}s"
    if full_response:
        full_response = full_response + "\n\n" + elapsed_text
    else:
        full_response = elapsed_text

    if video_paths and full_response:
        if frame_dirs:
            save_result_txt(video_paths, full_response, base_dir=frame_dirs[0], system_prompt=CONFIG["system_prompt"], prompt_text=text)
        else:
            save_result_txt(video_paths, full_response, system_prompt=CONFIG["system_prompt"], prompt_text=text)
    
    if uploaded_filenames:
        filenames_text = ", ".join(uploaded_filenames)
        if full_response:
            return f"Uploaded file(s): {filenames_text}\n\n{full_response}"
        return f"Uploaded file(s): {filenames_text}\n\n⏳ No response received"

    return full_response if full_response else "⏳ No response received"


# ============================================================================
# Settings Callbacks
# ============================================================================

def update_config(host, port, system_prompt, temperature, top_p, top_k, max_tokens, seconds_interval, max_frames, frame_start, frame_end, uneven_sampling):
    CONFIG["host"] = host
    CONFIG["port"] = int(port)
    CONFIG["system_prompt"] = system_prompt
    CONFIG["temperature"] = temperature
    CONFIG["top_p"] = top_p
    CONFIG["top_k"] = top_k
    CONFIG["max_tokens"] = int(max_tokens)
    CONFIG["seconds_interval"] = float(seconds_interval)
    CONFIG["max_frames"] = int(max_frames)
    CONFIG["frame_start"] = float(frame_start)
    CONFIG["frame_end"] = float(frame_end)
    CONFIG["uneven_sampling"] = bool(uneven_sampling)
    return check_server_health()


# ============================================================================
# Main UI
# ============================================================================

with gr.Blocks(title=PAGE_TITLE) as demo:
    gr.Markdown(f"# 🤖 {PAGE_TITLE}")
    gr.Markdown(
        "## 사용 가이드\n"
        "- Check Status를 눌러 `⏳ Model is loading...` 이 작동하면 됩니다.\n"
        "- 검색창에 비디오를 넣고 실행하세요.\n"
        "- System Prompt, Parameters, Video Settings는 건드리지 않아도 됩니다.\n"
        "- 업로드한 비디오 파일 이름이 출력됩니다.\n"
    )
    
    with gr.Row():
        with gr.Column(scale=3):
            # Use ChatInterface for automatic message handling
            chat = gr.ChatInterface(
                fn=respond_multimodal,
                multimodal=True,
                chatbot=gr.Chatbot(height=500),
                textbox=gr.MultimodalTextbox(
                    placeholder="Type a message or upload images/videos...",
                    file_types=["image", ".mp4", ".avi", ".mov", ".mkv", ".webm"],
                    file_count="multiple",
                ),
            )

            with gr.Accordion("Batch & Viewer", open=False):
                gr.Markdown("### 📁 Batch Process")
                batch_path = gr.Textbox(label="File or Folder Path", placeholder="/path/to/video_or_folder")
                batch_file = gr.File(label="Select Video File", file_types=[".mp4", ".avi", ".mov", ".mkv", ".webm"], type="filepath")
                use_file_btn = gr.Button("Use Selected File")
                batch_run = gr.Button("Run Batch")
                batch_out_path = gr.Textbox(label="Result TXT Path", interactive=False)
                batch_status = gr.Textbox(label="Batch Status", interactive=False)

                gr.Markdown("### 📄 Result Viewer")
                viewer_path = gr.Textbox(label="TXT Path", placeholder="/path/to/result.txt")
                viewer_btn = gr.Button("Load Result")
                viewer_content = gr.Textbox(label="TXT Content", lines=12)
        
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
            
            with gr.Accordion("Parameters", open=False):
                temperature = gr.Slider(0.0, 2.0, value=0.6, step=0.1, label="Temperature")
                top_p = gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="Top P")
                top_k = gr.Slider(1, 100, value=2, step=1, label="Top K")
                max_tokens = gr.Number(value=4096, label="Max Tokens", precision=0)

            with gr.Accordion("Video Settings", open=False):
                seconds_interval = gr.Slider(0.1, 5.0, value=0.3, step=0.1, label="Seconds Interval (초 단위)")
                max_frames = gr.Slider(4, 900, value=100, step=2, label="Max Frames")
                frame_start = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Frame Start Ratio")
                frame_end = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Frame End Ratio")
                uneven_sampling = gr.Checkbox(value=False, label="Uneven Sampling (center-weighted)")
            
            apply_btn = gr.Button("Apply Settings", variant="primary")
    
    # Event handlers
    use_file_btn.click(
        fn=extract_path_from_file,
        inputs=[batch_file],
        outputs=[batch_path]
    )

    batch_run.click(
        fn=batch_process_path,
        inputs=[batch_path],
        outputs=[batch_out_path, batch_status]
    )

    viewer_btn.click(
        fn=load_text_file,
        inputs=[viewer_path],
        outputs=[viewer_content]
    )

    batch_out_path.change(
        fn=load_text_file_with_path,
        inputs=[batch_out_path],
        outputs=[viewer_path, viewer_content]
    )

    check_btn.click(
        fn=lambda h, p: update_config(h, p, CONFIG["system_prompt"],
                                       CONFIG["temperature"], CONFIG["top_p"],
                                       CONFIG["top_k"], CONFIG["max_tokens"],
                                       CONFIG["seconds_interval"], CONFIG["max_frames"],
                                       CONFIG["frame_start"], CONFIG["frame_end"],
                                       CONFIG["uneven_sampling"]),
        inputs=[host_input, port_input],
        outputs=[status_text]
    )

    apply_btn.click(
        fn=update_config,
        inputs=[host_input, port_input, system_prompt, temperature, top_p, top_k, max_tokens, seconds_interval, max_frames, frame_start, frame_end, uneven_sampling],
        outputs=[status_text]
    )
    
    demo.load(fn=check_server_health, outputs=[status_text])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7863)
