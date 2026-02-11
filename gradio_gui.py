#최소프레임 추가
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
import uuid
import time
from typing import Generator, List, Dict, Any, Tuple

import cv2
import gradio as gr
from PIL import Image

# ============================================================================
# Configuration
# ============================================================================

SAVE_PNG = True  # Extracted frames will be saved as PNG files

PAGE_TITLE = "GLM-4.6V Inference GUI(10.1.1.60)"
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8000

LLAMA_SERVERS = {
    "V100 (10.1.1.60)": {"host": "10.1.1.60", "port": 8000},
    # "RTX4090 (10.1.1.67)": {"host": "10.1.1.67", "port": 8000},
    # "RTX3090 (172.28.5.247)": {"host": "172.28.5.247", "port": 8000},
}

# Global config (mutable via UI)
CONFIG = {
    "server_key": "V100 (10.1.1.60)",

    "system_prompt": """
    You are an expert in analyzing road CCTV footage to determine whether a traffic accident has occurred.

## Input Format
Frames extracted from the video are provided **in chronological order**. The first image is the earliest point in time, and the last image is the latest.

## Analysis Method
1. **First, scan through all frames in order** to understand the overall flow of the situation.
2. **Track each vehicle individually** to check for continuity in their movement.
3. **If any anomaly is detected**, focus your analysis on the frames immediately before and after.
4. Look for the following **transition points**:
   - Normal driving → Abnormal movement
   - Moving → Sudden stop
   - Maintaining distance between vehicles → Contact/Collision

## Analysis Criteria
Evaluate each of the following 6 categories by checking their respective sub-items:

### 1. Vehicle Stop Pattern
Mark "Yes" if any of the following apply:
**Abnormal stop in normal traffic flow**: Is there a stationary vehicle among moving vehicles?
**Shoulder stop**: Is there a vehicle stopped on the shoulder while other vehicles are driving?
**Single lane congestion**: Is one lane stopped while adjacent lanes are flowing normally?
**Stopped in the middle of an intersection**: Is a vehicle stopped in the center of an intersection, blocking traffic flow?

### 2. Hazard Light Signals
Mark "Yes" if any of the following apply:
**Rear hazard lights flashing**: Are the rear hazard lights of a vehicle blinking?
**Front hazard lights flashing**: Are the front hazard lights of a vehicle blinking?
**Multiple vehicles with hazard lights on simultaneously**: Are several vehicles flashing their hazard lights at the same time?
**Brake lights staying on**: Are one or more vehicles' brake lights continuously illuminated without moving?

### 3. Physical Collision / Damage Evidence
Mark "Yes" if any of the following apply:
**Vehicle-to-vehicle collision**: Are there signs of collision (rear-end or otherwise) between vehicles?
**Collision with guardrail/barrier**: Are there signs of a vehicle hitting a guardrail or traffic barrier?
**Vehicle rollover**: Is there an overturned or flipped vehicle?
**Debris and fragments**: Are vehicle fragments or debris scattered on the road?
**Vehicle contact**: Are two vehicles touching or overlapping each other?

### 4. Presence of People
Mark "Yes" if the following applies:
**Person near stopped vehicle**: Is there a person (pedestrian or driver) standing or moving near a stopped vehicle?

### 5. Abnormal Vehicle Movement (Changes Between Frames)
Mark "Yes" if any of the following apply:
**Sudden direction change**: Is a vehicle rotating or turning at an abnormal angle?
**Trajectory discontinuity**: Is a vehicle suddenly deviating from its normal driving path?
**Sudden speed change**: Has a moving vehicle suddenly stopped, or is there a drastic change in position compared to the previous frame?
**Rapid decrease in distance between vehicles**: Is the distance between two vehicles rapidly decreasing across frames?
**Loss of control**: Does a vehicle appear to be spinning, skidding, or moving sideways?

### 6. Collision Moment Signs
Mark "Yes" if any of the following apply:
**Collision flash/spark**: Is there a bright light or spark occurring at the point of contact between vehicles (especially visible at night)?
**Vehicle posture change**: Is a vehicle tilting, lifting, or rotating abnormally?
**Spin/Rotation**: Is a vehicle spinning in place or appears to have lost control?
**Sudden trajectory intersection**: Are two vehicles' paths intersecting at the same point simultaneously?

## Final Judgment Criteria (accident)
Determine whether an accident has occurred based on the following rules. Mark "Yes" if **any one** of these conditions is met:

1. **"Physical Collision / Damage Evidence" is "Yes"** → Accident confirmed
2. **"Collision Moment Signs" is "Yes"** → Accident confirmed
3. **"Abnormal Vehicle Movement" is "Yes"** AND **"Vehicle Stop Pattern" is "Yes"** → Accident highly likely
4. **"Vehicle Stop Pattern" is "Yes"** AND **two or more other categories are also "Yes"** → Accident highly likely

If only "Hazard Light Signals" is "Yes" alone → Further verification needed (may be simple congestion)

## Response Format
Respond only in the JSON format below. Do not include any other explanation.
json
{
  "vehicle_stop_pattern": "Yes or No",
  "hazard_light_signals": "Yes or No",
  "physical_collision": "Yes or No",
  "presence_of_people": "Yes or No",
  "abnormal_vehicle_movement": "Yes or No",
  "collision_moment_signs": "Yes or No",
  "accident": "Yes or No"
}
    """,
    "temperature": 0.3,
    "top_p": 0.6,
    "top_k": 2,
    "max_tokens": 8192,
    "seconds_interval": 0.3,
    "min_frames": 10,
    "max_frames": 300,
    "frame_start": 0.0,
    "frame_end": 1.0,
    "uneven_sampling": False,
    "resize_ratio": 1.0,
    "use_filename_prompt": True,
    "system_prompts": [],
}

# Initialize system prompt tabs


def load_system_prompts() -> list[str]:
    prompts = []
    for name in SYSTEM_PROMPT_FILES:
        path = os.path.join(SYSTEM_PROMPT_DIR, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                prompts.append(f.read())
        except FileNotFoundError:
            prompts.append("")
    return prompts


def save_system_prompt_file(index: int, prompt_text: str) -> None:
    if index < 0 or index >= len(SYSTEM_PROMPT_FILES):
        return
    os.makedirs(SYSTEM_PROMPT_DIR, exist_ok=True)
    path = os.path.join(SYSTEM_PROMPT_DIR, SYSTEM_PROMPT_FILES[index])
    with open(path, "w", encoding="utf-8") as f:
        f.write(prompt_text)


# ============================================================================
# API Functions
# ============================================================================

def get_server_url() -> str:
    server = LLAMA_SERVERS[CONFIG["server_key"]]
    return f"http://{server['host']}:{server['port']}"



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
SYSTEM_PROMPT_DIR = "/home/radar01/sdb1/jw/glm-4.6v/system_prompt"
SYSTEM_PROMPT_FILES = ["sp1.txt", "sp2.txt", "sp3.txt", "sp4.txt", "sp5.txt"]
CONFIG["system_prompts"] = load_system_prompts()
CONFIG["system_prompt"] = CONFIG["system_prompts"][0] if CONFIG["system_prompts"] else CONFIG["system_prompt"]
VIDEO_EXTS = {"mp4", "avi", "mov", "mkv", "webm"}


def extract_video_frames(
    file_path: str,
    seconds_interval: float = 1.0,
    min_frames: int = 0,
    max_frames: int = 8,
    max_size: int = 512,
    start_ratio: float = 0.0,
    end_ratio: float = 1.0,
    uneven_sampling: bool = False,
    return_dir: bool = False,
    resize_ratio: float = 1.0,
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
        resize_ratio: 0.1~1.0 비율로 해상도 축소

    Returns:
        base64 인코딩된 이미지 data URI 리스트 (또는 (리스트, 저장 디렉토리))
    """
    
    t_open = 0.0
    t_read = 0.0
    t_resize = 0.0
    t_png = 0.0
    t_b64 = 0.0
    t_total_start = time.time()

    t0 = time.time()
    cap = cv2.VideoCapture(file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    t_open += time.time() - t0

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

    # 최소 프레임
    total_available = max(0, end_idx - start_idx)

    if min_frames > 0 and len(candidate_indices) < min_frames and total_available > len(candidate_indices):
        target = min(min_frames, total_available)

        if target > 1:
            step = (end_idx - start_idx - 1) / (target - 1)
            candidate_indices = [
                int(start_idx + i * step) for i in range(target)
            ]
        else:
            candidate_indices = [start_idx]

        candidate_indices = sorted(set(candidate_indices))
    # --------------------------------

    if len(candidate_indices) > max_frames:
        mid = len(candidate_indices) // 2
        half = max_frames // 2
        start = max(0, mid - half)
        end = start + max_frames
        if end > len(candidate_indices):
            end = len(candidate_indices)
            start = max(0, end - max_frames)
        candidate_indices = candidate_indices[start:end]

    # 날짜/시간 + 비디오명 기반 저장 디렉토리 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_base = os.path.splitext(os.path.basename(file_path))[0]
    save_dir = os.path.join(FRAME_SAVE_DIR, f"{timestamp}_{video_base}")
    os.makedirs(save_dir, exist_ok=True)

    frames = []
    frame_count = 0
    for frame_idx in candidate_indices:
        if len(frames) >= max_frames:
            break
        t1 = time.time()
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        t_read += time.time() - t1
        if not ret:
            break

        # BGR to RGB, then encode
        t2 = time.time()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        w, h = img.size
        ratio = max(0.1, min(1.0, float(resize_ratio)))
        new_w, new_h = int(w * ratio), int(h * ratio)
        if max_size > 0 and max(new_w, new_h) > max_size:
            scale = max_size / max(new_w, new_h)
            new_w, new_h = int(new_w * scale), int(new_h * scale)
        if new_w < 1:
            new_w = 1
        if new_h < 1:
            new_h = 1
        if new_w != w or new_h != h:
            img = img.resize((new_w, new_h), Image.LANCZOS)
        t_resize += time.time() - t2


        # 프레임을 PNG 파일로 저장
        if SAVE_PNG:
            frame_count += 1
            frame_filename = f"frame_{frame_count}.png"
            frame_path = os.path.join(save_dir, frame_filename)
            
            t3 = time.time()
            img.save(frame_path, format="PNG")
            t_png += time.time() - t3
            print(f"[Frame saved] {frame_path}")

        # Base64 인코딩 (API 전송용)
        t4 = time.time()
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=80)
        base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        frames.append(f"data:image/jpeg;base64,{base64_data}")
        t_b64 += time.time() - t4


    cap.release()
    t_total = time.time() - t_total_start
    print(
        f"[Frame Timing]\n"
        f"- open & seek     : {t_open:.2f}s\n"
        f"- frame decode    : {t_read:.2f}s\n"
        f"- resize          : {t_resize:.2f}s\n"
        f"- png save        : {t_png:.2f}s\n"
        f"- jpeg + base64   : {t_b64:.2f}s\n"
        f"- total extract   : {t_total:.2f}s"
    )

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
                print(data["choices"][0]["message"].get("reasoning_content", ""))
                yield data["choices"][0]["message"].get("content", "")
                
    except requests.exceptions.RequestException as e:
        yield f"❌ Error: {str(e)}"


def get_prompt_override(file_path: str) -> str | None:
    if not CONFIG.get("use_filename_prompt", True):
        return None
    return build_text_override(file_path)


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
    txt_path = os.path.join(prompt_dir, f"{matched_keyword}.txt")
    if not os.path.exists(txt_path):
        return None
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def analyze_video_file(file_path: str) -> str:
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
            min_frames=CONFIG["min_frames"],
            max_frames=CONFIG["max_frames"],
            start_ratio=CONFIG["frame_start"],
            end_ratio=CONFIG["frame_end"],
            uneven_sampling=CONFIG["uneven_sampling"],
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
            prompt_text = (get_prompt_override(input_path) or "")
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
            summaries = []
            for video in videos:
                result = analyze_video_file(video)
                summaries.append(os.path.basename(video) + "\n" + result + "\n")
            summary_text = "\n".join(summaries).strip() + "\n"
            batch_dir = make_batch_output_dir(input_path)
            os.makedirs(batch_dir, exist_ok=True)
            return save_result_txt(
                videos,
                summary_text,
                folder_name=folder_name,
                base_dir=batch_dir,
                system_prompt=CONFIG["system_prompt"],
                prompt_text="BATCH_SUMMARY",
            )

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
    frames_used: int | None = None,
) -> tuple[str, str]:
    if not video_paths:
        return "", "❌ No video paths"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    payload = {
        "systemPrompt": system_prompt or "",
        "prompt": prompt_text or "",
        "response": result_text,
        "framesUsed": frames_used,
    }
    content = json.dumps(payload, ensure_ascii=False)

    # Folder summary mode
    if folder_name is not None:
        target_dir = base_dir or os.path.dirname(video_paths[0])
        out_path = os.path.join(target_dir, f"{folder_name}.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(content)
            return out_path, f"✅ Saved: {out_path}"
        except OSError:
            fallback = os.path.join(RESULTS_DIR, f"{folder_name}.json")
            with open(fallback, "w", encoding="utf-8") as f:
                f.write(content)
            return fallback, f"✅ Saved (fallback): {fallback}"

    # Single/individual file mode
    last_path = ""
    for vp in video_paths:
        base_name = os.path.splitext(os.path.basename(vp))[0]
        target_dir = base_dir or os.path.dirname(vp)
        out_path = os.path.join(target_dir, f"{base_name}.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(content)
            last_path = out_path
        except OSError:
            fallback = os.path.join(RESULTS_DIR, f"{base_name}.json")
            with open(fallback, "w", encoding="utf-8") as f:
                f.write(content)
            last_path = fallback

    return last_path, f"✅ Saved: {last_path}" if last_path else "❌ Save failed"


def save_result_txt_to_dirs(
    video_paths: list[str],
    frame_dirs: list[str],
    result_text: str,
    system_prompt: str | None = None,
    prompt_text: str | None = None,
    frames_used: int | None = None,
) -> None:
    if not video_paths or not frame_dirs:
        return
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for vp, d in zip(video_paths, frame_dirs):
        save_result_txt(
            [vp],
            result_text,
            base_dir=d,
            system_prompt=system_prompt,
            prompt_text=prompt_text,
            frames_used=frames_used,
        )


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
    t_total_start = time.time()
    frames_time = 0.0
    infer_time = 0.0
    # Build API messages (히스토리 무시 - 현재 메시지만 전송)
    api_messages = [{"role": "system", "content": CONFIG["system_prompt"]}]
    # history 파라미터는 무시됨 - 토큰 수 일정하게 유지
    
    # Build multimodal content
    content_parts = []
    
    # Process files
    uploaded_filenames = []
    video_paths = []
    frame_dirs = []
    frames_used_counts = []
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
                json_text_override = get_prompt_override(file_path)
                
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
                t_frames_start = time.time()

                frames, save_dir = extract_video_frames(
                    file_path,
                    seconds_interval=CONFIG["seconds_interval"],
                    min_frames=CONFIG["min_frames"],
                    max_frames=CONFIG["max_frames"],
                    start_ratio=CONFIG["frame_start"],
                    end_ratio=CONFIG["frame_end"],
                    uneven_sampling=CONFIG["uneven_sampling"],
                    return_dir=True,
                    resize_ratio=CONFIG["resize_ratio"],
                )

                frames_time += time.time() - t_frames_start

                if save_dir:
                    frame_dirs.append(save_dir)
                frames_used_counts.append(len(frames))
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
    t_infer_start = time.time()

    full_response = ""
    for chunk in send_chat_request(api_messages, stream=False):
        full_response += chunk

    infer_time = time.time() - t_infer_start

    if video_paths and full_response:
        save_result_txt_to_dirs(
            video_paths,
            frame_dirs,
            full_response,
            system_prompt=CONFIG["system_prompt"],
            prompt_text=text,  # respond_multimodal에서 최종으로 넣은 text 변수
            frames_used=frames_used_counts[0] if frames_used_counts else None,
        )
    
    t_total = time.time() - t_total_start

    frames_used_line = ""
    if frames_used_counts:
        frames_used_line = f"Frames used: {frames_used_counts[0]}\n"

    timing_info = (
        f"\n\n---\n"
        f"⏱ Processing Time\n"
        + frames_used_line +
        f"- Frame extraction: {frames_time:.2f}s  Model inference: {infer_time:.2f}s\n"
        f"- Total: {t_total:.2f}s\n"
    )

    return full_response + timing_info if full_response else "⏳ No response received"


def save_system_prompt(index: int, prompt_text: str) -> str:
    if index < 0 or index >= len(CONFIG["system_prompts"]):
        return "❌ Invalid tab index"
    CONFIG["system_prompts"][index] = prompt_text
    CONFIG["system_prompt"] = prompt_text
    save_system_prompt_file(index, prompt_text)
    return f"✅ Saved & Using Tab {index + 1}"



def add_prompt_tab(tab_count: int):
    tab_count = min(5, max(1, int(tab_count) + 1))
    show_tab2 = tab_count >= 2
    show_tab3 = tab_count >= 3
    show_tab4 = tab_count >= 4
    show_tab5 = tab_count >= 5
    # create sp5.txt when tab 5 is enabled
    if show_tab5:
        os.makedirs(SYSTEM_PROMPT_DIR, exist_ok=True)
        sp5_path = os.path.join(SYSTEM_PROMPT_DIR, "sp5.txt")
        if not os.path.exists(sp5_path):
            with open(sp5_path, "w", encoding="utf-8") as f:
                f.write("")
    return (
        tab_count,
        gr.update(visible=show_tab2),
        gr.update(visible=show_tab2),
        gr.update(visible=show_tab3),
        gr.update(visible=show_tab3),
        gr.update(visible=show_tab4),
        gr.update(visible=show_tab4),
        gr.update(visible=show_tab5),
        gr.update(visible=show_tab5),
    )


# ============================================================================
# Settings Callbacks
# ============================================================================

def update_config(server_key, temperature, top_p, top_k, max_tokens, seconds_interval, min_frames, max_frames, frame_start, frame_end, uneven_sampling, use_filename_prompt, resize_ratio):
    CONFIG["server_key"] = server_key
    CONFIG["temperature"] = temperature
    CONFIG["top_p"] = top_p
    CONFIG["top_k"] = top_k
    CONFIG["max_tokens"] = int(max_tokens)
    CONFIG["seconds_interval"] = float(seconds_interval)
    CONFIG["min_frames"] = int(min_frames)
    CONFIG["max_frames"] = int(max_frames)
    CONFIG["frame_start"] = float(frame_start)
    CONFIG["frame_end"] = float(frame_end)
    CONFIG["uneven_sampling"] = bool(uneven_sampling)
    CONFIG["use_filename_prompt"] = bool(use_filename_prompt)
    CONFIG["resize_ratio"] = float(resize_ratio)

    #min_frames가 max_frames보다 크면 조정
    CONFIG["min_frames"] = min(CONFIG["min_frames"], CONFIG["max_frames"])
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
                chatbot=gr.Chatbot(height=700),
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
                viewer_path = gr.Textbox(label="TXT Path", placeholder="/path/to/result.json")
                viewer_btn = gr.Button("Load Result")
                viewer_content = gr.Textbox(label="TXT Content", lines=12)
        
        with gr.Column(scale=1):
            gr.Markdown("## ⚙️ Settings")
            
            with gr.Accordion("Server", open=True):
                server_select = gr.Dropdown(
                    choices=list(LLAMA_SERVERS.keys()),
                    value=CONFIG["server_key"],
                    label="LLaMA Server"
                )
                status_text = gr.Textbox(label="Status", value="⚪ Unknown", interactive=False)
                check_btn = gr.Button("🔄 Check Status")

            tab_count_state = gr.State(4)
            with gr.Accordion("System Prompt", open=False):
                use_filename_prompt = gr.Checkbox(label="Use Filename Prompt", value=CONFIG["use_filename_prompt"])

                prompt_status = gr.Textbox(label="Prompt Status", interactive=False)

                with gr.Tabs():
                    with gr.TabItem("System Prompt 1", id=0):
                        system_prompt_1 = gr.Textbox(
                            label="System Prompt 1",
                            value=CONFIG["system_prompts"][0],
                            lines=6,
                            max_lines=6,
                        )
                        save_prompt_1 = gr.Button("Save & Use")
                    with gr.TabItem("System Prompt 2", id=1):
                        system_prompt_2 = gr.Textbox(
                            label="System Prompt 2",
                            value=CONFIG["system_prompts"][1],
                            lines=6,
                            max_lines=6,
                            visible=True,
                        )
                        save_prompt_2 = gr.Button("Save & Use", visible=True)
                    with gr.TabItem("System Prompt 3", id=2):
                        system_prompt_3 = gr.Textbox(
                            label="System Prompt 3",
                            value=CONFIG["system_prompts"][2],
                            lines=6,
                            max_lines=6,
                            visible=True,
                        )
                        save_prompt_3 = gr.Button("Save & Use", visible=True)
                    with gr.TabItem("System Prompt 4", id=3):
                        system_prompt_4 = gr.Textbox(
                            label="System Prompt 4",
                            value=CONFIG["system_prompts"][3] if len(CONFIG["system_prompts"]) > 3 else "",
                            lines=6,
                            max_lines=6,
                            visible=True,
                        )
                        save_prompt_4 = gr.Button("Save & Use", visible=True)
                    with gr.TabItem("System Prompt 5", id=4):
                        system_prompt_5 = gr.Textbox(
                            label="System Prompt 5",
                            value=CONFIG["system_prompts"][4] if len(CONFIG["system_prompts"]) > 4 else "",
                            lines=6,
                            max_lines=6,
                            visible=False,
                        )
                        save_prompt_5 = gr.Button("Save & Use", visible=False)

                add_tab_btn = gr.Button("Add Tab (shows 5)")

            with gr.Accordion("Parameters", open=False):
                temperature = gr.Slider(0.0, 2.0, value=0.6, step=0.1, label="Temperature")
                top_p = gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="Top P")
                top_k = gr.Slider(1, 100, value=2, step=1, label="Top K")
                max_tokens = gr.Number(value=4096, label="Max Tokens", precision=0)

            with gr.Accordion("Video Settings", open=False):
                seconds_interval = gr.Slider(0.1, 5.0, value=0.3, step=0.1, label="Seconds Interval (초 단위)")
                min_frames = gr.Slider(0, 200, value=CONFIG["min_frames"], step=1, label="Min Frames")
                max_frames = gr.Slider(4, 900, value=100, step=1, label="Max Frames")
                frame_start = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Frame Start Ratio")
                frame_end = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Frame End Ratio")
                uneven_sampling = gr.Checkbox(value=False, label="Uneven Sampling (center-weighted)")
                resize_ratio = gr.Slider(0.1, 1.0, value=CONFIG["resize_ratio"], step=0.1, label="Resize Ratio")

            
            apply_btn = gr.Button("Apply Settings", variant="primary")
    
    # Event handlers
    save_prompt_1.click(fn=lambda p: save_system_prompt(0, p), inputs=[system_prompt_1], outputs=[prompt_status])
    save_prompt_2.click(fn=lambda p: save_system_prompt(1, p), inputs=[system_prompt_2], outputs=[prompt_status])
    save_prompt_3.click(fn=lambda p: save_system_prompt(2, p), inputs=[system_prompt_3], outputs=[prompt_status])
    save_prompt_4.click(fn=lambda p: save_system_prompt(3, p), inputs=[system_prompt_4], outputs=[prompt_status])
    save_prompt_5.click(fn=lambda p: save_system_prompt(4, p), inputs=[system_prompt_5], outputs=[prompt_status])

    add_tab_btn.click(
        fn=add_prompt_tab,
        inputs=[tab_count_state],
        outputs=[tab_count_state, system_prompt_2, save_prompt_2, system_prompt_3, save_prompt_3, system_prompt_4, save_prompt_4, system_prompt_5, save_prompt_5]
    )

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
        fn=lambda s: (
            CONFIG.__setitem__("server_key", s),
            check_server_health()
        )[1],
        inputs=[server_select],
        outputs=[status_text]
    )


    apply_btn.click(
        fn=update_config,
        inputs=[
            server_select,
            temperature, top_p, top_k, max_tokens,
            seconds_interval, min_frames, max_frames,
            frame_start, frame_end, uneven_sampling, use_filename_prompt, resize_ratio
        ],
        outputs=[status_text]
    )

    
    demo.load(fn=check_server_health, outputs=[status_text])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8932)
