# (Full script as you provided, with edits. I've omitted parts that are unchanged above for brevity in presentation,
#  but this paste is the entire runnable script — only the necessary edits were made.)

import os
import sys
import time
import threading
import queue
import base64
import io
import json
import traceback
from collections import Counter, deque

# Load optional .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import requests
import numpy as np
from PIL import Image
import cv2
import platform
import subprocess

import torch
# Check for MPS support and select appropriate torch.device
mps_built = torch.backends.mps.is_built() if hasattr(torch.backends, "mps") else False
mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False
DEVICE = torch.device("mps" if mps_built and mps_available else "cpu")
# DEVICE = "cpu"
print(f"Using device: {DEVICE} (MPS built: {mps_built}, available: {mps_available})")
# --- Audio tuning (tweak these values if needed) ---
# --- Audio tuning (tweak these values if needed) ---
HOTWORD_LISTEN_TIMEOUT = 6        # initial hotword listen timeout (seconds)
HOTWORD_PHRASE_LIMIT  = 6        # phrase_time_limit for the initial hotword chunk
FOLLOWUP_PHRASE_LIMIT = 10       # how long to listen for the user's full question after 'helper'
AMBIENT_ADJUST_SEC     = 0.9     # duration to adjust_for_ambient_noise
PAUSE_THRESHOLD        = 1.0     # how long of silence marks end of sentence (higher -> wait longer)
NON_SPEAKING_DURATION  = 0.7     # recognizer.non_speaking_duration
ENERGY_THRESHOLD       = None    # set to None to keep dynamic; set int (e.g., 400) for noisy env
# TTS on macOS
TTS_VOICE = os.environ.get("TTS_VOICE", "Karen")   # try "Alex", "Samantha", etc.
TTS_RATE  = int(os.environ.get("TTS_RATE", "190"))  # words per minute for 'say -r'
print("Audio tuning:", "AMBIENT_ADJUST_SEC=", AMBIENT_ADJUST_SEC,
    "PAUSE_THRESHOLD=", PAUSE_THRESHOLD,
    "FOLLOWUP_PHRASE_LIMIT=", FOLLOWUP_PHRASE_LIMIT,
    "HOTWORD_TIMEOUT=", HOTWORD_LISTEN_TIMEOUT,
    "TTS_VOICE=", TTS_VOICE, "TTS_RATE=", TTS_RATE)
# --- Detection tuning (tweak these values to speed up) ---
# DETECT_IMG_SIZE and DETECTION_INTERVAL added above tuning prints
DETECT_IMG_SIZE = int(os.environ.get("DETECT_IMG_SIZE", "320"))  # resize for faster inference
DETECTION_INTERVAL = int(os.environ.get("DETECTION_INTERVAL", "1"))  # run detection every N frames
print("Detection tuning:", "DETECT_IMG_SIZE=", DETECT_IMG_SIZE, "DETECTION_INTERVAL=", DETECTION_INTERVAL)
 # Helper to configure SR Recognizer with tuned parameters
def _make_recognizer():
    r = sr.Recognizer()
    if ENERGY_THRESHOLD is not None:
        r.dynamic_energy_threshold = False
        r.energy_threshold = ENERGY_THRESHOLD
    else:
        r.dynamic_energy_threshold = True
    r.pause_threshold = PAUSE_THRESHOLD
    r.non_speaking_duration = NON_SPEAKING_DURATION
    return r

# YOLO (ultralytics)
try:
    from ultralytics import YOLO
except Exception as e:
    print("ERROR: ultralytics is required. Install with: pip install ultralytics")
    raise

# TTS
try:
    import pyttsx3
    _tts_engine = None
    try:
        _tts_engine = pyttsx3.init()
    except Exception as e:
        print("Warning: pyttsx3 initialization failed; falling back to print-only TTS.", e)
        _tts_engine = None
except Exception as e:
    print("Warning: pyttsx3 not available; falling back to print-only TTS.")
    pyttsx3 = None
    _tts_engine = None

# Speech recognition
try:
    import speech_recognition as sr
except Exception as e:
    print("ERROR: speech_recognition is required. Install with: pip install SpeechRecognition")
    raise

import threading
# Shared microphone to avoid device conflicts across threads
mic_lock = threading.Lock()
try:
    shared_mic = sr.Microphone()
except Exception as e:
    print("Shared microphone unavailable:", e)
    shared_mic = None

# Optional Vosk
VOSK_AVAILABLE = False
VOSK_MODEL = None
VOSK_MODEL_PATH = os.environ.get("VOSK_MODEL_PATH", "").strip()
if VOSK_MODEL_PATH:
    try:
        if os.path.exists(VOSK_MODEL_PATH):
            import vosk  # may raise
            VOSK_MODEL = vosk.Model(VOSK_MODEL_PATH)
            VOSK_AVAILABLE = True
            print(f"Vosk model loaded from {VOSK_MODEL_PATH} (offline STT enabled).")
        else:
            print(f"VOSK_MODEL_PATH set but path does not exist: {VOSK_MODEL_PATH}")
    except Exception as e:
        print("Vosk import/initialization failed; falling back to online STT if available.", e)

# Configuration from env
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
CAM_INDEX = int(os.environ.get("CAM_INDEX", "0"))
YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolov8x.pt")

# YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolov8n.pt")

try:
    CONF_THRESH = float(os.environ.get("CONF_THRESH", "0.35"))
except Exception:
    CONF_THRESH = 0.35

# Groq LLM config
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Threading primitives and shared state
stop_event = threading.Event()
latest_frame_lock = threading.Lock()
latest_frame = None  # BGR numpy array
qa_queue = queue.Queue()
tts_queue = queue.Queue()
status_lock = threading.Lock()
status_text = "Idle"

# NEW: block_event indicates "exclusive audio+LLM" mode: pause detections and other listeners
block_event = threading.Event()   # ### NEW

# Keep track of last spoken text (normalized) to suppress duplicates
last_spoken_text = ""
last_spoken_lock = threading.Lock()

# Add tracking of last summary to speak only when changed
last_summary_time = 0
summary_interval = 0  # seconds
last_summary = None  # track last spoken summary
last_detections = []

# FPS calculation
fps_deque = deque(maxlen=30)

def _normalize_for_compare(s: str) -> str:
    """Normalize text for comparison: strip, lowercase, collapse spaces."""
    if s is None:
        return ""
    return " ".join(s.strip().lower().split())

# Create TTS worker thread
def tts_worker():
    global _tts_engine
    while not stop_event.is_set():
        try:
            text = tts_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        try:
            if platform.system() == 'Darwin':
                # use macOS 'say' with chosen voice and rate
                subprocess.run(['say', '-v', TTS_VOICE, '-r', str(TTS_RATE), text])
            elif _tts_engine:
                # configure pyttsx3 voice and rate
                try:
                    voices = _tts_engine.getProperty('voices')
                    if voices:
                        _tts_engine.setProperty('voice', voices[0].id)
                    _tts_engine.setProperty('rate', TTS_RATE)
                except Exception:
                    pass
                _tts_engine.say(text)
                _tts_engine.runAndWait()
            else:
                print("[TTS]", text)
        except Exception as e:
            print("TTS error:", e)
            print("[TTS fallback print]", text)

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

def speak(text: str):
    """Queue text for TTS (non-blocking), but only if it differs from the last spoken text.

    Normalization: lowercased, trimmed, and inner whitespace collapsed.
    """
    global last_spoken_text
    if not text:
        return
    norm = _normalize_for_compare(text)
    with last_spoken_lock:
        if norm == last_spoken_text:
            # Duplicate: skip enqueueing to avoid repeated TTS
            return
        # Update last_spoken_text immediately to suppress near-simultaneous duplicates
        last_spoken_text = norm
    print(f"[SPEAK] {text}", flush=True)
    tts_queue.put(text)

# Helper: convert frame (BGR) to data URI PNG
def frame_to_data_uri(frame_bgr: np.ndarray) -> str:
    try:
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        data_uri = f"data:image/png;base64,{b64}"
        return data_uri
    except Exception as e:
        print("frame_to_data_uri error:", e)
        return ""

# Speech transcription helper
def transcribe_audio_data(audio_data, recognizer: sr.Recognizer):
    """Transcribe sr.AudioData using manual Vosk if available; otherwise use Google.
    Avoids calling recognizer.recognize_vosk() (which prints that 'please download model' message).
    Returns recognized text or ''.
    """
    try:
        # Prefer manual Vosk if we've loaded a model
        if VOSK_AVAILABLE and VOSK_MODEL is not None:
            try:
                # Create a Kaldi recognizer using the pre-loaded model (VOSK_MODEL)
                # audio_data.get_raw_data() yields PCM16 by default in speech_recognition
                rec = vosk.KaldiRecognizer(VOSK_MODEL, audio_data.sample_rate)
                rec.AcceptWaveform(audio_data.get_raw_data())
                # Use final result
                res = rec.Result()
                j = json.loads(res)
                text = j.get("text", "") or ""
                return text.strip()
            except Exception as e:
                # If manual Vosk fails for any reason, log and fall back to Google if available
                print("Manual Vosk recognition failed, falling back to online STT:", e)

        # Fall back to Google online recognizer
        try:
            text = recognizer.recognize_google(audio_data)
            return (text or "").strip()
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            # network/proxy error or Google side error
            print("Google STT RequestError:", e)
            return ""
        except Exception as e:
            print("Google STT failed:", e)
            return ""
    except Exception as e:
        print("transcribe_audio_data unexpected error:", e)
        return ""


def transcribe_chunk(timeout=3, phrase_time_limit=2):
    """Record a short chunk and transcribe it (blocking). Returns text or empty string."""
    recognizer = _make_recognizer()
    if shared_mic is None:
        return ""
    with mic_lock:
        with shared_mic as source:
            try:
                recognizer.adjust_for_ambient_noise(source, duration=AMBIENT_ADJUST_SEC)
            except Exception:
                pass
            try:
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            except sr.WaitTimeoutError:
                return ""
            except Exception as e:
                print("transcribe_chunk error:", e)
                return ""
    text = transcribe_audio_data(audio, recognizer)
    return (text or "").strip()

def listen_for_seconds(sec=5, phrase_time_limit=None):
    """Listen up to sec seconds and return transcript (blocking).
       If phrase_time_limit is None, allow full sec; else use phrase_time_limit."""
    recognizer = _make_recognizer()
    if shared_mic is None:
        return ""
    with mic_lock:
        with shared_mic as source:
            try:
                recognizer.adjust_for_ambient_noise(source, duration=AMBIENT_ADJUST_SEC)
            except Exception:
                pass
            try:
                ptl = sec if (phrase_time_limit is None) else phrase_time_limit
                audio = recognizer.listen(source, timeout=sec, phrase_time_limit=ptl)
            except sr.WaitTimeoutError:
                return ""
            except Exception as e:
                print("listen_for_seconds error:", e)
                return ""
    text = transcribe_audio_data(audio, recognizer)
    return (text or "").strip()

# YOLO model loading (CPU by default)
print(f"Loading YOLO model '{YOLO_MODEL}'. This may download weights on first run...")
try:
    yolo = YOLO(YOLO_MODEL)
    yolo(torch.zeros((1, 3, DETECT_IMG_SIZE, DETECT_IMG_SIZE)).to(DEVICE))
    # Move model once to desired device and fuse for faster inference
    try:
        yolo.to(DEVICE)
    except Exception:
        try:
            yolo.model.to(DEVICE)
        except Exception:
            pass
    try:
        yolo.fuse()
    except Exception:
        pass
    names = yolo.model.names if hasattr(yolo, "model") and hasattr(yolo.model, "names") else {}
    # Verify model device
    try:
        dev = next(yolo.model.parameters()).device
        print(f"YOLO model parameters are on device: {dev}")
    except Exception:
        pass
    print("YOLO model loaded.")
except Exception as e:
    print("Failed to load YOLO model:", e)
    raise

def detect_objects(frame_bgr):
    """Run YOLO inference on BGR frame. Returns list of (cls_name, conf, (x1,y1,x2,y2))."""
    try:
        # ultralytics accepts numpy array (BGR) directly; model already on DEVICE
        # run inference directly on full frame (let YOLO handle resizing) for lower Python overhead
        results = yolo(frame_bgr, imgsz=DETECT_IMG_SIZE, conf=CONF_THRESH, verbose=False)
        dets = []
        if results and len(results) > 0:
            r = results[0]
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                return dets
            for box in boxes:
                try:
                    xyxy = box.xyxy.tolist()[0]  # [x1,y1,x2,y2]
                    conf = float(box.conf.tolist()[0])
                    cls_idx = int(box.cls.tolist()[0])
                    cls_name = names.get(cls_idx, str(cls_idx))
                    dets.append((cls_name, conf, tuple(map(int, xyxy))))
                except Exception:
                    # Fallback extraction
                    try:
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_idx = int(box.cls[0].cpu().numpy())
                        cls_name = names.get(cls_idx, str(cls_idx))
                        dets.append((cls_name, conf, tuple(map(int, xyxy))))
                    except Exception:
                        continue
        return dets
    except Exception as e:
        print("detect_objects error:", e)
        return []

def summarize_detections(detections, top_k=3):
    """Return a compact English summary of detections list [(cls,conf,box), ...]."""
    if not detections:
        return "I see nothing significant."
    counts = Counter([cls for cls, _, _ in detections])
    most = counts.most_common(top_k)
    parts = []
    for cls, cnt in most:
        if cnt == 1:
            parts.append(f"a {cls}")
        else:
            parts.append(f"{cnt} {cls}s")
    summary = ", ".join(parts)
    return f"I see {summary}."

def draw_overlay(frame, detections, fps, status):
    """Draw boxes, labels, FPS and status on frame (in-place)."""
    for cls, conf, (x1, y1, x2, y2) in detections:
        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        label = f"{cls} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w + 6, y1), (0, 200, 0), -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    # FPS and status
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"Status: {status}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

def call_groq_llm(user_query: str, frame_bgr: np.ndarray, max_retries=2, timeout=30):
    """Call Groq's OpenAI-compatible Chat Completions API with data URI image and user query."""
    if not GROQ_API_KEY:
        speak("Groq API key not set.")
        print("GROQ_API_KEY missing; cannot perform Q&A.")
        return None
    data_uri = frame_to_data_uri(frame_bgr)
    if not data_uri:
        speak("Failed to process the image.")
        return None
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful, concise assistant that can see an image and answer questions about the user’s surroundings."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_query},
                    {"type": "image_url", "image_url": {"url": data_uri}}
                ]
            }
        ],
        "temperature": 0,
        "max_tokens": 4096
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    attempt = 0
    while attempt <= max_retries and not stop_event.is_set():
        attempt += 1
        try:
            resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=timeout)
            if resp.status_code != 200:
                print(f"Groq API returned {resp.status_code}: {resp.text}")
                time.sleep(1 + attempt)
                continue
            j = resp.json()
            # Parse response: follow OpenAI response shape
            choices = j.get("choices") or []
            if not choices:
                # fallback to top-level 'text' or 'message' keys
                text = j.get("text") or str(j)
                return text
            # Typical: choices[0].message.content
            first = choices[0]
            msg = first.get("message") or first.get("text") or {}
            if isinstance(msg, dict):
                content = msg.get("content") or ""
                if isinstance(content, list):
                    # content could be array of parts; join
                    parts = []
                    for c in content:
                        if isinstance(c, dict):
                            parts.append(c.get("text") or "")
                        else:
                            parts.append(str(c))
                    return "\n".join(parts).strip()
                return content if isinstance(content, str) else str(content)
            else:
                return str(msg)
        except requests.RequestException as e:
            print("Groq API request error:", e)
            time.sleep(1 + attempt)
        except Exception as e:
            print("Unexpected Groq API processing error:", e, traceback.format_exc())
            time.sleep(1 + attempt)
    speak("Sorry, I couldn't get an answer right now.")
    return None

def qa_worker_thread():
    """Background worker to process Q&A tasks from queue. Each task is (query_text, frame_bgr)."""
    global status_text
    while not stop_event.is_set():
        try:
            query, frame_bgr = qa_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            with status_lock:
                status_text = "Answering..."
            speak("Answering.")
            print("Sending Q&A to Groq:", query)
            answer = call_groq_llm(query, frame_bgr)
            if answer:
                print("LLM Answer:", answer)
                speak(answer)
            else:
                speak("I couldn't retrieve an answer.")
        except Exception as e:
            print("QA worker error:", e, traceback.format_exc())
            speak("An error occurred while answering.")
        finally:
            with status_lock:
                status_text = "Idle"

qa_thread = threading.Thread(target=qa_worker_thread, daemon=True)
qa_thread.start()

def hotword_listener_thread():
    """Continuously listen in short chunks for 'helper' prefix. On detection, capture query and dispatch QA.

    ### CHANGED: This now performs the QA inline while holding `block_event` so
    that *everything else* pauses until the LLM answer is received.
    """
    print("Hotword listener started.")
    recognizer = _make_recognizer()
    if shared_mic is None:
        print("Hotword listener: shared microphone unavailable; exiting.")
        return
    while not stop_event.is_set():
        # If another exclusive flow is running, don't try to grab the mic — wait a bit
        if block_event.is_set():
            time.sleep(0.05)
            continue

        audio = None
        # Capture audio while holding the mic lock so only one thread uses the device at a time
        with mic_lock:
            with shared_mic as source:
                # calibrate ambient noise using tuned constant
                try:
                    recognizer.adjust_for_ambient_noise(source, duration=AMBIENT_ADJUST_SEC)
                except Exception:
                    pass
                try:
                    audio = recognizer.listen(source, timeout=HOTWORD_LISTEN_TIMEOUT, phrase_time_limit=HOTWORD_PHRASE_LIMIT)
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    print("Hotword listener error:", e)
                    time.sleep(0.5)
                    continue

        # Process audio after releasing mic lock
        if audio is None:
            continue
        try:
            text = transcribe_audio_data(audio, recognizer).strip()
            if not text:
                continue
            text_low = text.lower()
            if text_low.startswith("helper"):
                # Enter exclusive mode: block other operations
                block_event.set()   # ### CHANGED: block everything else
                try:
                    # Extract remainder after hotword
                    remainder = text[len("helper"):].strip(" ,.!?")
                    if remainder:
                        query = remainder
                    else:
                        with status_lock:
                            status_text = "Listening..."
                        speak("Listening for your question.")
                        # Synchronously listen for a longer query
                        query = listen_for_seconds(FOLLOWUP_PHRASE_LIMIT, phrase_time_limit=FOLLOWUP_PHRASE_LIMIT)

                    if not query:
                        # try one more short listen
                        query = listen_for_seconds(3, phrase_time_limit=3)
                        if not query:
                            speak("I didn't catch that.")
                        continue

                    # Capture latest frame
                    with latest_frame_lock:
                        frame_copy = latest_frame.copy() if latest_frame is not None else None
                    if frame_copy is None:
                        speak("No camera frame available.")
                        continue

                    # Do the LLM call inline (synchronously) and wait for the result.
                    with status_lock:
                        status_text = "Answering..."
                    speak("Answering.")
                    print("Sending Q&A to Groq (inline):", query)
                    answer = call_groq_llm(query, frame_copy)
                    if answer:
                        print("LLM Answer (inline):", answer)
                        speak(answer)
                    else:
                        speak("I couldn't retrieve an answer.")
                except Exception as e:
                    print("Hotword processing error:", e, traceback.format_exc())
                    speak("An error occurred while answering.")
                finally:
                    with status_lock:
                        status_text = "Idle"
                    block_event.clear()   # ### CHANGED: release the block so normal ops resume
        except Exception as e:
            print("Hotword processing error:", e)
            time.sleep(0.2)
    print("Hotword listener exiting.")

hotword_thread = threading.Thread(target=hotword_listener_thread, daemon=True)
hotword_thread.start()

# After each spoken detection summary, we need to open a 5-second listening window but not block detection loop.
# We'll spawn a helper thread for that.
def post_summary_listen_worker():
    """Listen for 5 seconds and, if 'helper' appears, listen for a question and answer inline.

    ### CHANGED: run in exclusive mode (block_event set) and perform LLM call inline,
    so nothing else runs while we listen + wait for the answer.
    """
    # If another exclusive flow is running already, bail out.
    if block_event.is_set():
        print("post_summary_listen_worker: another exclusive session active; exiting.")
        return

    # Enter exclusive mode
    block_event.set()
    try:
        with status_lock:
            status_text = "Listening..."
        # Listen for the hotword within the tuned window
        transcript = listen_for_seconds(HOTWORD_LISTEN_TIMEOUT, phrase_time_limit=HOTWORD_PHRASE_LIMIT)
        if not transcript:
            return

        text_low = transcript.lower()
        print("post_summary_listen_worker heard:", text_low)
        if "helper" not in text_low and "help" not in text_low:
            print("post_summary_listen_worker: hotword not detected; ignoring audio.")
            return

        speak("Listening for your question.")
        query = listen_for_seconds(FOLLOWUP_PHRASE_LIMIT, phrase_time_limit=FOLLOWUP_PHRASE_LIMIT)
        if not query:
            speak("I didn't catch that.")
            return

        with latest_frame_lock:
            frame_copy = latest_frame.copy() if latest_frame is not None else None
        if frame_copy is None:
            speak("No camera frame available.")
            return

        # Synchronously perform the LLM call and speak result
        with status_lock:
            status_text = "Answering..."
        speak("Answering.")
        print("post_summary_listen_worker: sending query inline:", query)
        answer = call_groq_llm(query, frame_copy)
        if answer:
            print("post_summary_listen_worker LLM answer:", answer)
            speak(answer)
        else:
            speak("I couldn't retrieve an answer.")
    except Exception as e:
        print("post_summary_listen_worker listen error:", e, traceback.format_exc())
    finally:
        with status_lock:
            status_text = "Idle"
        block_event.clear()


def main():
    global latest_frame, status_text
    cap = cv2.VideoCapture(CAM_INDEX)
    # drop frames to keep low latency
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if not cap.isOpened():
        err = f"Cannot open camera index {CAM_INDEX}."
        print(err)
        speak(err)
        return
    # Try to set FPS
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("Camera opened. Starting main loop.")
    last_summary_time = 0
    summary_interval = 2.0  # seconds
    last_summary = None  # track last spoken summary
    last_detections = []
    window_name = "AssistantCam - Press 'q' to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # NEW: frame_idx declaration before loop
    frame_idx = 0  # frame counter for detection interval

    try:
        while not stop_event.is_set():
            t0 = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Frame read failed; exiting.")
                speak("Camera read failed.")
                break
            # Store latest frame
            with latest_frame_lock:
                latest_frame = frame.copy()

            # Check blocking exclusive mode or currently answering a Q&A request
            with status_lock:
                st = status_text

            # Determine detections: only pause during actual answering
            if st != "Answering...":
                if frame_idx % DETECTION_INTERVAL == 0:
                    detections = detect_objects(frame)
                    last_detections = detections
                else:
                    detections = last_detections
                frame_idx += 1
            else:
                detections = []

            # Debug: immediate print of detections with flush
            now_ts = time.strftime("%H:%M:%S")
            # if detections:
            #     print(f"[{now_ts}] detections (immediate): {detections}", flush=True)
            # else:
            #     print(f"[{now_ts}] no detections", flush=True)

            # FPS calculation: smooth using a deque of the last values
            t1 = time.time()
            elapsed = t1 - t0
            fps_deque.append(1.0 / max(elapsed, 1e-6))
            fps = float(np.mean(fps_deque)) if fps_deque else 0.0

            # Overlay and display: show "Listening/Answering" status during exclusive mode
            status_label = "Listening/Answering" if block_event.is_set() else st
            draw_overlay(frame, detections, fps, status_label)
            cv2.imshow(window_name, frame)

            # Periodic summary every ~summary_interval (only if not in exclusive mode)
            now = time.time()
            if not block_event.is_set() and (now - last_summary_time >= summary_interval):
                # Summarize top 3 classes
                summary = summarize_detections(detections, top_k=3) if 'detections' in locals() else "I see nothing significant."
                # Speak only if summary changed since last time
                if summary != last_summary:
                    speak(summary)
                    # After speaking summary, start 5-second listen in background only if not blocked
                    listen_thread = threading.Thread(target=post_summary_listen_worker, daemon=True)
                    listen_thread.start()
                    last_summary = summary
                last_summary_time = now

            # Handle keypresses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting by user request.")
                stop_event.set()
                break
            # Small sleep to yield CPU (tune for performance)
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received; exiting.")
        stop_event.set()
    except Exception as e:
        print("Main loop error:", e, traceback.format_exc())
    finally:
        stop_event.set()
        time.sleep(0.2)
        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("Waiting for background threads to finish...")
        # Wait for QA and hotword threads a short while
        qa_thread.join(timeout=1.0)
        hotword_thread.join(timeout=0.5)
        tts_thread.join(timeout=0.5)
        print("Shutdown complete.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Fatal error:", e, traceback.format_exc())
        stop_event.set()
