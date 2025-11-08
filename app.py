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
from datetime import datetime, timedelta

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

# Modern UI imports
try:
    import tkinter as tk
    from tkinter import ttk
    import customtkinter as ctk
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.animation import FuncAnimation
    import seaborn as sns
    UI_AVAILABLE = True
    print("Modern UI components loaded successfully")
except ImportError as e:
    print(f"UI components not available: {e}")
    UI_AVAILABLE = False

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

# Dashboard configuration
ENABLE_DASHBOARD = os.environ.get("ENABLE_DASHBOARD", "true").lower() == "true"

# Web interface configuration
ENABLE_WEB_INTERFACE = os.environ.get("ENABLE_WEB_INTERFACE", "true").lower() == "true"
WEB_HOST = os.environ.get("WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.environ.get("WEB_PORT", "5001"))

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

# Stop/Interrupt system
interrupt_event = threading.Event()  # For interrupting current operations
tts_process = None  # Track current TTS process for killing
interrupt_lock = threading.Lock()

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

# Dashboard and Analytics
class AnalyticsTracker:
    def __init__(self):
        self.detection_history = deque(maxlen=1000)  # Store last 1000 detections
        self.fps_history = deque(maxlen=100)
        self.object_counts = Counter()
        self.session_start = datetime.now()
        self.total_detections = 0
        
    def log_detection(self, detections, fps):
        """Log detection events with timestamp"""
        timestamp = datetime.now()
        self.fps_history.append(fps)
        
        for cls, conf, bbox in detections:
            detection_data = {
                'timestamp': timestamp,
                'class': cls,
                'confidence': conf,
                'bbox': bbox
            }
            self.detection_history.append(detection_data)
            self.object_counts[cls] += 1
            self.total_detections += 1
    
    def get_recent_stats(self, minutes=5):
        """Get statistics for the last N minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_detections = [d for d in self.detection_history if d['timestamp'] > cutoff_time]
        
        recent_counts = Counter()
        for detection in recent_detections:
            recent_counts[detection['class']] += 1
            
        return {
            'total_recent': len(recent_detections),
            'unique_objects': len(recent_counts),
            'top_objects': recent_counts.most_common(5),
            'avg_fps': np.mean(list(self.fps_history)[-30:]) if self.fps_history else 0
        }

class ModernDashboard:
    def __init__(self, analytics_tracker):
        if not UI_AVAILABLE:
            self.enabled = False
            return
            
        self.enabled = True
        self.analytics = analytics_tracker
        self.setup_ui()
        self.setup_plots()
        
    def setup_ui(self):
        """Initialize the modern dashboard UI"""
        # Set modern theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Main window
        self.root = ctk.CTk()
        self.root.title("AI Vision Assistant - Real-time Dashboard")
        self.root.geometry("1400x900")
        
        # Create main container
        self.main_container = ctk.CTkFrame(self.root)
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left panel for camera feed
        self.left_panel = ctk.CTkFrame(self.main_container)
        self.left_panel.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        # Camera feed label
        self.camera_label = ctk.CTkLabel(
            self.left_panel, 
            text="Camera Feed Loading...",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.camera_label.pack(pady=10)
        
        # Right panel for dashboard
        self.right_panel = ctk.CTkFrame(self.main_container)
        self.right_panel.pack(side="right", fill="y", padx=(5, 0))
        
        # Dashboard title
        title_label = ctk.CTkLabel(
            self.right_panel,
            text="📊 Real-time Analytics",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.pack(pady=(10, 20))
        
        # Stats frame
        self.stats_frame = ctk.CTkFrame(self.right_panel)
        self.stats_frame.pack(fill="x", padx=10, pady=5)
        
        # Create stat labels
        self.fps_label = ctk.CTkLabel(self.stats_frame, text="FPS: --", font=ctk.CTkFont(size=14))
        self.fps_label.pack(pady=5)
        
        self.detection_label = ctk.CTkLabel(self.stats_frame, text="Detections: --", font=ctk.CTkFont(size=14))
        self.detection_label.pack(pady=5)
        
        self.status_label = ctk.CTkLabel(self.stats_frame, text="Status: Initializing", font=ctk.CTkFont(size=14))
        self.status_label.pack(pady=5)
        
        # Control buttons
        self.controls_frame = ctk.CTkFrame(self.right_panel)
        self.controls_frame.pack(fill="x", padx=10, pady=10)
        
        self.voice_button = ctk.CTkButton(
            self.controls_frame,
            text="🎤 Voice Command",
            command=self.trigger_voice_command,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40
        )
        self.voice_button.pack(pady=5, fill="x")
        
        self.summary_button = ctk.CTkButton(
            self.controls_frame,
            text="📋 Get Summary",
            command=self.get_scene_summary,
            font=ctk.CTkFont(size=14),
            height=35
        )
        self.summary_button.pack(pady=5, fill="x")
        
    def setup_plots(self):
        """Setup matplotlib plots for analytics"""
        if not self.enabled:
            return
            
        # Create matplotlib frame
        self.plot_frame = ctk.CTkFrame(self.right_panel)
        self.plot_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Setup matplotlib with dark theme
        plt.style.use('dark_background')
        
        # Create figure for plots
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(6, 8), facecolor='#2b2b2b')
        self.fig.patch.set_facecolor('#2b2b2b')
        
        # FPS plot
        self.ax1.set_title('Real-time FPS', color='white', fontsize=12, pad=10)
        self.ax1.set_ylabel('FPS', color='white')
        self.ax1.tick_params(colors='white')
        self.ax1.set_facecolor('#1e1e1e')
        
        # Object detection plot
        self.ax2.set_title('Object Detection Count', color='white', fontsize=12, pad=10)
        self.ax2.set_ylabel('Count', color='white')
        self.ax2.tick_params(colors='white')
        self.ax2.set_facecolor('#1e1e1e')
        
        # Embed plots in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Initialize plot data
        self.fps_data = deque(maxlen=50)
        self.time_data = deque(maxlen=50)
        
    def update_camera_feed(self, frame):
        """Update camera feed in dashboard"""
        if not self.enabled:
            return
            
        try:
            # Convert frame to format suitable for tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (640, 480))
            
            # Convert to PIL image then to PhotoImage
            pil_image = Image.fromarray(frame_resized)
            photo = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(640, 480))
            
            # Update label
            self.camera_label.configure(image=photo, text="")
        except Exception as e:
            print(f"Error updating camera feed: {e}")
    
    def update_stats(self, fps, status, detections):
        """Update real-time statistics"""
        if not self.enabled:
            return
            
        try:
            # Update text labels
            self.fps_label.configure(text=f"📈 FPS: {fps:.1f}")
            self.status_label.configure(text=f"🔄 Status: {status}")
            self.detection_label.configure(text=f"👁️ Objects: {len(detections)}")
            
            # Update plots
            current_time = datetime.now()
            self.fps_data.append(fps)
            self.time_data.append(current_time)
            
            # Clear and redraw FPS plot
            self.ax1.clear()
            self.ax1.set_title('Real-time FPS', color='white', fontsize=12, pad=10)
            self.ax1.set_ylabel('FPS', color='white')
            self.ax1.tick_params(colors='white')
            self.ax1.set_facecolor('#1e1e1e')
            
            if len(self.fps_data) > 1:
                times = [(t - self.time_data[0]).total_seconds() for t in self.time_data]
                self.ax1.plot(times, list(self.fps_data), color='#00ff41', linewidth=2)
                self.ax1.fill_between(times, list(self.fps_data), alpha=0.3, color='#00ff41')
            
            # Update object detection plot
            self.ax2.clear()
            self.ax2.set_title('Top Objects (Last 5 min)', color='white', fontsize=12, pad=10)
            self.ax2.set_ylabel('Count', color='white')
            self.ax2.tick_params(colors='white')
            self.ax2.set_facecolor('#1e1e1e')
            
            stats = self.analytics.get_recent_stats(5)
            if stats['top_objects']:
                objects, counts = zip(*stats['top_objects'][:5])
                colors = plt.cm.Set3(np.linspace(0, 1, len(objects)))
                bars = self.ax2.bar(objects, counts, color=colors)
                self.ax2.tick_params(axis='x', rotation=45)
            
            # Refresh canvas
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error updating dashboard: {e}")
    
    def trigger_voice_command(self):
        """Trigger voice command from dashboard"""
        print("Voice command triggered from dashboard")
        # This will be handled by the existing voice system
        
    def get_scene_summary(self):
        """Get scene summary from dashboard"""
        print("Scene summary requested from dashboard")
        # This can trigger the existing summary system
        
    def run(self):
        """Start the dashboard"""
        if self.enabled:
            self.root.after(100, self.update_loop)
            self.root.mainloop()
    
    def update_loop(self):
        """Update loop for dashboard"""
        if self.enabled and not stop_event.is_set():
            self.root.after(100, self.update_loop)

# Global analytics and dashboard instances
analytics_tracker = AnalyticsTracker()
dashboard = None
web_interface = None

# Command history and context
command_history = deque(maxlen=10)
last_command_time = 0
context_memory = {}

# Intelligent Voice Commands System
VOICE_COMMANDS = {
    'describe': 'Provide detailed scene description',
    'count': 'Count specific objects in the scene',
    'navigate': 'Provide navigation assistance',
    'read': 'Read text in the image using OCR',
    'identify': 'Identify specific objects or people',
    'translate': 'Translate visible text to another language',
    'reminder': 'Set location-based reminders',
    'find': 'Find specific objects in the scene',
    'compare': 'Compare objects or elements in the scene',
    'analyze': 'Analyze the current situation or environment',
    'safety': 'Check for safety concerns in the area',
    'accessibility': 'Provide accessibility information',
    'weather': 'Describe weather conditions if visible',
    'time': 'Tell the current time',
    'help': 'List available commands',
    'status': 'Report system status and statistics'
}

def process_advanced_command(command_text, frame):
    """Process advanced voice commands with intelligent parsing and context awareness"""
    global last_command_time, command_history, context_memory
    
    command_lower = command_text.lower().strip()
    current_time = time.time()
    
    # Add to command history
    command_entry = {
        'command': command_text,
        'timestamp': current_time,
        'context': get_scene_context(frame)
    }
    command_history.append(command_entry)
    
    # Handle contextual follow-up commands
    if any(word in command_lower for word in ['again', 'repeat', 'same']):
        if command_history and len(command_history) > 1:
            last_cmd = command_history[-2]['command']  # Get the command before current
            return f"Repeating: {process_advanced_command(last_cmd, frame)}"
        else:
            return "I don't have a previous command to repeat."
    
    # Handle relative references
    if any(word in command_lower for word in ['that', 'it', 'this']):
        if context_memory.get('last_object'):
            command_lower = command_lower.replace('that', context_memory['last_object'])
            command_lower = command_lower.replace('it', context_memory['last_object'])
            command_lower = command_lower.replace('this', context_memory['last_object'])
    
    # Enhanced command routing with fuzzy matching
    command_result = None
    
    if any(word in command_lower for word in ['count', 'how many', 'number of']):
        command_result = process_count_command(command_lower, frame)
    elif any(word in command_lower for word in ['read', 'text', 'sign', 'writing']):
        command_result = process_read_command(command_lower, frame)
    elif any(word in command_lower for word in ['navigate', 'direction', 'way', 'path', 'go', 'route']):
        command_result = process_navigation_command(command_lower, frame)
    elif any(word in command_lower for word in ['find', 'locate', 'where is', 'where are', 'search']):
        command_result = process_find_command(command_lower, frame)
    elif any(word in command_lower for word in ['safety', 'danger', 'hazard', 'safe', 'risk']):
        command_result = process_safety_command(command_lower, frame)
    elif any(word in command_lower for word in ['accessibility', 'accessible', 'wheelchair', 'disability']):
        command_result = process_accessibility_command(command_lower, frame)
    elif any(word in command_lower for word in ['compare', 'difference', 'similar', 'contrast']):
        command_result = process_compare_command(command_lower, frame)
    elif any(word in command_lower for word in ['analyze', 'analysis', 'situation', 'scene']):
        command_result = process_analyze_command(command_lower, frame)
    elif any(word in command_lower for word in ['help', 'commands', 'what can you do', 'assistance']):
        command_result = process_help_command()
    elif any(word in command_lower for word in ['status', 'statistics', 'stats', 'report']):
        command_result = process_status_command()
    elif any(word in command_lower for word in ['time', 'clock', 'what time']):
        command_result = process_time_command()
    elif any(word in command_lower for word in ['weather', 'temperature', 'sunny', 'rainy', 'cloudy']):
        command_result = process_weather_command(command_lower, frame)
    elif any(word in command_lower for word in ['color', 'colors', 'what color']):
        command_result = process_color_command(command_lower, frame)
    elif any(word in command_lower for word in ['size', 'big', 'small', 'large', 'tiny']):
        command_result = process_size_command(command_lower, frame)
    elif any(word in command_lower for word in ['stop', 'halt', 'pause', 'quiet', 'silence']):
        interrupt_processing()
        return "Stopping current operation."
    else:
        # Default to general description with context
        command_result = process_general_query(command_text, frame)
    
    # Update context memory based on the command and result
    update_context_memory(command_text, command_result, frame)
    last_command_time = current_time
    
    return command_result

def get_scene_context(frame):
    """Extract basic context from the current scene"""
    try:
        detections = detect_objects(frame)
        return {
            'object_count': len(detections),
            'objects': [cls for cls, conf, bbox in detections],
            'timestamp': datetime.now()
        }
    except:
        return {'object_count': 0, 'objects': [], 'timestamp': datetime.now()}

def update_context_memory(command, result, frame):
    """Update context memory for future reference"""
    global context_memory
    
    # Extract mentioned objects from command
    detections = detect_objects(frame)
    if detections:
        context_memory['last_object'] = detections[0][0]  # Most confident detection
        context_memory['last_objects'] = [cls for cls, conf, bbox in detections]
    
    # Store recent command context
    context_memory['last_command'] = command
    context_memory['last_result'] = result
    context_memory['last_update'] = time.time()

def process_weather_command(command, frame):
    """Handle weather-related commands"""
    weather_query = """Based on what you can see in this image, describe any weather conditions. 
    Look for: lighting conditions (bright/dark), shadows, wet surfaces, people's clothing, 
    outdoor/indoor setting, and any weather indicators visible."""
    
    return call_groq_llm(weather_query, frame) or "I cannot determine weather conditions from the current view."

def process_color_command(command, frame):
    """Handle color-related commands"""
    color_query = "Describe the main colors you can see in this image. What are the dominant colors and where do you see them?"
    return call_groq_llm(color_query, frame) or "I cannot analyze colors at the moment."

def process_size_command(command, frame):
    """Handle size and dimension-related commands"""
    size_query = "Describe the relative sizes of objects in this image. What looks large, small, or medium-sized? Compare the sizes of different elements."
    return call_groq_llm(size_query, frame) or "I cannot analyze sizes at the moment."

def process_count_command(command, frame):
    """Handle counting commands"""
    # Extract object to count from command
    count_patterns = ['count', 'how many', 'number of']
    object_to_count = command
    
    for pattern in count_patterns:
        if pattern in command:
            object_to_count = command.split(pattern)[-1].strip()
            break
    
    # Get current detections
    detections = detect_objects(frame)
    
    if not object_to_count or object_to_count in ['things', 'objects', 'items']:
        total_count = len(detections)
        object_counts = {}
        for cls, conf, bbox in detections:
            object_counts[cls] = object_counts.get(cls, 0) + 1
        
        if total_count == 0:
            return "I don't see any objects to count."
        
        count_summary = f"I can see {total_count} objects in total: "
        count_details = [f"{count} {obj}" for obj, count in object_counts.items()]
        return count_summary + ", ".join(count_details) + "."
    else:
        # Count specific object
        specific_count = 0
        for cls, conf, bbox in detections:
            if object_to_count.lower() in cls.lower():
                specific_count += 1
        
        if specific_count == 0:
            return f"I don't see any {object_to_count} in the current view."
        else:
            return f"I can see {specific_count} {object_to_count}{'s' if specific_count != 1 else ''} in the scene."

def process_read_command(command, frame):
    """Handle text reading commands using OCR simulation"""
    # This is a placeholder for OCR functionality
    # In a real implementation, you would use libraries like EasyOCR or Tesseract
    try:
        # Simulate OCR detection
        mock_text_areas = [
            "EXIT", "STOP", "WELCOME", "OPEN", "CLOSED", 
            "PARKING", "ENTRANCE", "RESTROOM", "INFORMATION"
        ]
        
        # Use LLM to identify if there's text in the image
        ocr_query = "Are there any signs, text, or written words visible in this image? If so, what do they say?"
        return call_groq_llm(ocr_query, frame) or "I cannot detect any readable text in the current view."
        
    except Exception as e:
        return "I'm having trouble reading text in the image right now."

def process_navigation_command(command, frame):
    """Handle navigation assistance commands"""
    nav_query = """Please provide navigation assistance based on what you can see in this image. 
    Look for: doorways, pathways, obstacles, stairs, ramps, signs, or any navigation-relevant features. 
    Describe the safest path forward and any obstacles to avoid."""
    
    return call_groq_llm(nav_query, frame) or "I cannot provide navigation assistance at the moment."

def process_find_command(command, frame):
    """Handle object finding commands"""
    # Extract what to find from the command
    find_patterns = ['find', 'locate', 'where is', 'where are']
    target_object = command
    
    for pattern in find_patterns:
        if pattern in command:
            target_object = command.split(pattern)[-1].strip()
            break
    
    find_query = f"Can you see {target_object} in this image? If so, describe where it is located and provide directions to reach it."
    return call_groq_llm(find_query, frame) or f"I cannot locate {target_object} in the current view."

def process_safety_command(command, frame):
    """Handle safety assessment commands"""
    safety_query = """Please analyze this image for any potential safety concerns or hazards. 
    Look for: wet floors, obstacles, sharp objects, unstable surfaces, poor lighting, 
    crowded areas, or any other safety risks. Provide safety recommendations."""
    
    return call_groq_llm(safety_query, frame) or "I cannot assess safety conditions at the moment."

def process_accessibility_command(command, frame):
    """Handle accessibility information commands"""
    accessibility_query = """Please analyze this image for accessibility features and barriers. 
    Look for: ramps, elevators, wide doorways, accessible parking, braille signs, 
    handrails, level surfaces, and any barriers that might affect accessibility."""
    
    return call_groq_llm(accessibility_query, frame) or "I cannot provide accessibility information at the moment."

def process_compare_command(command, frame):
    """Handle comparison commands"""
    compare_query = "Please compare and contrast the different objects, colors, sizes, or elements you can see in this image."
    return call_groq_llm(compare_query, frame) or "I cannot make comparisons at the moment."

def process_analyze_command(command, frame):
    """Handle analysis commands"""
    analyze_query = """Please provide a detailed analysis of this scene. Consider:
    - The type of environment (indoor/outdoor, public/private)
    - Activities happening
    - Time of day indicators
    - Overall atmosphere and context
    - Any notable patterns or interesting details"""
    
    return call_groq_llm(analyze_query, frame) or "I cannot analyze the scene at the moment."

def process_help_command():
    """Provide dynamic help information based on context and usage"""
    global command_history
    
    # Basic help text
    help_text = "I'm your AI Vision Assistant! Here are my capabilities:\n\n"
    
    # Core commands
    help_text += "🔍 SCENE ANALYSIS:\n"
    help_text += "• 'Describe' or 'What do you see?' - General scene description\n"
    help_text += "• 'Analyze' - Detailed scene analysis\n"
    help_text += "• 'Count [objects]' - Count specific items\n\n"
    
    help_text += "📖 TEXT & READING:\n"
    help_text += "• 'Read' - Read visible text or signs\n"
    help_text += "• 'Find [text]' - Look for specific text\n\n"
    
    help_text += "🧭 NAVIGATION & LOCATION:\n"
    help_text += "• 'Navigate' - Get movement guidance\n"
    help_text += "• 'Find [object]' - Locate specific items\n"
    help_text += "• 'Where is [object]?' - Get directions to objects\n\n"
    
    help_text += "🛡️ SAFETY & ACCESSIBILITY:\n"
    help_text += "• 'Safety' - Check for hazards\n"
    help_text += "• 'Accessibility' - Accessibility information\n\n"
    
    help_text += "🎨 DETAILED ANALYSIS:\n"
    help_text += "• 'Colors' - Describe colors in the scene\n"
    help_text += "• 'Size' - Compare object sizes\n"
    help_text += "• 'Compare' - Compare different elements\n"
    help_text += "• 'Weather' - Describe weather conditions\n\n"
    
    help_text += "⚙️ SYSTEM INFO:\n"
    help_text += "• 'Status' - System statistics\n"
    help_text += "• 'Time' - Current time\n\n"
    
    # Add usage statistics if available
    if command_history:
        recent_commands = [cmd['command'] for cmd in list(command_history)[-5:]]
        help_text += f"📊 Your recent commands: {', '.join(recent_commands)}\n\n"
    
    help_text += "� STOP CONTROLS:\n"
    help_text += "• Say 'Helper stop' or 'stop' to interrupt speech\n"
    help_text += "• Press 'S' key or ESC to stop immediately\n"
    help_text += "• Say 'quiet', 'silence', or 'enough' to pause\n\n"
    
    help_text += "�💡 TIPS:\n"
    help_text += "• Say 'Helper' first, then your command\n"
    help_text += "• Use 'again' to repeat the last command\n"
    help_text += "• Be specific for better results!\n"
    help_text += "• Try 'Helper describe' for a general overview"
    
    return help_text

def process_status_command():
    """Provide system status and statistics"""
    stats = analytics_tracker.get_recent_stats(5)
    current_time = datetime.now().strftime("%I:%M %p")
    
    status_text = f"System Status Report at {current_time}:\n"
    status_text += f"• Average FPS: {stats['avg_fps']:.1f}\n"
    status_text += f"• Objects detected in last 5 minutes: {stats['total_recent']}\n"
    status_text += f"• Unique object types: {stats['unique_objects']}\n"
    status_text += f"• Session uptime: {datetime.now() - analytics_tracker.session_start}\n"
    
    if stats['top_objects']:
        status_text += "• Most frequent objects: "
        top_3 = stats['top_objects'][:3]
        status_text += ", ".join([f"{obj} ({count})" for obj, count in top_3])
    
    return status_text

def process_time_command():
    """Provide current time"""
    current_time = datetime.now()
    time_str = current_time.strftime("%I:%M %p on %A, %B %d")
    return f"The current time is {time_str}."

def process_general_query(query, frame):
    """Process general queries that don't match specific commands"""
    return call_groq_llm(query, frame)

def _normalize_for_compare(s: str) -> str:
    """Normalize text for comparison: strip, lowercase, collapse spaces."""
    if s is None:
        return ""
    return " ".join(s.strip().lower().split())

# Create TTS worker thread
def tts_worker():
    global _tts_engine, tts_process
    while not stop_event.is_set():
        try:
            text = tts_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        
        # Check if we should interrupt before speaking
        if interrupt_event.is_set():
            continue
            
        try:
            if platform.system() == 'Darwin':
                # use macOS 'say' with chosen voice and rate
                with interrupt_lock:
                    if not interrupt_event.is_set():
                        tts_process = subprocess.Popen(['say', '-v', TTS_VOICE, '-r', str(TTS_RATE), text])
                        
                # Wait for process completion or interruption
                while tts_process and tts_process.poll() is None:
                    if interrupt_event.is_set():
                        try:
                            tts_process.terminate()
                            tts_process.wait(timeout=1)
                        except:
                            try:
                                tts_process.kill()
                            except:
                                pass
                        tts_process = None
                        break
                    time.sleep(0.1)
                tts_process = None
                    
            elif _tts_engine:
                # configure pyttsx3 voice and rate
                try:
                    voices = _tts_engine.getProperty('voices')
                    if voices:
                        _tts_engine.setProperty('voice', voices[0].id)
                    _tts_engine.setProperty('rate', TTS_RATE)
                except Exception:
                    pass
                
                if not interrupt_event.is_set():
                    _tts_engine.say(text)
                    # Check for interruption during speech
                    start_time = time.time()
                    _tts_engine.startLoop(False)
                    while _tts_engine.isBusy():
                        if interrupt_event.is_set():
                            _tts_engine.stop()
                            break
                        _tts_engine.iterate()
                        time.sleep(0.1)
                    _tts_engine.endLoop()
            else:
                if not interrupt_event.is_set():
                    print("[TTS]", text)
        except Exception as e:
            print("TTS error:", e)
            if not interrupt_event.is_set():
                print("[TTS fallback print]", text)

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

def speak(text: str):
    """Queue text for TTS (non-blocking), but only if it differs from the last spoken text.

    Normalization: lowercased, trimmed, and inner whitespace collapsed.
    """
    global last_spoken_text, web_interface
    if not text:
        return
    
    # Don't speak if we're interrupted
    if interrupt_event.is_set():
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
    
    # Update web interface transcript
    if web_interface:
        try:
            web_interface.emit_voice_transcript(text, 'assistant')
        except Exception as e:
            pass  # Don't crash on web interface errors

def stop_speaking():
    """Stop current speech and clear TTS queue"""
    global tts_process
    
    print("[INTERRUPT] Stopping speech...")
    interrupt_event.set()
    
    # Clear the TTS queue
    while not tts_queue.empty():
        try:
            tts_queue.get_nowait()
        except queue.Empty:
            break
    
    # Kill current TTS process if running
    with interrupt_lock:
        if tts_process:
            try:
                tts_process.terminate()
                tts_process.wait(timeout=1)
            except:
                try:
                    tts_process.kill()
                except:
                    pass
            tts_process = None
    
    # Brief pause then clear interrupt
    time.sleep(0.5)
    interrupt_event.clear()
    print("[INTERRUPT] Speech stopped.")

def interrupt_processing():
    """Interrupt current processing and return to idle state"""
    print("[INTERRUPT] Interrupting current processing...")
    
    # Stop speech
    stop_speaking()
    
    # Clear block event to resume normal operations
    block_event.clear()
    
    # Update status
    with status_lock:
        global status_text
        status_text = "Interrupted"
    
    speak("Operation stopped.")
    
    # Return to idle after brief pause
    time.sleep(1)
    with status_lock:
        status_text = "Idle"

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
                text = text.strip()
                
                # Check for stop commands
                if check_for_stop_command(text):
                    return "STOP_COMMAND_DETECTED"
                    
                return text
            except Exception as e:
                # If manual Vosk fails for any reason, log and fall back to Google if available
                print("Manual Vosk recognition failed, falling back to online STT:", e)

        # Fall back to Google online recognizer
        try:
            text = recognizer.recognize_google(audio_data)
            text = (text or "").strip()
            
            # Check for stop commands
            if check_for_stop_command(text):
                return "STOP_COMMAND_DETECTED"
                
            return text
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

def check_for_stop_command(text):
    """Check if the text contains stop commands"""
    if not text:
        return False
        
    stop_commands = [
        'stop', 'halt', 'pause', 'quiet', 'silence', 'shut up',
        'cancel', 'abort', 'interrupt', 'enough', 'nevermind',
        'stop it', 'stop that', 'be quiet', 'shush'
    ]
    
    text_lower = text.lower()
    return any(cmd in text_lower for cmd in stop_commands)


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
        
    # Check if we're already interrupted
    if interrupt_event.is_set():
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
    
    # Check again if we were interrupted during listening
    if interrupt_event.is_set():
        return ""
        
    text = transcribe_audio_data(audio, recognizer)
    text = (text or "").strip()
    
    # Handle stop commands
    if text == "STOP_COMMAND_DETECTED":
        interrupt_processing()
        return ""
        
    return text

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
    """Draw enhanced modern overlay with gradients, animations, and better styling."""
    h, w = frame.shape[:2]
    
    # Create modern gradient background for status bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
    cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
    
    # Modern font styling
    font = cv2.FONT_HERSHEY_DUPLEX
    
    # Animated status indicator based on status
    if status == "Idle":
        color = (0, 255, 0)  # Green
        icon = "●"
    elif status == "Listening/Answering":
        color = (0, 165, 255)  # Orange
        icon = "🎤"
    elif status == "Answering...":
        color = (255, 0, 255)  # Magenta
        icon = "🧠"
    else:
        color = (255, 255, 255)  # White
        icon = "○"
    
    # Status indicator circle
    cv2.circle(frame, (30, 30), 12, color, -1)
    cv2.circle(frame, (30, 30), 12, (255, 255, 255), 2)
    
    # Modern status text with shadow effect
    status_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, status_text, (62, 25), font, 0.7, (0, 0, 0), 3)  # Shadow
    cv2.putText(frame, status_text, (60, 23), font, 0.7, (255, 255, 255), 2)  # Main text
    
    status_text2 = f"Status: {status}"
    cv2.putText(frame, status_text2, (62, 55), font, 0.7, (0, 0, 0), 3)  # Shadow
    cv2.putText(frame, status_text2, (60, 53), font, 0.7, color, 2)  # Main text
    
    # Object count
    obj_count_text = f"Objects: {len(detections)}"
    cv2.putText(frame, obj_count_text, (62, 85), font, 0.6, (0, 0, 0), 3)  # Shadow
    cv2.putText(frame, obj_count_text, (60, 83), font, 0.6, (255, 255, 255), 2)  # Main text
    
    # Control hints in bottom right
    control_hints = "Press: S=Stop, H=Help, Q=Quit"
    (text_w, text_h), _ = cv2.getTextSize(control_hints, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, control_hints, (w - text_w - 10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Enhanced bounding boxes with confidence bars and modern styling
    for cls, conf, (x1, y1, x2, y2) in detections:
        # Dynamic color based on confidence
        conf_normalized = min(max(conf, 0.0), 1.0)
        
        # Color gradient from red (low confidence) to green (high confidence)
        red = int(255 * (1 - conf_normalized))
        green = int(255 * conf_normalized)
        blue = 50
        box_color = (blue, green, red)
        
        # Main detection box with rounded corners effect
        thickness = 3
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
        
        # Corner markers for modern look
        corner_length = 20
        corner_thickness = 4
        # Top-left corner
        cv2.line(frame, (x1, y1), (x1 + corner_length, y1), box_color, corner_thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_length), box_color, corner_thickness)
        # Top-right corner
        cv2.line(frame, (x2, y1), (x2 - corner_length, y1), box_color, corner_thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_length), box_color, corner_thickness)
        # Bottom-left corner
        cv2.line(frame, (x1, y2), (x1 + corner_length, y2), box_color, corner_thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_length), box_color, corner_thickness)
        # Bottom-right corner
        cv2.line(frame, (x2, y2), (x2 - corner_length, y2), box_color, corner_thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_length), box_color, corner_thickness)
        
        # Confidence bar above the label
        bar_width = x2 - x1
        bar_height = 8
        confidence_width = int(bar_width * conf_normalized)
        
        # Background bar
        cv2.rectangle(frame, (x1, y1 - 35), (x2, y1 - 27), (50, 50, 50), -1)
        # Confidence fill
        cv2.rectangle(frame, (x1, y1 - 35), (x1 + confidence_width, y1 - 27), box_color, -1)
        
        # Modern label with background
        label = f"{cls} {conf:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(label, font, 0.6, 2)
        
        # Label background with padding
        label_bg_x1 = x1
        label_bg_y1 = y1 - 25
        label_bg_x2 = x1 + label_w + 16
        label_bg_y2 = y1 - 2
        
        # Gradient background for label
        label_overlay = frame[label_bg_y1:label_bg_y2, label_bg_x1:label_bg_x2].copy()
        cv2.rectangle(frame, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), (0, 0, 0), -1)
        
        # Label text with shadow
        cv2.putText(frame, label, (x1 + 9, y1 - 8), font, 0.6, (0, 0, 0), 3)  # Shadow
        cv2.putText(frame, label, (x1 + 8, y1 - 9), font, 0.6, (255, 255, 255), 2)  # Main text
    
    # Add subtle frame border
    cv2.rectangle(frame, (0, 0), (w-1, h-1), (100, 100, 100), 2)

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
            
            # Handle stop commands first
            if text == "STOP_COMMAND_DETECTED":
                interrupt_processing()
                if web_interface:
                    web_interface.emit_voice_transcript("Stop command detected", 'user')
                continue
                
            text_low = text.lower()
            if text_low.startswith("helper"):
                # Update web interface with detected voice command
                if web_interface:
                    web_interface.emit_voice_transcript(text, 'user')
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
                        speak("Listening for your command.")
                        # Synchronously listen for a longer query
                        query = listen_for_seconds(FOLLOWUP_PHRASE_LIMIT, phrase_time_limit=FOLLOWUP_PHRASE_LIMIT)

                    if not query:
                        # try one more short listen
                        query = listen_for_seconds(3, phrase_time_limit=3)
                        if not query:
                            speak("I didn't catch that. Try saying 'Helper help' for available commands.")
                            continue

                    # Capture latest frame
                    with latest_frame_lock:
                        frame_copy = latest_frame.copy() if latest_frame is not None else None
                    if frame_copy is None:
                        speak("No camera frame available.")
                        continue

                    # Process the command using intelligent command system
                    with status_lock:
                        status_text = "Processing..."
                    
                    print(f"Processing intelligent command: {query}")
                    
                    # Use the new intelligent command processing
                    answer = process_advanced_command(query, frame_copy)
                    
                    with status_lock:
                        status_text = "Answering..."
                    
                    if answer:
                        print("Command Response:", answer)
                        speak(answer)
                    else:
                        speak("I couldn't process that command. Try saying 'Helper help' for available commands.")
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

        speak("Listening for your command.")
        query = listen_for_seconds(FOLLOWUP_PHRASE_LIMIT, phrase_time_limit=FOLLOWUP_PHRASE_LIMIT)
        if not query:
            speak("I didn't catch that. Try saying 'Helper help' for available commands.")
            return

        with latest_frame_lock:
            frame_copy = latest_frame.copy() if latest_frame is not None else None
        if frame_copy is None:
            speak("No camera frame available.")
            return

        # Process using intelligent command system
        with status_lock:
            status_text = "Processing..."
        
        print("post_summary_listen_worker: processing intelligent command:", query)
        answer = process_advanced_command(query, frame_copy)
        
        with status_lock:
            status_text = "Answering..."
        
        if answer:
            print("post_summary_listen_worker command response:", answer)
            speak(answer)
        else:
            speak("I couldn't process that command. Try saying 'Helper help' for available commands.")
    except Exception as e:
        print("post_summary_listen_worker listen error:", e, traceback.format_exc())
    finally:
        with status_lock:
            status_text = "Idle"
        block_event.clear()


def main():
    global latest_frame, status_text, dashboard, web_interface
    
    # Initialize web interface if enabled
    web_interface = None
    if ENABLE_WEB_INTERFACE:
        try:
            from web_interface import WebInterface
            web_interface = WebInterface(
                analytics_tracker=analytics_tracker,
                latest_frame_lock=latest_frame_lock,
                latest_frame=lambda: latest_frame,
                status_lock=status_lock,
                status_text=lambda: status_text,
                process_advanced_command_func=process_advanced_command,
                interrupt_processing_func=interrupt_processing
            )
            
            # Start web interface in separate thread
            web_thread = threading.Thread(
                target=lambda: web_interface.run(host=WEB_HOST, port=WEB_PORT), 
                daemon=True
            )
            web_thread.start()
            print(f"Web interface started on http://{WEB_HOST}:{WEB_PORT}")
        except ImportError as e:
            print(f"Web interface dependencies not available: {e}")
            web_interface = None
        except Exception as e:
            print(f"Failed to start web interface: {e}")
            web_interface = None
    else:
        print("Web interface disabled (set ENABLE_WEB_INTERFACE=true to enable)")
    
    # Initialize dashboard if enabled
    dashboard = None
    if ENABLE_DASHBOARD:
        dashboard = ModernDashboard(analytics_tracker)
        
        # Start dashboard in separate thread if UI is available
        if dashboard.enabled:
            dashboard_thread = threading.Thread(target=dashboard.run, daemon=True)
            dashboard_thread.start()
            print("Modern dashboard started with real-time analytics")
        else:
            print("Dashboard requested but UI components not available")
    else:
        print("Dashboard disabled (set ENABLE_DASHBOARD=true to enable)")
    
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
    window_name = "AI Vision Assistant - Press 'q' to quit"
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

            # Log analytics data
            analytics_tracker.log_detection(detections, fps)

            # Overlay and display: show "Listening/Answering" status during exclusive mode
            status_label = "Listening/Answering" if block_event.is_set() else st
            draw_overlay(frame, detections, fps, status_label)
            cv2.imshow(window_name, frame)
            
            # Update dashboard if available
            if dashboard and dashboard.enabled:
                try:
                    dashboard.update_camera_feed(frame)
                    dashboard.update_stats(fps, status_label, detections)
                except Exception as e:
                    # Dashboard update errors shouldn't crash main loop
                    pass
            
            # Update web interface if available
            if web_interface:
                try:
                    web_interface.emit_camera_stats(fps, status_label, detections)
                except Exception as e:
                    # Web interface update errors shouldn't crash main loop
                    pass

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
            elif key == ord('s') or key == 27:  # 's' key or ESC key
                print("Stop command pressed!")
                interrupt_processing()
            elif key == ord('h'):  # 'h' for help
                print("Keyboard shortcuts:")
                print("  'q' - Quit application")
                print("  's' or ESC - Stop current speech/processing")
                print("  'h' - Show this help")
                speak("Press Q to quit, S or Escape to stop speech, or H for help.")
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
