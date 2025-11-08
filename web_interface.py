"""
Web interface for AI Vision Assistant
Provides real-time camera feed and statistics
"""

import cv2
import numpy as np
import time
import sys
from io import StringIO
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
from collections import deque
import threading

class TerminalOutputCapture:
    """Capture terminal output and broadcast to web clients"""
    def __init__(self, socketio, max_lines=500):
        self.socketio = socketio
        self.max_lines = max_lines
        self.output_buffer = deque(maxlen=max_lines)
        self.lock = threading.Lock()
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Keywords to filter what gets sent to web interface
        self.filter_keywords = [
            'helper',           # Hotword detection
            'SPEAK',            # TTS output
            'command',          # Voice commands
            'question',         # User questions
            'query',            # User queries
            'listening',        # Listening state
            'processing',       # Processing state
            'answer',           # LLM answers
            'detected:',        # Object detections
            'Web client',       # Connection status
            'ERROR',            # Errors
            'Failed',           # Failures
            '✅',               # Status indicators
            '❌',
            '🔔',
            '📤',
            '📭',
            '🎤',
            '🗣️'
        ]
        
    def write(self, text):
        """Capture output and broadcast"""
        # Always write to original stdout
        self.original_stdout.write(text)
        self.original_stdout.flush()
        
        # Only store and broadcast important messages
        if text.strip():
            text_lower = text.lower()
            
            # Check if this line contains important keywords
            should_broadcast = any(keyword.lower() in text_lower for keyword in self.filter_keywords)
            
            if should_broadcast:
                with self.lock:
                    timestamp = time.strftime('%H:%M:%S')
                    line = f"[{timestamp}] {text.rstrip()}"
                    self.output_buffer.append(line)
                    
                    # Broadcast to all connected clients
                    try:
                        self.socketio.emit('terminal_output', {'line': line}, namespace='/')
                    except Exception as e:
                        # Silent fail - may happen before clients connect
                        pass
    
    def flush(self):
        """Required for file-like object"""
        self.original_stdout.flush()
    
    def get_buffer(self):
        """Get current buffer contents"""
        with self.lock:
            return list(self.output_buffer)

class WebInterface:
    def __init__(self, analytics_tracker, latest_frame_lock, latest_frame, 
                 status_lock, status_text):
        self.analytics = analytics_tracker
        self.latest_frame_lock = latest_frame_lock
        self.latest_frame_func = latest_frame
        self.status_lock = status_lock
        self.status_text_func = status_text
        
        # Create Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'ai_vision_assistant_secret'
        self.socketio = SocketIO(
            self.app, 
            cors_allowed_origins="*", 
            async_mode='threading',
            ping_timeout=60,
            ping_interval=25,
            logger=False,
            engineio_logger=False
        )
        
        # Terminal output capture
        self.terminal_capture = TerminalOutputCapture(self.socketio)
        
        # Stats tracking
        self.session_start = time.time()
        self.total_commands = 0
        self.total_objects = 0
        
        self.setup_routes()
        self.setup_socketio()
        
    def start_output_capture(self):
        """Start capturing terminal output"""
        sys.stdout = self.terminal_capture
        sys.stderr = self.terminal_capture
        print("✅ Terminal output capture ACTIVE - voice commands will stream to web dashboard")
    
    def stop_output_capture(self):
        """Stop capturing terminal output"""
        sys.stdout = self.terminal_capture.original_stdout
        sys.stderr = self.terminal_capture.original_stderr
        
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template('dashboard.html')
        
        @self.app.route('/video_feed')
        def video_feed():
            """Video streaming route"""
            return Response(self.generate_frames(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @self.app.route('/api/stats')
        def get_stats():
            """API endpoint for getting current stats"""
            stats = self.analytics.get_recent_stats(5)
            return jsonify({
                'total_commands': self.total_commands,
                'session_time': int(time.time() - self.session_start),
                'avg_fps': stats['avg_fps'],
                'total_objects': stats['total_recent']
            })
    
    def setup_socketio(self):
        """Setup SocketIO event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            print(f"✅ Web client connected: {request.sid}")
            emit('connection_status', {'status': 'connected'})
            
            # Send existing terminal output buffer to newly connected client
            buffer = self.terminal_capture.get_buffer()
            if buffer:
                emit('terminal_history', {'lines': buffer})
            
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            print(f"Web client disconnected: {request.sid}")
        
        @self.socketio.on('clear_terminal')
        def handle_clear_terminal():
            """Clear terminal output buffer"""
            with self.terminal_capture.lock:
                self.terminal_capture.output_buffer.clear()
            emit('terminal_cleared', broadcast=True)
    
    def generate_frames(self):
        """Generate camera frames for web streaming"""
        while True:
            try:
                # Access the global latest_frame
                import sys
                main_module = sys.modules.get('__main__')
                if main_module and hasattr(main_module, 'latest_frame'):
                    with main_module.latest_frame_lock:
                        frame = main_module.latest_frame.copy() if main_module.latest_frame is not None else None
                else:
                    frame = None

                if frame is not None:
                    # Resize frame for web display
                    frame_resized = cv2.resize(frame, (800, 600))

                    # Encode frame as JPEG
                    ret, buffer = cv2.imencode('.jpg', frame_resized, 
                                             [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    # Send a black frame if no camera frame available
                    black_frame = np.zeros((600, 800, 3), dtype='uint8')
                    ret, buffer = cv2.imencode('.jpg', black_frame)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                time.sleep(0.1)  # ~10 FPS for web

            except Exception as e:
                print(f"Error generating web frame: {e}")
                time.sleep(1)
    
    def emit_camera_stats(self, fps, status, detections):
        """Emit camera statistics to web clients"""
        try:
            self.socketio.emit('camera_stats', {
                'fps': fps,
                'status': status,
                'objects': len(detections),
                'total_objects': self.analytics.get_recent_stats(5)['total_recent']
            })
        except Exception as e:
            print(f"Error emitting camera stats: {e}")
    
    def run(self, host='0.0.0.0', port=5000):
        """Run the web interface"""
        try:
            # Use threading mode instead of eventlet to avoid conflicts with audio processing
            self.socketio.run(
                self.app, 
                host=host, 
                port=port, 
                debug=False, 
                use_reloader=False,
                allow_unsafe_werkzeug=True
            )
        except Exception as e:
            print(f"❌ Error running web interface: {e}")


# Global web interface instance
web_interface = None