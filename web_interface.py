"""
Web interface for AI Vision Assistant
Provides real-time camera feed, voice transcript, and question interface
"""

import cv2
import base64
import time
import threading
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import eventlet

class WebInterface:
    def __init__(self, analytics_tracker, latest_frame_lock, latest_frame, 
                 status_lock, status_text, process_advanced_command_func,
                 interrupt_processing_func):
        self.analytics = analytics_tracker
        self.latest_frame_lock = latest_frame_lock
        self.latest_frame_func = latest_frame
        self.status_lock = status_lock
        self.status_text_func = status_text
        self.process_command = process_advanced_command_func
        self.interrupt_processing = interrupt_processing_func
        
        # Create Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'ai_vision_assistant_secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='eventlet')
        
        # Stats tracking
        self.session_start = time.time()
        self.total_commands = 0
        self.total_objects = 0
        
        self.setup_routes()
        self.setup_socketio()
        
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
            print(f"Web client connected: {request.sid}")
            emit('connection_status', {'status': 'connected'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            print(f"Web client disconnected: {request.sid}")
        
        @self.socketio.on('ask_question')
        def handle_question(data):
            """Handle questions from web interface"""
            question = data.get('question', '').strip()
            if not question:
                return

            print(f"Web question received: {question}")
            self.total_commands += 1

            # Emit transcript update
            self.socketio.emit('voice_transcript', {
                'text': question,
                'type': 'user',
                'timestamp': time.time()
            })

            try:
                # Get current frame
                import sys
                main_module = sys.modules.get('__main__')
                if main_module and hasattr(main_module, 'latest_frame'):
                    with main_module.latest_frame_lock:
                        frame = main_module.latest_frame.copy() if main_module.latest_frame is not None else None
                else:
                    frame = None

                if frame is None:
                    response = "No camera frame available."
                else:
                    # Process the question
                    response = self.process_command(question, frame)

                if not response:
                    response = "I couldn't process that question."

                print(f"Web question response: {response}")

                # Emit response
                self.socketio.emit('ai_response', {
                    'text': response,
                    'timestamp': time.time()
                })

            except Exception as e:
                print(f"Error processing web question: {e}")
                self.socketio.emit('ai_response', {
                    'text': "Sorry, I encountered an error processing your question.",
                    'timestamp': time.time()
                })        @self.socketio.on('voice_command')
        def handle_voice_command(data):
            """Handle voice commands from web interface"""
            command = data.get('command', '').strip()
            if not command:
                return

            print(f"Web voice command: {command}")
            self.total_commands += 1

            # Emit transcript update
            self.socketio.emit('voice_transcript', {
                'text': f"Helper {command}",
                'type': 'user',
                'timestamp': time.time()
            })

            try:
                # Get current frame
                import sys
                main_module = sys.modules.get('__main__')
                if main_module and hasattr(main_module, 'latest_frame'):
                    with main_module.latest_frame_lock:
                        frame = main_module.latest_frame.copy() if main_module.latest_frame is not None else None
                else:
                    frame = None

                if frame is None:
                    response = "No camera frame available."
                else:
                    # Process the command
                    response = self.process_command(command, frame)

                if not response:
                    response = "I couldn't process that command."

                # Emit response
                self.socketio.emit('command_response', {
                    'response': response,
                    'timestamp': time.time()
                })

                self.socketio.emit('ai_response', {
                    'text': response,
                    'timestamp': time.time()
                })

            except Exception as e:
                print(f"Error processing web command: {e}")
                self.socketio.emit('command_response', {
                    'response': "Sorry, I encountered an error processing that command.",
                    'timestamp': time.time()
                })
        
        @self.socketio.on('stop_speech')
        def handle_stop():
            """Handle stop speech command from web interface"""
            print("Web stop command received")
            if self.interrupt_processing:
                self.interrupt_processing()
            
            self.socketio.emit('voice_transcript', {
                'text': "Stop command executed",
                'type': 'system',
                'timestamp': time.time()
            })
    
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
                    black_frame = cv2.zeros((600, 800, 3), dtype='uint8')
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
    
    def emit_voice_transcript(self, text, transcript_type='assistant'):
        """Emit voice transcript to web clients"""
        try:
            self.socketio.emit('voice_transcript', {
                'text': text,
                'type': transcript_type,
                'timestamp': time.time()
            })
        except Exception as e:
            print(f"Error emitting voice transcript: {e}")
    
    def run(self, host='0.0.0.0', port=5000):
        """Run the web interface"""
        print(f"Starting web interface on http://{host}:{port}")
        print("Web dashboard features:")
        print("  📹 Real-time camera feed")
        print("  🎤 Voice transcript display")
        print("  💬 Ask AI questions")
        print("  🗣️ Voice command buttons")
        print("  📊 Live statistics")
        
        self.socketio.run(self.app, host=host, port=port, debug=False)

# Global web interface instance
web_interface = None