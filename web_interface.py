"""
Web interface for AI Vision Assistant
Provides real-time camera feed and statistics
"""

import cv2
import numpy as np
import time
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import eventlet

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
        print(f"Starting web interface on http://{host}:{port}")
        print("Web dashboard features:")
        print("  📹 Real-time camera feed")
        print("  📊 Live statistics")
        
        self.socketio.run(self.app, host=host, port=port, debug=False)

# Global web interface instance
web_interface = None