#!/usr/bin/env python3
"""
Simple test to verify SocketIO server works
"""
import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit
import time
import threading

app = Flask(__name__)
app.config['SECRET_KEY'] = 'test_secret'
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode='eventlet',
    ping_timeout=60,
    ping_interval=25,
    logger=True,
    engineio_logger=True
)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>SocketIO Test</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.js"></script>
</head>
<body>
    <h1>SocketIO Connection Test</h1>
    <div id="status">Connecting...</div>
    <div id="messages" style="border:1px solid #ccc; padding:10px; margin:10px 0; height:300px; overflow-y:auto;"></div>
    
    <script>
        const socket = io({
            transports: ['websocket', 'polling'],
            timeout: 60000
        });
        
        socket.on('connect', function() {
            console.log('Connected!');
            document.getElementById('status').textContent = 'Connected ✅';
            document.getElementById('status').style.color = 'green';
        });
        
        socket.on('disconnect', function() {
            console.log('Disconnected');
            document.getElementById('status').textContent = 'Disconnected ❌';
            document.getElementById('status').style.color = 'red';
        });
        
        socket.on('test_message', function(data) {
            console.log('Received:', data);
            const messages = document.getElementById('messages');
            messages.innerHTML += '<div>' + data.message + '</div>';
            messages.scrollTop = messages.scrollHeight;
        });
        
        socket.on('connect_error', function(error) {
            console.error('Connection error:', error);
            document.getElementById('status').textContent = 'Connection Error: ' + error;
            document.getElementById('status').style.color = 'red';
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('test_message', {'message': 'Connection established!'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

def send_messages():
    """Send test messages every 2 seconds"""
    time.sleep(2)  # Wait for server to start
    count = 0
    while True:
        try:
            count += 1
            socketio.emit('test_message', {'message': f'Test message #{count} at {time.strftime("%H:%M:%S")}'})
            print(f'Sent message #{count}')
            time.sleep(2)
        except Exception as e:
            print(f'Error sending message: {e}')
            break

if __name__ == '__main__':
    print('='*60)
    print('Starting SocketIO Test Server')
    print('Open http://localhost:5002 in your browser')
    print('='*60)
    
    # Start background thread to send messages
    msg_thread = threading.Thread(target=send_messages, daemon=True)
    msg_thread.start()
    
    # Run the server
    socketio.run(app, host='0.0.0.0', port=5002, debug=True, allow_unsafe_werkzeug=True)
