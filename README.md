# AI Vision Assistant for the Visually Impaired 👁️🤖

An intelligent assistive technology system designed to empower visually impaired individuals through real-time object detection, navigation assistance, and AI-powered voice interaction. This software can be integrated into smart glasses to provide seamless, hands-free assistance in daily life.

## 🌟 Features

### 1. **Real-Time Object Detection**
- Powered by YOLOv8x for accurate and fast object recognition
- Detects and identifies objects in the user's field of view
- Provides audio feedback about detected objects
- Optimized for various environments (indoor/outdoor)

### 2. **Navigation & Path Guidance**
- Assists users in navigating their surroundings
- Provides directional guidance to locations
- Real-time spatial awareness through AI vision

### 3. **AI Voice Assistant**
- Natural voice interaction using speech recognition
- Ask any question and receive intelligent responses
- Powered by Groq LLM for accurate and contextual answers
- Hotword activation ("helper") for hands-free operation
- Text-to-speech feedback for all interactions

### 4. **Modern User Interfaces**
- **Desktop Dashboard**: Real-time statistics, detection visualizations, and system monitoring
- **Web Interface**: Browser-based control panel with live camera feed
- **Terminal Interface**: Lightweight command-line operation

### 5. **Smart Features**
- Hotword detection for voice activation
- Real-time camera feed processing
- Detection history and statistics
- Multi-threaded performance optimization
- Support for macOS (MPS), GPU, and CPU acceleration

## 🛠️ Technology Stack

- **Computer Vision**: OpenCV, YOLOv8 (Ultralytics)
- **Deep Learning**: PyTorch with MPS/CUDA support
- **Speech Recognition**: Vosk, SpeechRecognition, PyAudio
- **AI/LLM**: Groq API
- **Text-to-Speech**: pyttsx3
- **Web Framework**: Flask, Flask-SocketIO
- **UI**: CustomTkinter, Matplotlib, Seaborn
- **Language**: Python 3.8+

## 📋 Prerequisites

- Python 3.8 or higher
- Webcam or camera device
- Microphone for voice commands
- Internet connection (for AI assistant features)
- Groq API key (for LLM functionality)

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Sarvesh0955/dti.git
cd dti
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Required Models

The YOLOv8x model (`yolov8x.pt`) should be present in the project directory. If not, it will be downloaded automatically on first run.

The Vosk speech recognition model is included in the `vosk-model-en-in-0.5/` directory.

### 5. Set Up Environment Variables

Create a `.env` file in the project root:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

Get your Groq API key from [https://console.groq.com](https://console.groq.com)

## 💻 Usage

### Running the Main Application

```bash
python app.py
```

This launches the full application with:
- Object detection system
- Voice assistant with hotword detection
- Modern dashboard UI with real-time statistics
- Web interface (accessible at `http://localhost:5000`)

### Running Web Interface Only

```bash
python web_interface.py
```

Access the web dashboard at `http://localhost:5000` in your browser.

### Keyboard Shortcuts

- **Spacebar**: Manually trigger object detection
- **Q**: Quit application
- **ESC**: Close windows

### Voice Commands

1. Say **"helper"** to activate the voice assistant
2. Ask your question or give a command
3. The system will process your request and respond with audio

## 📱 Interface Overview

### Desktop Dashboard
- Real-time camera feed with object detection overlays
- Live statistics: FPS, detection count, processing time
- Detection history visualization
- Object frequency charts
- System performance metrics

### Web Interface
- Browser-based control panel
- Live camera stream
- Real-time terminal output
- Detection statistics
- No installation required on client devices

### Terminal Output
- Timestamped logs
- Detection events
- Voice command transcriptions
- System status updates

## 🎯 Use Cases

1. **Daily Navigation**: Identify objects and obstacles in real-time
2. **Shopping**: Recognize products and read labels
3. **Social Interactions**: Identify people and surroundings
4. **Information Access**: Ask questions about your environment
5. **Independence**: Perform daily tasks with AI assistance

## 🏗️ Project Structure

```
dti/
├── app.py                      # Main application entry point
├── web_interface.py            # Flask web server
├── requirements.txt            # Python dependencies
├── yolov8x.pt                 # YOLOv8 model weights
├── vosk-model-en-in-0.5/      # Speech recognition model
├── templates/
│   └── dashboard.html         # Web interface template
└── README.md                  # Project documentation
```

## 🔧 Configuration

### Camera Settings
Modify camera index in `app.py`:
```python
cap = cv2.VideoCapture(0)  # Change 0 to your camera index
```

### Detection Confidence
Adjust YOLOv8 confidence threshold:
```python
results = model(frame, conf=0.5)  # Change 0.5 to desired confidence
```

### Voice Recognition
Configure Vosk model path if using a different model:
```python
model = vosk.Model("path/to/your/vosk-model")
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🐛 Troubleshooting

### Camera Not Working
- Check camera permissions in system settings
- Try different camera indices (0, 1, 2, etc.)
- Ensure no other application is using the camera

### Audio Issues
- Verify microphone permissions
- Check PyAudio installation
- Test system audio output

### Model Loading Errors
- Ensure YOLOv8 model file is present
- Check internet connection for model download
- Verify sufficient disk space

### API Errors
- Confirm Groq API key is correctly set in `.env`
- Check internet connectivity
- Verify API key is active and has quota

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics for object detection
- Vosk for offline speech recognition
- Groq for fast LLM inference
- OpenCV community for computer vision tools
- All contributors and supporters of assistive technology

## 📞 Support

For support, please open an issue in the GitHub repository or contact the maintainers.

---

**Made with ❤️ for accessibility and inclusion**
