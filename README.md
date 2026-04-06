# OpenAI Whisper Transcription Suite

A comprehensive collection of tools for audio/video transcription using OpenAI's Whisper model, featuring both batch processing and real-time speech-to-text capabilities.

## 🚀 Features

- **Web Interface**: Gradio-based web app for easy audio/video file upload and SRT subtitle generation
- **Real-time STT**: Live speech-to-text transcription from microphone input
- **Batch Processing**: Command-line tools for processing multiple files
- **Multi-language Support**: Automatic language detection and manual language selection
- **High Performance**: Uses faster-whisper with CUDA acceleration for fast processing
- **WebSocket API**: FastAPI server for real-time transcription over WebSocket connections
- **SRT Output**: Generates standard SRT subtitle files with precise timestamps

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for faster processing)
- FFmpeg (for audio extraction from video files)

## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/open-ai-whisper.git
cd open-ai-whisper
```

2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate the virtual environment:

```bash
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Install FFmpeg (if not already installed):
   - Download from https://ffmpeg.org/download.html
   - Add to your system PATH

## 🎯 Usage

### Web Interface (Recommended)

Run the Gradio web application:

```bash
python app.py
```

Or use the provided batch file:

```bash
start.bat
```

This will launch a web interface where you can:

- Upload audio or video files
- Select language (auto, zh, en)
- Generate SRT subtitle files

### Command Line Tools

#### Basic Transcription

```python
from transcribe import transcribe_to_srt

# Transcribe audio/video to SRT
srt_path = transcribe_to_srt("input.mp4", "zh")
```

#### Real-time Speech-to-Text

```bash
python realtime-stt/rt-app.py
```

#### Translation

```python
from faster_whisper import WhisperModel

model = WhisperModel("medium", device="cpu")
segments, info = model.transcribe("audio.wav", task="translate")
```

### API Server

Run the FastAPI server for real-time transcription:

```bash
python server/server-rt-stt.py
```

The server provides WebSocket endpoints for real-time STT with multi-user support.

## 📁 Project Structure

```
├── app.py                 # Main Gradio web application
├── transcribe.py          # Core transcription functionality
├── requirements.txt       # Python dependencies
├── start.bat             # Windows startup script
├── Demo/
│   ├── transcribe_to_srt.py  # Demo script for SRT generation
│   └── translate.py          # Demo script for translation
├── realtime-stt/
│   ├── rt-app.py            # Console real-time STT
│   ├── rt-app2.py           # Alternative real-time implementation
│   └── rt-gradio.py         # Gradio real-time STT interface
├── server/
│   └── server-rt-stt.py     # FastAPI WebSocket server
└── uploads/                # Directory for uploaded files
```

## 🔧 Configuration

### Model Selection

The project uses different Whisper models based on the use case:

- `large-v3`: Best accuracy (used in main app)
- `medium`: Balanced performance (used in demos)
- `deepdml/faster-whisper-large-v3-turbo-ct2`: Optimized version

### Device Configuration

- `device="cuda"`: Use GPU acceleration
- `device="cpu"`: Use CPU (slower but works without GPU)

### Language Options

- `"auto"`: Automatic language detection
- `"zh"`: Chinese
- `"en"`: English
- Other language codes supported by Whisper

## 🎨 Demos

### Basic SRT Generation

Located in `Demo/transcribe_to_srt.py` - processes a video file and generates SRT subtitles.

### Translation Demo

Located in `Demo/translate.py` - demonstrates translation capabilities using Whisper's translate task.

### Real-time Interfaces

Multiple real-time STT implementations in the `realtime-stt/` folder:

- Console-based real-time transcription
- Gradio web interface for real-time STT
- FastAPI server with WebSocket support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source. Please check the license file for details.

## ⚠️ Notes

- GPU acceleration significantly improves processing speed
- Large audio files may take considerable time to process
- Ensure sufficient disk space for temporary audio files
- The real-time STT features require a working microphone

## 🆘 Troubleshooting

### Common Issues

1. **CUDA errors**: Ensure CUDA is properly installed and compatible with your GPU
2. **FFmpeg not found**: Add FFmpeg to your system PATH
3. **Memory errors**: Try using a smaller model or reduce batch size
4. **Audio quality issues**: Ensure input audio is clear and properly formatted

### Performance Tips

- Use GPU acceleration when available
- Process shorter audio segments for better memory management
- Use appropriate model size based on your hardware capabilities</content>
  <parameter name="filePath">d:\Design\open-ai-whisper\README.md
