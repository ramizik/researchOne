# Multimodal Emotion and Voice Analysis

A comprehensive tool for analyzing facial emotions and voice characteristics with AI-powered insights using Google Gemini.

## Features

### 🎭 Facial Emotion Detection
- Real-time emotion detection (happiness, sadness, anger, fear, surprise, disgust, neutral)
- Per-second emotion tracking and analysis
- Emotional stability and volatility metrics
- Facial expression quality assessment

### 🎤 Voice Analysis
- Advanced pitch detection and voice type classification
- Singing characteristics analysis (style, quality, vibrato)
- Emotional voice indicators (arousal, tension, energy)
- Voice quality metrics (jitter, shimmer, breath control)
- Speaking rate and intensity analysis

### 🤖 AI-Powered Insights
- Google Gemini integration for comprehensive psychological analysis
- Multimodal coherence assessment
- Professional recommendations and insights
- Cross-modal correlation analysis

### 📝 Speech Transcription
- Google Cloud Speech-to-Text integration
- Word-level timing and confidence scores
- Language detection and enhanced models

## Quick Start

### Prerequisites
- Python 3.8+
- Webcam and microphone
- Google Cloud account (for transcription)
- Google AI Studio account (for Gemini analysis)

### Installation

#### Option 1: Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate multimodal-analysis
```

#### Option 2: Pip
```bash
python -m venv multimodal-env
# Windows:
multimodal-env\Scripts\activate
# macOS/Linux:
source multimodal-env/bin/activate

pip install -r requirements.txt
```

### Setup

1. **Download model files** (place in project directory):
   - `emotion-ferplus-8.onnx` (emotion detection model)
   - `RFB-320/RFB-320.prototxt` (face detection model)
   - `RFB-320/RFB-320.caffemodel` (face detection weights)

2. **Configure Google Cloud** (for speech transcription):
   - Follow [Google Cloud Setup Guide](documentation/google_cloud_setup.md)
   - Place `credentials.json` in project directory

3. **Configure Gemini AI** (for AI analysis):
   - Follow [Gemini Setup Guide](documentation/GEMINI_SETUP.md)
   - Set `GOOGLE_API_KEY` environment variable or add to `credentials.json`

### Usage

### CLI Mode (Command Line Interface)

```bash
# Run complete multimodal analysis
python multimodal_analysis.py

# Test individual components
python voice_analyzer.py
python expression_ssd_detect.py
```

### API Mode (REST API Server)

```bash
# Start the REST API server
python start_api.py

# Or use the startup scripts
# Windows:
start_api.bat

# Linux/Mac:
./start_api.sh
```

The API server will start on http://localhost:8000

- **API Documentation**: http://localhost:8000/api/docs
- **Alternative Docs**: http://localhost:8000/api/redoc

See [API_README.md](API_README.md) for complete API documentation and usage examples.

## Error Handling

The system includes comprehensive error handling with graceful shutdown. See [Error Handling Guide](documentation/ERROR_HANDLING.md) for detailed troubleshooting.

### Common Issues
- **Camera errors**: Check camera connection and permissions
- **Audio errors**: Verify microphone access and quality
- **Model errors**: Ensure all model files are present
- **API errors**: Check Google Cloud and Gemini configurations

## Output

The system generates:
- Real-time analysis display
- Comprehensive JSON output for LLM integration
- AI-powered insights and recommendations
- Detailed error messages with troubleshooting steps

## Documentation

- [Installation Guide](documentation/INSTALLATION_GUIDE.md) - Detailed setup instructions
- [Google Cloud Setup](documentation/google_cloud_setup.md) - Speech-to-Text configuration
- [Gemini Setup](documentation/GEMINI_SETUP.md) - AI analysis configuration
- [Error Handling](documentation/ERROR_HANDLING.md) - Troubleshooting guide

## Project Structure

```
multimodal-analysis/
├── multimodal_analysis.py      # Main CLI application
├── voice_analyzer.py           # Voice analysis module
├── speech_transcriber.py       # Speech transcription
├── expression_ssd_detect.py    # Emotion detection
├── api_server.py              # REST API server (FastAPI)
├── analysis_engine.py         # Analysis engine wrapper for API
├── start_api.py               # API startup script
├── test_api_client.py         # API test client
├── requirements.txt           # Dependencies
├── environment.yml            # Conda environment
├── API_README.md              # API documentation
├── documentation/             # Setup guides
└── model files/              # Required AI models
```

## Requirements

- Python 3.8+
- OpenCV, NumPy, SciPy
- Librosa, PyAudio
- ONNX Runtime
- Google Cloud Speech-to-Text
- Google Generative AI (Gemini)

## License

This project is for research and educational purposes.