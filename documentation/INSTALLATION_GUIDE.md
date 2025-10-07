# Installation Guide

Complete setup instructions for the Multimodal Emotion and Voice Analysis project.

## Prerequisites

- **Python 3.8 or higher**
- **Webcam and microphone** access
- **Google Cloud account** (for speech transcription)
- **Google AI Studio account** (for Gemini AI analysis)

## Installation Methods

### Method 1: Using Conda (Recommended)

```bash
# Create environment from the environment.yml file
conda env create -f environment.yml

# Activate the environment
conda activate multimodal-analysis
```

### Method 2: Using pip

```bash
# Create virtual environment
python -m venv multimodal-env

# Activate virtual environment
# On Windows:
multimodal-env\Scripts\activate
# On macOS/Linux:
source multimodal-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Required Model Files

Download and place these files in the project directory:

1. **Emotion Detection Model**: `emotion-ferplus-8.onnx`
2. **Face Detection Models**:
   - `RFB-320/RFB-320.prototxt`
   - `RFB-320/RFB-320.caffemodel`

## Service Configuration

### Google Cloud Speech-to-Text Setup

1. **Create Google Cloud Project**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create new project or select existing
   - Enable **Speech-to-Text API**

2. **Create Service Account**:
   - Go to **IAM & Admin** → **Service Accounts**
   - Create service account with **Speech-to-Text User** role
   - Download JSON key file as `credentials.json`

3. **Place credentials** in project directory

### Gemini AI Setup

1. **Get API Key**:
   - Go to [Google AI Studio](https://aistudio.google.com/)
   - Create API key

2. **Configure API Key**:
   ```bash
   # Option 1: Environment variable
   export GOOGLE_API_KEY="your_api_key_here"
   
   # Option 2: Add to credentials.json
   # Add "api_key": "your_api_key_here" to credentials.json
   ```

## Verification

Test your installation:

```bash
# Run the main application
python multimodal_analysis.py

# Test individual components
python voice_analyzer.py
python expression_ssd_detect.py
```

## Troubleshooting

### PyAudio Installation Issues

**Windows:**
```bash
conda install -c conda-forge pyaudio
# or
pip install pipwin
pipwin install pyaudio
```

**macOS:**
```bash
brew install portaudio
pip install pyaudio
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install portaudio19-dev
pip install pyaudio
```

### OpenCV Issues
```bash
# Try different backends
conda install -c conda-forge opencv
# or
pip install opencv-python-headless
```

### Common Errors

- **Camera Error**: Check camera connection and permissions
- **Audio Error**: Verify microphone access and quality
- **Model Error**: Ensure all model files are present
- **API Error**: Check Google Cloud and Gemini configurations

See [Error Handling Guide](ERROR_HANDLING.md) for detailed troubleshooting.

## Performance Optimization

### Optional Performance Packages
```bash
# For better performance
conda install -c conda-forge numba
pip install scikit-learn
```

### Development Tools
```bash
# For development
pip install pytest black flake8
```

## Usage Tips

### For Best Results:
1. **Good lighting** for emotion detection
2. **Clear audio** with minimal background noise
3. **Speak clearly** for better transcription
4. **Stable internet** for API calls

### Recording Tips:
- Speak at normal volume, 6-12 inches from microphone
- Ensure good lighting on your face
- Minimize background noise
- Keep still during recording for better emotion detection

## Next Steps

1. **Test the setup**: Run `python multimodal_analysis.py`
2. **Configure services**: Set up Google Cloud and Gemini AI
3. **Customize settings**: Modify parameters as needed
4. **Integrate**: Use modules in your own projects

## Support

If you encounter issues:
1. Check the [Error Handling Guide](ERROR_HANDLING.md)
2. Verify all dependencies are installed
3. Ensure all required files are present
4. Check service configurations

Happy analyzing! 🎭🎤📝