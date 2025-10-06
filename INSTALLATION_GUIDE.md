# Multimodal Analysis Project - Installation Guide

This guide will help you set up the Multimodal Emotion and Voice Analysis project on your system.

## Prerequisites

- **Python 3.8 or higher**
- **Conda** (recommended) or **pip**
- **Webcam** and **microphone** access
- **Google Cloud account** (for speech transcription features)

## Installation Methods

### Method 1: Using Conda (Recommended)

This is the **recommended approach** as it handles system-level dependencies better.

#### Step 1: Create the Environment
```bash
# Create environment from the environment.yml file
conda env create -f environment.yml

# Activate the environment
conda activate multimodal-analysis
```

#### Step 2: Verify Installation
```bash
# Test the installation
python test_speech_integration.py
```

### Method 2: Using pip only

If you prefer to use pip instead of conda:

#### Step 1: Create Virtual Environment
```bash
# Create virtual environment
python -m venv multimodal-env

# Activate virtual environment
# On Windows:
multimodal-env\Scripts\activate
# On macOS/Linux:
source multimodal-env/bin/activate
```

#### Step 2: Install Dependencies
```bash
# Install from requirements.txt
pip install -r requirements.txt
```

#### Step 3: Verify Installation
```bash
# Test the installation
python test_speech_integration.py
```

## Google Cloud Setup (Required for Speech Transcription)

### Step 1: Create Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the **Speech-to-Text API**

### Step 2: Create Service Account
1. Go to **IAM & Admin** → **Service Accounts**
2. Click **Create Service Account**
3. Name it "multimodal-analysis"
4. Grant **Speech-to-Text User** role
5. Create and download the JSON key file

### Step 3: Set Up Credentials
```bash
# Option 1: Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"

# Option 2: Place credentials.json in project directory
# Copy your downloaded JSON file to the project root as "credentials.json"
```

## Running the Project

### Basic Usage
```bash
# Run the main multimodal analysis
python multimodal_analysis.py
```

### Test Individual Components
```bash
# Test voice analysis only
python voice_analyzer.py

# Test speech transcription only
python speech_transcriber.py audio_file.wav

# Test integration
python test_speech_integration.py
```

## Troubleshooting

### Common Issues

#### 1. PyAudio Installation Issues
**Problem**: `pip install pyaudio` fails
**Solution**:
```bash
# On Windows with conda:
conda install -c conda-forge pyaudio

# On macOS:
brew install portaudio
pip install pyaudio

# On Ubuntu/Debian:
sudo apt-get install portaudio19-dev
pip install pyaudio
```

#### 2. OpenCV Issues
**Problem**: Camera not detected
**Solution**:
```bash
# Try different backends
conda install -c conda-forge opencv
# or
pip install opencv-python-headless
```

#### 3. Google Cloud Authentication
**Problem**: "Credentials not found" error
**Solution**:
1. Verify `GOOGLE_APPLICATION_CREDENTIALS` points to valid JSON file
2. Check that the service account has Speech-to-Text permissions
3. Ensure the JSON file is not corrupted

#### 4. Audio Recording Issues
**Problem**: "No audio detected" error
**Solution**:
1. Check microphone permissions
2. Test microphone with other applications
3. Try different audio devices
4. Increase recording volume

### Performance Optimization

#### For Better Performance:
```bash
# Install optional performance packages
conda install -c conda-forge numba
# or
pip install numba scikit-learn
```

#### For Development:
```bash
# Install development tools
pip install pytest black flake8
```

## Project Structure

```
multimodal-analysis/
├── multimodal_analysis.py      # Main application
├── voice_analyzer.py           # Voice analysis module
├── speech_transcriber.py       # Speech transcription module
├── test_speech_integration.py  # Integration tests
├── environment.yml             # Conda environment file
├── requirements.txt            # Pip requirements
├── google_cloud_setup.md       # Detailed Google Cloud setup
├── INSTALLATION_GUIDE.md      # This file
├── emotion-ferplus-8.onnx     # Emotion detection model
├── RFB-320/                   # Face detection models
│   ├── RFB-320.prototxt
│   └── RFB-320.caffemodel
└── credentials.json           # Google Cloud credentials (you need to add this)
```

## Usage Tips

### For Best Results:
1. **Good lighting** for emotion detection
2. **Clear audio** with minimal background noise
3. **Speak clearly** for better transcription
4. **Stable internet** for Google Cloud API calls

### Recording Tips:
- Speak at normal volume, 6-12 inches from microphone
- Ensure good lighting on your face
- Minimize background noise
- Keep still during recording for better emotion detection

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Run `python test_speech_integration.py` to diagnose issues
3. Check the `google_cloud_setup.md` file for detailed Google Cloud setup
4. Ensure all dependencies are properly installed

## Next Steps After Installation

1. **Test the setup**: Run `python test_speech_integration.py`
2. **Try the main application**: Run `python multimodal_analysis.py`
3. **Customize settings**: Modify parameters in the Python files as needed
4. **Integrate with your projects**: Import the modules into your own code

Happy analyzing! 🎭🎤📝
