# Setup Instructions

## Option 1: Using Conda (Recommended)

### Create and activate environment:
```bash
# Create environment from yml file
conda env create -f environment.yml

# Activate environment
conda activate multimodal-analysis
```

### Alternative conda commands:
```bash
# Create environment manually
conda create -n multimodal-analysis python=3.9

# Activate environment
conda activate multimodal-analysis

# Install packages
conda install numpy opencv scipy librosa soundfile onnxruntime
pip install pyaudio
```

## Option 2: Using Pip

### Create virtual environment:
```bash
# Create virtual environment
python -m venv multimodal-env

# Activate environment
# On Windows:
multimodal-env\Scripts\activate
# On macOS/Linux:
source multimodal-env/bin/activate

# Install requirements
pip install -r requirements.txt
```

## Option 3: System-wide Installation

```bash
pip install -r requirements.txt
```

## Troubleshooting

### PyAudio Installation Issues:

**Windows:**
```bash
# If pyaudio fails, try:
pip install pipwin
pipwin install pyaudio
```

**macOS:**
```bash
# Install portaudio first
brew install portaudio
pip install pyaudio
```

**Linux (Ubuntu/Debian):**
```bash
# Install system dependencies
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio
```

### OpenCV Issues:
```bash
# If opencv-python fails, try:
pip install opencv-python-headless
```

## Verify Installation

Run this test script to verify all packages are working:

```python
# test_installation.py
try:
    import cv2
    import numpy as np
    import librosa
    import pyaudio
    import scipy
    import onnxruntime
    print("✅ All packages installed successfully!")
except ImportError as e:
    print(f"❌ Missing package: {e}")
```

## Model Files Required

Make sure you have these files in your project directory:
- `emotion-ferplus-8.onnx` (emotion detection model)
- `RFB-320/RFB-320.prototxt` (face detection model)
- `RFB-320/RFB-320.caffemodel` (face detection weights)

## Running the Scripts

```bash
# Emotion detection only
python expression_ssd_detect.py

# Voice analysis only
python voice_analyzer.py

# Multimodal analysis (recommended)
python multimodal_analysis.py
```
