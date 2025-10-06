## Multimodal Emotion and Voice Analysis

This repository contains tools for analyzing facial emotions and voice characteristics.

### Scripts

#### 1. `expression_ssd_detect.py`
Records 15 seconds of video and analyzes facial emotions.
```bash
python expression_ssd_detect.py
```

#### 2. `voice_analyzer.py`
Records and analyzes voice characteristics.
```bash
python voice_analyzer.py
```

#### 3. `multimodal_analysis.py` 🌟
Simultaneously records video and audio for 15 seconds, providing comprehensive multimodal analysis.
```bash
python multimodal_analysis.py
```

### Features

**Emotion Detection:**
- Real-time facial emotion detection (happiness, sadness, anger, fear, surprise, disgust, neutral)
- Per-second emotion tracking
- Emotional stability analysis
- Timeline visualization

**Voice Analysis:**
- Pitch detection and voice type classification
- Vibrato rate analysis
- Voice quality metrics (jitter, shimmer)
- Dynamic range analysis
- Vocal stability assessment

**Multimodal Analysis:**
- Synchronized video and audio recording
- Cross-modal correlation insights
- Comprehensive CLI results presentation

### Installation

#### Quick Start (Conda - Recommended):
```bash
conda env create -f environment.yml
conda activate multimodal-analysis
```

#### Alternative (Pip):
```bash
pip install -r requirements.txt
```

See `setup_instructions.md` for detailed installation guide and troubleshooting.
