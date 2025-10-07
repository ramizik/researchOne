# Error Handling Guide

This document describes the comprehensive error handling system implemented in the Multimodal Analysis Tool.

## Overview

The system now implements proper error handling with graceful shutdown instead of fallback mechanisms. When any critical component fails, the system will:

1. **Identify the specific error type**
2. **Display a clear error message**
3. **Provide troubleshooting steps**
4. **Gracefully shutdown the application**

## Error Types

### 1. Camera Errors (`CameraError`)
**Triggers when:**
- Camera cannot be opened
- Camera is already in use by another application
- Camera permissions are denied

**Error Message Example:**
```
❌ CAMERA ERROR: Cannot open camera. Please check if camera is connected and not being used by another application.

💡 Troubleshooting steps:
   • Check if camera is connected
   • Close other applications using the camera
   • Check camera permissions
   • Try restarting the application
```

### 2. Model Errors (`ModelError`)
**Triggers when:**
- Required ONNX model files are missing
- Face detection model files are missing
- Model files are corrupted

**Error Message Example:**
```
❌ MODEL ERROR: Missing ONNX model: /path/to/emotion-ferplus-8.onnx

💡 Troubleshooting steps:
   • Ensure emotion-ferplus-8.onnx is in the project directory
   • Ensure RFB-320 folder with .prototxt and .caffemodel files exists
   • Check file permissions
```

### 3. Audio Errors (`AudioError`)
**Triggers when:**
- Audio recording fails
- Microphone is not available
- Audio system errors

**Error Message Example:**
```
❌ AUDIO ERROR: Audio analysis failed: [specific error details]

💡 Troubleshooting steps:
   • Check if microphone is connected and working
   • Check microphone permissions
   • Close other applications using the microphone
   • Try speaking louder and closer to the microphone
```

### 4. Voice Analysis Errors (`VoiceAnalysisError`, `InsufficientDataError`, `AudioQualityError`)
**Triggers when:**
- Insufficient audio data for analysis
- Poor audio quality (too quiet, too loud, clipping)
- Voice analysis algorithm failures

**Error Message Example:**
```
❌ VOICE ANALYSIS ERROR: Insufficient pitch data for analysis. Please try: 1) Speak louder and closer to the microphone, 2) Try singing a sustained note (like 'ahhh') instead of talking, 3) Ensure there's minimal background noise, 4) Try a longer recording duration

💡 Troubleshooting steps:
   • Check audio quality
   • Try speaking louder
   • Ensure minimal background noise
   • Try a longer recording
   • Check microphone connection and permissions
```

### 5. Gemini AI Errors (`GeminiError`)
**Triggers when:**
- Google Generative AI package not installed
- API key not configured
- Network connectivity issues
- API quota exceeded

**Error Message Example:**
```
❌ GEMINI AI ERROR: Google API key not found. Please set GOOGLE_API_KEY environment variable or add 'api_key' to credentials.json

💡 Troubleshooting steps:
   • Install: pip install google-generativeai
   • Set up API key (see GEMINI_SETUP.md)
   • Check internet connection
   • Verify API key is valid
```

### 6. Transcription Errors (`TranscriptionError`)
**Triggers when:**
- Google Cloud Speech-to-Text service fails
- Invalid credentials
- Network connectivity issues
- Audio format not supported

**Error Message Example:**
```
❌ TRANSCRIPTION ERROR: [specific error details]

💡 Troubleshooting steps:
   • Check Google Cloud credentials
   • Verify internet connection
   • Check microphone quality
   • Try speaking more clearly
```

## Error Handling Behavior

### Before (Fallback Mechanisms)
- ❌ Returned mock/simulated data
- ❌ Continued with incomplete analysis
- ❌ Provided misleading results
- ❌ No clear indication of failures

### After (Proper Error Handling)
- ✅ Clear error identification
- ✅ Specific troubleshooting guidance
- ✅ Graceful application shutdown
- ✅ No misleading results

## Implementation Details

### Exception Hierarchy
```python
MultimodalAnalysisError (Base)
├── CameraError
├── AudioError
├── ModelError
├── GeminiError
├── TranscriptionError
└── VoiceAnalysisError
    ├── InsufficientDataError
    └── AudioQualityError
```

### Error Flow
1. **Detection**: Specific error conditions are detected
2. **Classification**: Error is classified into appropriate exception type
3. **Propagation**: Exception is raised and propagated up the call stack
4. **Handling**: Main function catches and handles each exception type
5. **User Notification**: Clear error message and troubleshooting steps displayed
6. **Graceful Shutdown**: Application exits with appropriate exit code

## Best Practices

### For Users
1. **Read error messages carefully** - They contain specific guidance
2. **Follow troubleshooting steps** - They address the most common issues
3. **Check system requirements** - Ensure all dependencies are installed
4. **Verify hardware** - Check camera and microphone connections

### For Developers
1. **Use specific exception types** - Don't use generic exceptions
2. **Provide clear error messages** - Include actionable guidance
3. **Include troubleshooting steps** - Help users resolve issues
4. **Test error conditions** - Ensure error handling works correctly

## Testing Error Conditions

You can test different error conditions by:

1. **Camera Errors**: Disconnect camera or run another camera application
2. **Model Errors**: Move or rename model files
3. **Audio Errors**: Disconnect microphone or mute audio
4. **Gemini Errors**: Remove API key or disconnect internet
5. **Transcription Errors**: Remove Google Cloud credentials

## Exit Codes

- `0`: Successful completion
- `1`: Error occurred (any of the above error types)

## Recovery

After fixing the underlying issue:
1. Restart the application
2. Follow any remaining troubleshooting steps
3. Verify the fix by running the analysis again

This error handling system ensures users get clear feedback when something goes wrong and know exactly how to fix it, rather than receiving misleading results from fallback mechanisms.
