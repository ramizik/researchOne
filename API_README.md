# Multimodal Analysis REST API

This REST API provides backend services for the multimodal emotion and voice analysis system. It wraps the existing Python analysis functionality with HTTP endpoints suitable for web frontend integration.

## Quick Start

### 1. Install Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Or using conda
conda env create -f environment.yml
conda activate multimodal-analysis
```

### 2. Start the API Server

**Option 1: Using the startup script (recommended)**

```bash
# Windows
start_api.bat

# Linux/Mac
chmod +x start_api.sh
./start_api.sh

# Or directly with Python
python start_api.py
```

**Option 2: Using uvicorn directly**

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Access the API

- **API Base URL**: http://localhost:8000
- **Interactive API Documentation (Swagger)**: http://localhost:8000/api/docs
- **Alternative Documentation (ReDoc)**: http://localhost:8000/api/redoc
- **Health Check**: http://localhost:8000/api/health

## API Endpoints

### Session Management

#### `POST /api/analysis/start`
Start a new analysis session.

**Response:**
```json
{
  "session_id": "uuid-string",
  "status": "pending",
  "message": "Session created. Upload video and audio to begin analysis."
}
```

### File Upload

#### `POST /api/analysis/upload-video`
Upload video file for analysis (15 seconds).

**Parameters:**
- `session_id` (query parameter): Session ID from `/api/analysis/start`
- `file` (form data): Video file (WebM, MP4, etc.)

**Response:**
```json
{
  "session_id": "uuid-string",
  "status": "video_received",
  "message": "Video uploaded. Waiting for audio."
}
```

#### `POST /api/analysis/upload-audio`
Upload audio file for analysis (15 seconds).

**Parameters:**
- `session_id` (query parameter): Session ID from `/api/analysis/start`
- `file` (form data): Audio file (WebM, WAV, MP3, etc.)

**Response:**
```json
{
  "session_id": "uuid-string",
  "status": "audio_received",
  "message": "Audio uploaded. Analysis started."
}
```

**Note:** Analysis automatically starts when both video and audio are uploaded.

### Analysis Status & Results

#### `GET /api/analysis/status/{session_id}`
Get current analysis status.

**Response:**
```json
{
  "session_id": "uuid-string",
  "status": "processing_video" | "processing_audio" | "processing_ai" | "completed" | "failed",
  "progress": 0-100,
  "message": "Processing video for emotion detection...",
  "error": null
}
```

**Status Values:**
- `pending`: Session created, waiting for uploads
- `processing_video`: Analyzing video for emotions
- `processing_audio`: Analyzing audio for voice characteristics
- `processing_ai`: Generating AI insights
- `completed`: Analysis complete, results available
- `failed`: Analysis failed (check error field)

#### `GET /api/analysis/results/{session_id}`
Get complete analysis results (only when status is `completed`).

**Response:**
```json
{
  "metadata": {
    "session_id": "string",
    "timestamp": "ISO-8601",
    "recording_duration": 15,
    "analysis_version": "2.0",
    "status": "completed"
  },
  "emotion_analysis": {
    "emotions_by_second": {...},
    "emotional_analysis": {...},
    "facial_expression_quality": {...},
    "emotional_stability_metrics": {...}
  },
  "voice_analysis": {
    "mean_pitch": 250.5,
    "voice_type": "tenor",
    "singing_characteristics": {...},
    "emotional_indicators": {...}
  },
  "transcription_analysis": {
    "transcription": "text",
    "confidence": 0.95,
    "success": true,
    "word_count": 25
  },
  "multimodal_insights": {
    "emotional_coherence": "high",
    "overall_emotional_state": "coherent_happiness",
    "confidence_score": 0.9
  },
  "gemini_insights": "AI-generated comprehensive analysis..."
}
```

### Utility Endpoints

#### `GET /`
API information and available endpoints.

#### `GET /api/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "ISO-8601",
  "gemini_available": true
}
```

#### `GET /api/sessions`
List all active sessions.

#### `DELETE /api/analysis/session/{session_id}`
Delete a session and cleanup associated files.

## Usage Workflow

### Complete Analysis Flow

```javascript
// 1. Start a new session
const sessionResponse = await fetch('http://localhost:8000/api/analysis/start', {
  method: 'POST'
});
const { session_id } = await sessionResponse.json();

// 2. Upload video
const videoFormData = new FormData();
videoFormData.append('file', videoBlob);

await fetch(`http://localhost:8000/api/analysis/upload-video?session_id=${session_id}`, {
  method: 'POST',
  body: videoFormData
});

// 3. Upload audio
const audioFormData = new FormData();
audioFormData.append('file', audioBlob);

await fetch(`http://localhost:8000/api/analysis/upload-audio?session_id=${session_id}`, {
  method: 'POST',
  body: audioFormData
});

// 4. Poll for status
const pollStatus = setInterval(async () => {
  const statusResponse = await fetch(
    `http://localhost:8000/api/analysis/status/${session_id}`
  );
  const status = await statusResponse.json();

  console.log(`Progress: ${status.progress}%`);

  if (status.status === 'completed') {
    clearInterval(pollStatus);

    // 5. Get results
    const resultsResponse = await fetch(
      `http://localhost:8000/api/analysis/results/${session_id}`
    );
    const results = await resultsResponse.json();

    console.log('Analysis complete!', results);
  } else if (status.status === 'failed') {
    clearInterval(pollStatus);
    console.error('Analysis failed:', status.error);
  }
}, 1000); // Poll every second
```

## Configuration

### Environment Variables

- `GOOGLE_API_KEY`: Google Gemini API key (optional, for AI insights)
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to Google Cloud credentials JSON (optional, for speech transcription)

### Required Files

The following files must be present in the project directory:

1. **Emotion Detection Model**
   - `emotion-ferplus-8.onnx`

2. **Face Detection Models**
   - `RFB-320/RFB-320.prototxt`
   - `RFB-320/RFB-320.caffemodel`

3. **Credentials (Optional but Recommended)**
   - `credentials.json` - Contains Google Cloud and Gemini API keys

See [INSTALLATION_GUIDE.md](documentation/INSTALLATION_GUIDE.md) for detailed setup instructions.

## CORS Configuration

By default, the API allows requests from all origins during development:

```python
allow_origins=["*"]
```

For production, update `api_server.py` to specify allowed origins:

```python
allow_origins=[
  "http://localhost:3000",
  "https://your-frontend-domain.com"
]
```

## Error Handling

The API returns standard HTTP status codes:

- `200 OK`: Success
- `400 Bad Request`: Invalid request (e.g., requesting results before analysis is complete)
- `404 Not Found`: Session not found
- `500 Internal Server Error`: Server-side error during analysis

Error responses include details:

```json
{
  "detail": "Error message describing what went wrong"
}
```

## Performance Considerations

- **File Size**: Video and audio files should be ~15 seconds (~5-15 MB depending on quality)
- **Processing Time**: Analysis typically takes 30-60 seconds depending on:
  - Video resolution and quality
  - Audio quality
  - Whether Gemini AI analysis is enabled
  - System resources

- **Concurrent Sessions**: The API can handle multiple sessions simultaneously
- **Session Cleanup**: Temporary files are automatically cleaned up after analysis

## Development

### Running in Development Mode

```bash
# Auto-reload on code changes
python start_api.py

# Or with uvicorn directly
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

### Running in Production

```bash
# Use production-ready settings
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 4
```

### Testing the API

Use the interactive Swagger documentation at `/api/docs` to test endpoints directly in your browser.

Or use curl:

```bash
# Health check
curl http://localhost:8000/api/health

# Start session
curl -X POST http://localhost:8000/api/analysis/start

# Upload video
curl -X POST \
  "http://localhost:8000/api/analysis/upload-video?session_id=YOUR_SESSION_ID" \
  -F "file=@video.webm"

# Get status
curl http://localhost:8000/api/analysis/status/YOUR_SESSION_ID

# Get results
curl http://localhost:8000/api/analysis/results/YOUR_SESSION_ID
```

## Troubleshooting

### Common Issues

**1. "Cannot import module 'api_server'"**
- Solution: Make sure you're in the project directory and all dependencies are installed

**2. "Model files not found"**
- Solution: Download and place model files as described in INSTALLATION_GUIDE.md

**3. "Speech transcription failed"**
- Solution: Check that `credentials.json` exists and contains valid Google Cloud credentials

**4. "Gemini AI not available"**
- Solution: Install `google-generativeai` and set `GOOGLE_API_KEY` or add to `credentials.json`

**5. Port 8000 already in use**
- Solution: Change port in `start_api.py` or use: `uvicorn api_server:app --port 8080`

### Logs

The API logs all operations. Check console output for detailed information about:
- Session creation and processing
- File uploads
- Analysis progress
- Errors and warnings

## Architecture

```
Frontend (React)
    ↓ HTTP/REST
API Server (FastAPI)
    ↓
Analysis Engine
    ├── Emotion Detection (OpenCV + ONNX)
    ├── Voice Analysis (Librosa)
    ├── Speech Transcription (Google Cloud)
    └── AI Insights (Google Gemini)
```

## Related Documentation

- [Installation Guide](documentation/INSTALLATION_GUIDE.md)
- [Google Cloud Setup](documentation/google_cloud_setup.md)
- [Gemini Setup](documentation/GEMINI_SETUP.md)
- [Main README](README.md)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the documentation in the `documentation/` folder
3. Check API logs for detailed error messages
4. Use `/api/docs` to verify correct API usage
