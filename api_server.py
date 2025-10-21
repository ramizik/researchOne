"""
FastAPI REST API Server for Multimodal Emotion and Voice Analysis
Provides REST endpoints for the React frontend to interact with the analysis backend
"""

import os
import sys
import uuid
import json
import logging
import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from enum import Enum

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn

# Import the existing analysis modules
from multimodal_analysis import (
    record_video_emotions,
    record_audio_analysis_with_transcription,
    AnalysisMemory,
    get_gemini_analysis,
    GEMINI_AVAILABLE
)
from voice_analyzer import VoiceAnalyzer, VoiceAnalysisError, InsufficientDataError, AudioQualityError
from speech_transcriber import SpeechTranscriber

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multimodal Analysis API",
    description="REST API for facial emotion detection, voice analysis, and speech transcription",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware - allow all origins during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Data Models ====================

class AnalysisStatus(str, Enum):
    PENDING = "pending"
    PROCESSING_VIDEO = "processing_video"
    PROCESSING_AUDIO = "processing_audio"
    PROCESSING_AI = "processing_ai"
    COMPLETED = "completed"
    FAILED = "failed"

class SessionResponse(BaseModel):
    session_id: str
    status: str
    message: str

class UploadResponse(BaseModel):
    session_id: str
    status: str
    message: str

class StatusResponse(BaseModel):
    session_id: str
    status: str
    progress: int
    message: Optional[str] = None
    error: Optional[str] = None

# ==================== Session Management ====================

class SessionManager:
    """Manage analysis sessions and their data"""

    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.temp_dir = Path(tempfile.gettempdir()) / "multimodal_analysis"
        self.temp_dir.mkdir(exist_ok=True)
        logger.info(f"Session manager initialized. Temp directory: {self.temp_dir}")

    def create_session(self) -> str:
        """Create a new analysis session"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "status": AnalysisStatus.PENDING,
            "created_at": datetime.now().isoformat(),
            "progress": 0,
            "video_received": False,
            "audio_received": False,
            "error": None
        }
        logger.info(f"Created new session: {session_id}")
        return session_id

    def update_status(self, session_id: str, status: AnalysisStatus, progress: int, message: str = None, error: str = None):
        """Update session status"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")

        self.sessions[session_id]["status"] = status
        self.sessions[session_id]["progress"] = progress
        self.sessions[session_id]["updated_at"] = datetime.now().isoformat()

        if message:
            self.sessions[session_id]["message"] = message
        if error:
            self.sessions[session_id]["error"] = error

        logger.info(f"Session {session_id}: {status} ({progress}%)")

    def get_status(self, session_id: str) -> Dict[str, Any]:
        """Get session status"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        return self.sessions[session_id]

    def store_results(self, session_id: str, results: Dict[str, Any]):
        """Store analysis results"""
        self.results[session_id] = results
        logger.info(f"Stored results for session {session_id}")

    def get_results(self, session_id: str) -> Dict[str, Any]:
        """Get analysis results"""
        if session_id not in self.results:
            raise ValueError(f"Results for session {session_id} not found")
        return self.results[session_id]

    def mark_video_received(self, session_id: str):
        """Mark that video has been received"""
        if session_id in self.sessions:
            self.sessions[session_id]["video_received"] = True

    def mark_audio_received(self, session_id: str):
        """Mark that audio has been received"""
        if session_id in self.sessions:
            self.sessions[session_id]["audio_received"] = True

    def is_ready_for_processing(self, session_id: str) -> bool:
        """Check if session has both video and audio"""
        if session_id not in self.sessions:
            return False
        session = self.sessions[session_id]
        return session["video_received"] and session["audio_received"]

    def get_video_path(self, session_id: str) -> Path:
        """Get path for storing video file"""
        return self.temp_dir / f"{session_id}_video.webm"

    def get_audio_path(self, session_id: str) -> Path:
        """Get path for storing audio file"""
        return self.temp_dir / f"{session_id}_audio.webm"

    def cleanup_session(self, session_id: str):
        """Clean up session files"""
        try:
            video_path = self.get_video_path(session_id)
            audio_path = self.get_audio_path(session_id)

            if video_path.exists():
                video_path.unlink()
            if audio_path.exists():
                audio_path.unlink()

            logger.info(f"Cleaned up session files for {session_id}")
        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {e}")

# Global session manager instance
session_manager = SessionManager()

# ==================== Analysis Functions ====================

async def process_video_emotions(session_id: str, video_path: Path) -> Dict[str, Any]:
    """Process video file for emotion detection"""
    try:
        logger.info(f"Processing video for session {session_id}")

        # Use the emotion analysis engine
        from analysis_engine import get_emotion_engine
        engine = get_emotion_engine()

        # Analyze video
        emotion_data = engine.analyze_video(video_path)

        logger.info(f"Video processing complete for session {session_id}")
        return emotion_data

    except Exception as e:
        logger.error(f"Error processing video: {e}", exc_info=True)
        raise

async def process_audio_analysis(session_id: str, audio_path: Path) -> Dict[str, Any]:
    """Process audio file for voice analysis and transcription"""
    try:
        logger.info(f"Processing audio for session {session_id}")

        import librosa

        # Load audio file
        audio_data, sample_rate = librosa.load(str(audio_path), sr=22050)

        # Initialize analyzers
        voice_analyzer = VoiceAnalyzer()

        # Analyze voice
        voice_results = voice_analyzer.analyze_recorded_audio(audio_data, sample_rate)

        # Transcribe speech
        try:
            transcriber = SpeechTranscriber(credentials_path="credentials.json")
            transcription_results = transcriber.transcribe_audio_data(audio_data, sample_rate)
        except Exception as e:
            logger.warning(f"Speech transcription failed: {e}")
            transcription_results = {
                "transcription": "",
                "confidence": 0.0,
                "success": False,
                "error": str(e),
                "word_count": 0
            }

        # Combine results
        combined_results = {
            **voice_results,
            "transcription": transcription_results
        }

        logger.info(f"Audio processing complete for session {session_id}")
        return combined_results

    except Exception as e:
        logger.error(f"Error processing audio: {e}", exc_info=True)
        raise

async def run_full_analysis(session_id: str):
    """Run complete multimodal analysis"""
    try:
        session_manager.update_status(
            session_id,
            AnalysisStatus.PROCESSING_VIDEO,
            10,
            "Processing video for emotion detection..."
        )

        # Process video
        video_path = session_manager.get_video_path(session_id)
        emotion_data = await process_video_emotions(session_id, video_path)

        session_manager.update_status(
            session_id,
            AnalysisStatus.PROCESSING_AUDIO,
            40,
            "Processing audio for voice analysis..."
        )

        # Process audio
        audio_path = session_manager.get_audio_path(session_id)
        voice_data = await process_audio_analysis(session_id, audio_path)

        session_manager.update_status(
            session_id,
            AnalysisStatus.PROCESSING_AI,
            70,
            "Generating AI insights..."
        )

        # Generate multimodal insights
        from multimodal_analysis import AnalysisMemory
        memory = AnalysisMemory()
        memory.store_emotion_data(emotion_data)
        memory.store_voice_data(voice_data)
        memory.store_transcription_data(voice_data.get("transcription", {}))

        complete_results = memory.get_complete_analysis()

        # Get Gemini analysis if available
        if GEMINI_AVAILABLE:
            try:
                gemini_insights = get_gemini_analysis(emotion_data, voice_data)
                complete_results["gemini_insights"] = gemini_insights
            except Exception as e:
                logger.warning(f"Gemini analysis failed: {e}")
                complete_results["gemini_insights"] = None

        # Store results
        session_manager.store_results(session_id, complete_results)

        session_manager.update_status(
            session_id,
            AnalysisStatus.COMPLETED,
            100,
            "Analysis completed successfully"
        )

        # Cleanup
        session_manager.cleanup_session(session_id)

    except Exception as e:
        logger.error(f"Analysis failed for session {session_id}: {e}", exc_info=True)
        session_manager.update_status(
            session_id,
            AnalysisStatus.FAILED,
            0,
            error=str(e)
        )

# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "name": "Multimodal Analysis API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/api/docs",
            "start_session": "POST /api/analysis/start",
            "upload_video": "POST /api/analysis/upload-video",
            "upload_audio": "POST /api/analysis/upload-audio",
            "get_status": "GET /api/analysis/status/{session_id}",
            "get_results": "GET /api/analysis/results/{session_id}"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "gemini_available": GEMINI_AVAILABLE
    }

@app.post("/api/analysis/start", response_model=SessionResponse)
async def start_analysis():
    """
    Start a new analysis session
    Returns a session_id to be used for uploading video/audio and retrieving results
    """
    try:
        session_id = session_manager.create_session()
        return SessionResponse(
            session_id=session_id,
            status="pending",
            message="Session created. Upload video and audio to begin analysis."
        )
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analysis/upload-video", response_model=UploadResponse)
async def upload_video(
    session_id: str,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload video file for analysis
    Accepts WebM, MP4, or other video formats
    """
    try:
        # Validate session
        if session_id not in session_manager.sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        # Save video file
        video_path = session_manager.get_video_path(session_id)

        with open(video_path, "wb") as f:
            content = await file.read()
            f.write(content)

        session_manager.mark_video_received(session_id)

        # If both video and audio are ready, start processing
        if session_manager.is_ready_for_processing(session_id):
            background_tasks.add_task(run_full_analysis, session_id)
            message = "Video uploaded. Analysis started."
        else:
            message = "Video uploaded. Waiting for audio."

        return UploadResponse(
            session_id=session_id,
            status="video_received",
            message=message
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analysis/upload-audio", response_model=UploadResponse)
async def upload_audio(
    session_id: str,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload audio file for analysis
    Accepts WebM, WAV, MP3, or other audio formats
    """
    try:
        # Validate session
        if session_id not in session_manager.sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        # Save audio file
        audio_path = session_manager.get_audio_path(session_id)

        with open(audio_path, "wb") as f:
            content = await file.read()
            f.write(content)

        session_manager.mark_audio_received(session_id)

        # If both video and audio are ready, start processing
        if session_manager.is_ready_for_processing(session_id):
            background_tasks.add_task(run_full_analysis, session_id)
            message = "Audio uploaded. Analysis started."
        else:
            message = "Audio uploaded. Waiting for video."

        return UploadResponse(
            session_id=session_id,
            status="audio_received",
            message=message
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analysis/status/{session_id}", response_model=StatusResponse)
async def get_analysis_status(session_id: str):
    """
    Get the current status of an analysis session
    """
    try:
        status = session_manager.get_status(session_id)

        return StatusResponse(
            session_id=session_id,
            status=status["status"],
            progress=status["progress"],
            message=status.get("message"),
            error=status.get("error")
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analysis/results/{session_id}")
async def get_analysis_results(session_id: str):
    """
    Get the complete analysis results for a session
    Returns comprehensive JSON with emotion, voice, transcription, and AI insights
    """
    try:
        # Check if analysis is complete
        status = session_manager.get_status(session_id)

        if status["status"] != AnalysisStatus.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail=f"Analysis not completed. Current status: {status['status']}"
            )

        # Get results
        results = session_manager.get_results(session_id)

        return JSONResponse(content=results)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/analysis/session/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session and its associated data
    """
    try:
        if session_id not in session_manager.sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        # Cleanup files
        session_manager.cleanup_session(session_id)

        # Remove from sessions
        del session_manager.sessions[session_id]
        if session_id in session_manager.results:
            del session_manager.results[session_id]

        return {"message": f"Session {session_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions")
async def list_sessions():
    """
    List all active sessions
    """
    return {
        "sessions": [
            {
                "session_id": sid,
                "status": data["status"],
                "progress": data["progress"],
                "created_at": data["created_at"]
            }
            for sid, data in session_manager.sessions.items()
        ]
    }

# ==================== Startup & Main ====================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("=== Multimodal Analysis API Starting ===")
    logger.info(f"Gemini AI Available: {GEMINI_AVAILABLE}")

    # Initialize models (lazy loading happens in multimodal_analysis)
    from multimodal_analysis import BASE, onnx_path, proto_path, caffemodel_path

    # Check if model files exist
    if not onnx_path.exists():
        logger.warning(f"ONNX model not found: {onnx_path}")
    if not proto_path.exists() or not caffemodel_path.exists():
        logger.warning(f"Face detection models not found")

    logger.info("API ready to accept requests")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("=== Multimodal Analysis API Shutting Down ===")

    # Cleanup all sessions
    for session_id in list(session_manager.sessions.keys()):
        session_manager.cleanup_session(session_id)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multimodal Analysis API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    logger.info(f"Starting API server on {args.host}:{args.port}")

    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )
