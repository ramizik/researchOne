#!/usr/bin/env python3
"""
Startup script for the Multimodal Analysis API Server
Simple wrapper to launch the FastAPI server with default configuration
"""

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_requirements():
    """Check if all required files and dependencies are present"""
    issues = []

    # Check model files
    model_files = {
        "Emotion Model": Path("emotion-ferplus-8.onnx"),
        "Face Detection Prototxt": Path("RFB-320/RFB-320.prototxt"),
        "Face Detection Model": Path("RFB-320/RFB-320.caffemodel"),
    }

    for name, path in model_files.items():
        if not path.exists():
            issues.append(f"❌ {name} not found: {path}")
        else:
            logger.info(f"✅ {name} found: {path}")

    # Check for credentials (optional but recommended)
    creds_path = Path("credentials.json")
    if not creds_path.exists():
        logger.warning("⚠️  credentials.json not found. Speech transcription and Gemini AI will not work.")
        logger.warning("   See documentation/GEMINI_SETUP.md and documentation/google_cloud_setup.md")
    else:
        logger.info(f"✅ credentials.json found")

    # Check Python dependencies
    try:
        import fastapi
        import uvicorn
        import cv2
        import numpy
        import librosa
        logger.info("✅ All core Python packages available")
    except ImportError as e:
        issues.append(f"❌ Missing Python package: {e.name}")
        issues.append("   Run: pip install -r requirements.txt")

    return issues

def main():
    """Main entry point"""
    print("="*70)
    print("  🎭🎤 MULTIMODAL ANALYSIS API SERVER 🎤🎭")
    print("="*70)
    print()

    # Check requirements
    logger.info("Checking requirements...")
    issues = check_requirements()

    if issues:
        print()
        print("⚠️  SETUP ISSUES DETECTED:")
        for issue in issues:
            print(f"  {issue}")
        print()
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response != 'y':
            print("\nPlease fix the issues above and try again.")
            print("See documentation/INSTALLATION_GUIDE.md for setup instructions.")
            sys.exit(1)

    print()
    print("🚀 Starting API server...")
    print()
    print("📚 API Documentation will be available at:")
    print("   - Swagger UI: http://localhost:8000/api/docs")
    print("   - ReDoc: http://localhost:8000/api/redoc")
    print()
    print("🔗 API Base URL: http://localhost:8000")
    print()
    print("Press CTRL+C to stop the server")
    print("="*70)
    print()

    # Import and run the API server
    try:
        import uvicorn
        uvicorn.run(
            "api_server:app",
            host="0.0.0.0",
            port=8000,
            reload=True,  # Enable auto-reload for development
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
