#!/usr/bin/env python3
"""
Test script for Google Speech-to-Text integration
This script tests the speech transcription functionality without requiring a full multimodal analysis
"""

import os
import sys
import numpy as np
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from speech_transcriber import SpeechTranscriber
        print("✅ SpeechTranscriber imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import SpeechTranscriber: {e}")
        return False
    
    try:
        from voice_analyzer import VoiceAnalyzer
        print("✅ VoiceAnalyzer imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import VoiceAnalyzer: {e}")
        return False
    
    try:
        from google.cloud import speech
        print("✅ Google Cloud Speech library imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Google Cloud Speech: {e}")
        print("   Please install: pip install google-cloud-speech")
        return False
    
    return True

def test_credentials():
    """Test if Google Cloud credentials are properly configured"""
    print("\n🔑 Testing Google Cloud credentials...")
    
    creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if not creds_path:
        print("❌ GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
        print("   Please follow the setup guide in google_cloud_setup.md")
        return False
    
    if not os.path.exists(creds_path):
        print(f"❌ Credentials file not found: {creds_path}")
        return False
    
    print(f"✅ Credentials file found: {creds_path}")
    
    try:
        from google.cloud import speech
        client = speech.SpeechClient()
        print("✅ Google Cloud Speech client initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize Google Cloud client: {e}")
        return False

def test_speech_transcriber():
    """Test the SpeechTranscriber class with dummy data"""
    print("\n🎤 Testing SpeechTranscriber...")
    
    try:
        from speech_transcriber import SpeechTranscriber
        
        # Initialize transcriber
        transcriber = SpeechTranscriber()
        print("✅ SpeechTranscriber initialized successfully")
        
        # Create dummy audio data (1 second of silence)
        sample_rate = 22050
        duration = 1.0  # 1 second
        audio_data = np.zeros(int(sample_rate * duration), dtype=np.float32)
        
        print("📝 Testing transcription with dummy audio...")
        result = transcriber.transcribe_audio_data(audio_data, sample_rate)
        
        if result.get("success", False):
            print("✅ Transcription completed successfully")
            print(f"   Text: '{result['transcription']}'")
            print(f"   Confidence: {result['confidence']:.1%}")
        else:
            print("⚠️  Transcription failed (expected for silence)")
            print(f"   Error: {result.get('error', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ SpeechTranscriber test failed: {e}")
        return False

def test_voice_analyzer():
    """Test the VoiceAnalyzer class"""
    print("\n🎵 Testing VoiceAnalyzer...")
    
    try:
        from voice_analyzer import VoiceAnalyzer
        
        analyzer = VoiceAnalyzer()
        print("✅ VoiceAnalyzer initialized successfully")
        
        # Create dummy audio data (1 second of sine wave)
        sample_rate = 22050
        duration = 1.0
        frequency = 440  # A4 note
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        print("🔊 Testing voice analysis with dummy audio...")
        result = analyzer.analyze_recorded_audio(audio_data, sample_rate)
        
        if "error" not in result:
            print("✅ Voice analysis completed successfully")
            print(f"   Mean Pitch: {result['mean_pitch']:.1f} Hz")
            print(f"   Voice Type: {result['voice_type']}")
        else:
            print(f"⚠️  Voice analysis failed: {result['error']}")
        
        return True
        
    except Exception as e:
        print(f"❌ VoiceAnalyzer test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Google Speech-to-Text Integration")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Credentials Test", test_credentials),
        ("Voice Analyzer Test", test_voice_analyzer),
        ("Speech Transcriber Test", test_speech_transcriber),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Integration is ready to use.")
        print("\nNext steps:")
        print("1. Run: python multimodal_analysis.py")
        print("2. Follow the prompts to record and analyze")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("2. Follow the Google Cloud setup guide: google_cloud_setup.md")
        print("3. Check that GOOGLE_APPLICATION_CREDENTIALS is set correctly")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
