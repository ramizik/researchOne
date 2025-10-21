#!/usr/bin/env python3
"""
Test client for the Multimodal Analysis API
Demonstrates how to interact with the API endpoints
"""

import requests
import time
import sys
from pathlib import Path

# API configuration
API_BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    response = requests.get(f"{API_BASE_URL}/api/health")

    if response.status_code == 200:
        data = response.json()
        print(f"✅ API is healthy")
        print(f"   Status: {data['status']}")
        print(f"   Gemini Available: {data['gemini_available']}")
        return True
    else:
        print(f"❌ Health check failed: {response.status_code}")
        return False

def start_analysis_session():
    """Start a new analysis session"""
    print("\nStarting new analysis session...")
    response = requests.post(f"{API_BASE_URL}/api/analysis/start")

    if response.status_code == 200:
        data = response.json()
        session_id = data['session_id']
        print(f"✅ Session created: {session_id}")
        print(f"   Status: {data['status']}")
        print(f"   Message: {data['message']}")
        return session_id
    else:
        print(f"❌ Failed to create session: {response.status_code}")
        return None

def upload_video(session_id, video_path):
    """Upload video file"""
    print(f"\nUploading video: {video_path}")

    if not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        return False

    with open(video_path, 'rb') as f:
        files = {'file': (Path(video_path).name, f, 'video/webm')}
        response = requests.post(
            f"{API_BASE_URL}/api/analysis/upload-video",
            params={'session_id': session_id},
            files=files
        )

    if response.status_code == 200:
        data = response.json()
        print(f"✅ Video uploaded")
        print(f"   Status: {data['status']}")
        print(f"   Message: {data['message']}")
        return True
    else:
        print(f"❌ Video upload failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def upload_audio(session_id, audio_path):
    """Upload audio file"""
    print(f"\nUploading audio: {audio_path}")

    if not Path(audio_path).exists():
        print(f"❌ Audio file not found: {audio_path}")
        return False

    with open(audio_path, 'rb') as f:
        files = {'file': (Path(audio_path).name, f, 'audio/webm')}
        response = requests.post(
            f"{API_BASE_URL}/api/analysis/upload-audio",
            params={'session_id': session_id},
            files=files
        )

    if response.status_code == 200:
        data = response.json()
        print(f"✅ Audio uploaded")
        print(f"   Status: {data['status']}")
        print(f"   Message: {data['message']}")
        return True
    else:
        print(f"❌ Audio upload failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def poll_status(session_id, max_attempts=120):
    """Poll for analysis status"""
    print(f"\nPolling for analysis status (session: {session_id})...")

    for attempt in range(max_attempts):
        response = requests.get(f"{API_BASE_URL}/api/analysis/status/{session_id}")

        if response.status_code == 200:
            data = response.json()
            status = data['status']
            progress = data['progress']
            message = data.get('message', '')

            print(f"\r   Status: {status:20} | Progress: {progress:3}% | {message[:50]:50}", end='')

            if status == 'completed':
                print()
                print(f"✅ Analysis completed!")
                return True
            elif status == 'failed':
                print()
                print(f"❌ Analysis failed!")
                error = data.get('error', 'Unknown error')
                print(f"   Error: {error}")
                return False

            time.sleep(1)  # Wait 1 second before next poll
        else:
            print(f"\n❌ Failed to get status: {response.status_code}")
            return False

    print(f"\n❌ Timeout waiting for analysis to complete")
    return False

def get_results(session_id):
    """Get analysis results"""
    print(f"\nFetching results...")
    response = requests.get(f"{API_BASE_URL}/api/analysis/results/{session_id}")

    if response.status_code == 200:
        data = response.json()
        print(f"✅ Results retrieved")

        # Display summary
        print(f"\n{'='*70}")
        print(f"ANALYSIS RESULTS SUMMARY")
        print(f"{'='*70}")

        # Emotion analysis
        if 'emotion_analysis' in data:
            emotion_data = data['emotion_analysis']
            if 'emotional_analysis' in emotion_data:
                ea = emotion_data['emotional_analysis']
                print(f"\n🎭 EMOTION ANALYSIS:")
                print(f"   Dominant Emotion: {ea.get('dominant_emotion', 'unknown')}")
                print(f"   Emotional Intensity: {ea.get('emotional_intensity', 'unknown')}")
                print(f"   Emotional Consistency: {ea.get('emotional_consistency', 0):.1f}%")

        # Voice analysis
        if 'voice_analysis' in data:
            voice_data = data['voice_analysis']
            print(f"\n🎤 VOICE ANALYSIS:")
            print(f"   Mean Pitch: {voice_data.get('mean_pitch', 0):.1f} Hz")
            print(f"   Voice Type: {voice_data.get('voice_type', 'unknown')}")

            if 'emotional_indicators' in voice_data:
                ei = voice_data['emotional_indicators']
                print(f"   Energy Level: {ei.get('energy_level', 'unknown')}")
                print(f"   Emotional Arousal: {ei.get('emotional_arousal', 'unknown')}")

        # Transcription
        if 'transcription_analysis' in data:
            trans_data = data['transcription_analysis']
            if trans_data.get('success'):
                print(f"\n📝 TRANSCRIPTION:")
                print(f"   Text: \"{trans_data.get('transcription', '')[:100]}...\"")
                print(f"   Confidence: {trans_data.get('confidence', 0):.1%}")
                print(f"   Word Count: {trans_data.get('word_count', 0)}")

        # Multimodal insights
        if 'multimodal_insights' in data:
            insights = data['multimodal_insights']
            print(f"\n🧠 MULTIMODAL INSIGHTS:")
            print(f"   Overall Emotional State: {insights.get('overall_emotional_state', 'unknown')}")
            print(f"   Emotional Coherence: {insights.get('emotional_coherence', 'unknown')}")
            print(f"   Confidence Score: {insights.get('confidence_score', 0):.2f}")

        # Gemini insights
        if 'gemini_insights' in data and data['gemini_insights']:
            print(f"\n🤖 GEMINI AI INSIGHTS:")
            gemini_text = data['gemini_insights']
            # Show first 500 characters
            print(f"   {gemini_text[:500]}...")

        print(f"\n{'='*70}")

        return data
    else:
        print(f"❌ Failed to get results: {response.status_code}")
        print(f"   Error: {response.text}")
        return None

def run_complete_test(video_path, audio_path):
    """Run complete API test workflow"""
    print("="*70)
    print("MULTIMODAL ANALYSIS API - TEST CLIENT")
    print("="*70)

    # 1. Health check
    if not test_health_check():
        print("\n❌ API is not available. Make sure the server is running:")
        print("   python start_api.py")
        return False

    # 2. Start session
    session_id = start_analysis_session()
    if not session_id:
        return False

    # 3. Upload files
    if not upload_video(session_id, video_path):
        return False

    if not upload_audio(session_id, audio_path):
        return False

    # 4. Poll for completion
    if not poll_status(session_id):
        return False

    # 5. Get results
    results = get_results(session_id)
    if not results:
        return False

    print(f"\n✅ Test completed successfully!")
    print(f"   Session ID: {session_id}")

    return True

def main():
    """Main entry point"""
    if len(sys.argv) < 3:
        print("Usage: python test_api_client.py <video_file> <audio_file>")
        print("\nExample:")
        print("  python test_api_client.py video.webm audio.webm")
        print("\nOr test with existing files:")
        print("  python test_api_client.py video1.mp4 audio.wav")
        sys.exit(1)

    video_path = sys.argv[1]
    audio_path = sys.argv[2]

    success = run_complete_test(video_path, audio_path)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
