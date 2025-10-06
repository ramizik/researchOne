"""
Google Speech-to-Text integration for multimodal analysis
Handles audio transcription using Google Cloud Speech-to-Text API
"""

import os
import tempfile
import wave
import logging
from typing import Dict, Any, Optional
import numpy as np
from google.cloud import speech

logger = logging.getLogger(__name__)

class SpeechTranscriber:
    """Google Speech-to-Text integration for audio transcription"""
    
    def __init__(self, credentials_path: Optional[str] = None):
        """
        Initialize the Speech-to-Text client
        
        Args:
            credentials_path: Path to Google Cloud service account JSON file
        """
        try:
            # Set up credentials if provided
            if credentials_path and os.path.exists(credentials_path):
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
                logger.info(f"Using credentials from: {credentials_path}")
            elif 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
                # Try to use credentials.json in the current directory
                default_creds_path = "credentials.json"
                if os.path.exists(default_creds_path):
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = default_creds_path
                    logger.info(f"Using default credentials from: {default_creds_path}")
                else:
                    logger.warning("No Google Cloud credentials found. Please set GOOGLE_APPLICATION_CREDENTIALS environment variable or place credentials.json in the project directory")
            
            # Initialize the Speech-to-Text client
            self.client = speech.SpeechClient()
            logger.info("Speech-to-Text client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Speech-to-Text client: {str(e)}")
            raise
    
    def transcribe_audio_data(self, audio_data: np.ndarray, sample_rate: int = 22050, 
                            language_code: str = "en-US") -> Dict[str, Any]:
        """
        Transcribe audio data using Google Speech-to-Text API
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            language_code: Language code for transcription (default: en-US)
            
        Returns:
            Dictionary containing transcription results
        """
        try:
            logger.info(f"Starting transcription: {len(audio_data)} samples, {sample_rate} Hz")
            
            # Save audio data to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                self._save_audio_to_wav(audio_data, sample_rate, temp_path)
            
            try:
                # Transcribe the audio file
                result = self._transcribe_wav_file(temp_path, sample_rate, language_code)
                return result
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            return {
                "transcription": "",
                "confidence": 0.0,
                "error": str(e),
                "success": False
            }
    
    def _save_audio_to_wav(self, audio_data: np.ndarray, sample_rate: int, file_path: str) -> None:
        """
        Save numpy audio array to WAV file
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            file_path: Path to save the WAV file
        """
        try:
            # Convert float audio to 16-bit PCM
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Save as WAV file
            with wave.open(file_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
                
            logger.info(f"Audio saved to WAV file: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save audio to WAV: {str(e)}")
            raise
    
    def _transcribe_wav_file(self, file_path: str, sample_rate: int, language_code: str) -> Dict[str, Any]:
        """
        Transcribe WAV file using Google Speech-to-Text API
        
        Args:
            file_path: Path to the WAV file
            sample_rate: Sample rate of the audio
            language_code: Language code for transcription
            
        Returns:
            Dictionary containing transcription results
        """
        try:
            # Read the audio file
            with open(file_path, 'rb') as audio_file:
                content = audio_file.read()
            
            # Configure the recognition settings
            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=sample_rate,
                language_code=language_code,
                enable_automatic_punctuation=True,
                enable_word_time_offsets=True,
                model='latest_long',  # Use the latest long model for better accuracy
            )
            
            # Perform the transcription
            logger.info("Sending audio to Google Speech-to-Text API...")
            response = self.client.recognize(config=config, audio=audio)
            
            # Process the response
            if response.results:
                result = response.results[0]
                alternative = result.alternatives[0]
                
                transcription_result = {
                    "transcription": alternative.transcript,
                    "confidence": alternative.confidence,
                    "success": True,
                    "error": None,
                    "word_count": len(alternative.transcript.split()),
                    "language_code": language_code
                }
                
                # Add word-level timing if available
                if hasattr(alternative, 'words') and alternative.words:
                    transcription_result["words"] = [
                        {
                            "word": word.word,
                            "start_time": word.start_time.total_seconds() if word.start_time else 0,
                            "end_time": word.end_time.total_seconds() if word.end_time else 0
                        }
                        for word in alternative.words
                    ]
                
                logger.info(f"Transcription successful: {len(alternative.transcript)} characters")
                return transcription_result
            else:
                logger.warning("No transcription results returned from API")
                return {
                    "transcription": "",
                    "confidence": 0.0,
                    "success": False,
                    "error": "No speech detected",
                    "word_count": 0,
                    "language_code": language_code
                }
                
        except Exception as e:
            # Check if it's a Google Cloud specific error
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['google', 'cloud', 'api', 'credentials', 'permission', 'quota']):
                logger.error(f"Google Cloud API error: {str(e)}")
                return {
                    "transcription": "",
                    "confidence": 0.0,
                    "success": False,
                    "error": f"Google Cloud API error: {str(e)}",
                    "word_count": 0,
                    "language_code": language_code
                }
            else:
                logger.error(f"Unexpected error during transcription: {str(e)}")
                return {
                    "transcription": "",
                    "confidence": 0.0,
                    "success": False,
                    "error": f"Unexpected error: {str(e)}",
                    "word_count": 0,
                    "language_code": language_code
                }
    
    def transcribe_with_enhanced_config(self, audio_data: np.ndarray, sample_rate: int = 22050,
                                      language_code: str = "en-US", 
                                      enhanced_model: bool = True) -> Dict[str, Any]:
        """
        Transcribe audio with enhanced configuration for better accuracy
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            language_code: Language code for transcription
            enhanced_model: Whether to use enhanced model
            
        Returns:
            Dictionary containing transcription results
        """
        try:
            # Save audio data to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                self._save_audio_to_wav(audio_data, sample_rate, temp_path)
            
            try:
                # Read the audio file
                with open(temp_path, 'rb') as audio_file:
                    content = audio_file.read()
                
                # Enhanced configuration
                audio = speech.RecognitionAudio(content=content)
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=sample_rate,
                    language_code=language_code,
                    enable_automatic_punctuation=True,
                    enable_word_time_offsets=True,
                    enable_word_confidence=True,
                    use_enhanced=enhanced_model,
                    model='latest_long' if enhanced_model else 'default',
                    # Add speech adaptation hints for better accuracy
                    adaptation=speech.SpeechAdaptation(
                        phrase_sets=[
                            speech.PhraseSet(
                                phrases=[
                                    speech.PhraseSet.Phrase(
                                        value="emotion analysis",
                                        boost=20.0
                                    ),
                                    speech.PhraseSet.Phrase(
                                        value="voice analysis",
                                        boost=20.0
                                    ),
                                    speech.PhraseSet.Phrase(
                                        value="multimodal",
                                        boost=15.0
                                    )
                                ]
                            )
                        ]
                    )
                )
                
                # Perform the transcription
                logger.info("Sending audio to Google Speech-to-Text API with enhanced config...")
                response = self.client.recognize(config=config, audio=audio)
                
                # Process the response
                if response.results:
                    result = response.results[0]
                    alternative = result.alternatives[0]
                    
                    transcription_result = {
                        "transcription": alternative.transcript,
                        "confidence": alternative.confidence,
                        "success": True,
                        "error": None,
                        "word_count": len(alternative.transcript.split()),
                        "language_code": language_code,
                        "enhanced_model": enhanced_model
                    }
                    
                    # Add word-level timing and confidence if available
                    if hasattr(alternative, 'words') and alternative.words:
                        transcription_result["words"] = [
                            {
                                "word": word.word,
                                "start_time": word.start_time.total_seconds() if word.start_time else 0,
                                "end_time": word.end_time.total_seconds() if word.end_time else 0,
                                "confidence": getattr(word, 'confidence', 0.0)
                            }
                            for word in alternative.words
                        ]
                    
                    logger.info(f"Enhanced transcription successful: {len(alternative.transcript)} characters")
                    return transcription_result
                else:
                    logger.warning("No transcription results returned from enhanced API")
                    return {
                        "transcription": "",
                        "confidence": 0.0,
                        "success": False,
                        "error": "No speech detected",
                        "word_count": 0,
                        "language_code": language_code,
                        "enhanced_model": enhanced_model
                    }
                    
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            logger.error(f"Enhanced transcription failed: {str(e)}")
            return {
                "transcription": "",
                "confidence": 0.0,
                "success": False,
                "error": str(e),
                "word_count": 0,
                "language_code": language_code,
                "enhanced_model": enhanced_model
            }


def format_transcription_results(results: Dict[str, Any]) -> None:
    """
    Format and display transcription results in a user-friendly way
    
    Args:
        results: Dictionary containing transcription results
    """
    print("\n" + "="*60)
    print("           SPEECH-TO-TEXT RESULTS")
    print("="*60)
    
    if results.get("success", False):
        print(f"\n📝 TRANSCRIPTION:")
        print(f"   Text: \"{results['transcription']}\"")
        print(f"   Confidence: {results['confidence']:.1%}")
        print(f"   Word Count: {results['word_count']}")
        print(f"   Language: {results['language_code']}")
        
        if results.get("enhanced_model"):
            print(f"   Model: Enhanced")
        
        # Show word-level timing if available
        if "words" in results and results["words"]:
            print(f"\n⏱️  WORD TIMING:")
            for i, word_info in enumerate(results["words"][:10]):  # Show first 10 words
                start_time = word_info.get("start_time", 0)
                end_time = word_info.get("end_time", 0)
                confidence = word_info.get("confidence", 0)
                print(f"   {i+1:2d}. '{word_info['word']}' ({start_time:.1f}s-{end_time:.1f}s, {confidence:.1%})")
            
            if len(results["words"]) > 10:
                print(f"   ... and {len(results['words']) - 10} more words")
    else:
        print(f"\n❌ TRANSCRIPTION FAILED:")
        print(f"   Error: {results.get('error', 'Unknown error')}")
        print(f"   Success: {results.get('success', False)}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    """
    Test the SpeechTranscriber with a sample audio file
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python speech_transcriber.py <audio_file_path>")
        sys.exit(1)
    
    audio_file_path = sys.argv[1]
    
    try:
        # Initialize transcriber
        transcriber = SpeechTranscriber()
        
        # Load audio file
        import librosa
        audio_data, sample_rate = librosa.load(audio_file_path, sr=22050)
        
        # Transcribe
        results = transcriber.transcribe_audio_data(audio_data, sample_rate)
        
        # Display results
        format_transcription_results(results)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
