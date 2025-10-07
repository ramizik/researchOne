"""
Custom voice analysis module implementing VibratoScope-like functionality
Enhanced with advanced pitch detection and voice quality metrics
"""
import os
import logging
from typing import Dict, Any, Optional, Tuple, List
import librosa
import numpy as np
from scipy import signal
from scipy.stats import linregress, kurtosis, skew
from scipy.interpolate import interp1d
import warnings
import pyaudio
import wave
import tempfile
import time

# Suppress librosa warnings for cleaner logs
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

# ---------- Custom Exceptions ----------
class VoiceAnalysisError(Exception):
    """Base exception for voice analysis errors"""
    pass

class InsufficientDataError(VoiceAnalysisError):
    """Insufficient data for analysis"""
    pass

class AudioQualityError(VoiceAnalysisError):
    """Audio quality issues"""
    pass

class VoiceAnalyzer:
    """Custom voice analysis implementation"""
    
    def __init__(self):
        logger.info("VoiceAnalyzer initialized successfully")
        self.audio = pyaudio.PyAudio()
    
    def __del__(self):
        """Clean up audio resources"""
        if hasattr(self, 'audio'):
            self.audio.terminate()
    
    def record_audio(self, duration: int = 5, sample_rate: int = 22050) -> np.ndarray:
        """
        Record audio from microphone for specified duration
        
        Args:
            duration: Recording duration in seconds
            sample_rate: Sample rate for recording
            
        Returns:
            Numpy array of recorded audio data
        """
        try:
            # Audio recording parameters
            chunk = 1024
            format = pyaudio.paInt16
            channels = 1
            
            print(f"Recording for {duration} seconds...")
            print("Speak now!")
            
            # Open audio stream
            stream = self.audio.open(
                format=format,
                channels=channels,
                rate=sample_rate,
                input=True,
                frames_per_buffer=chunk
            )
            
            frames = []
            
            # Record audio
            for i in range(0, int(sample_rate / chunk * duration)):
                data = stream.read(chunk)
                frames.append(data)
                
                # Show progress
                if i % (sample_rate // chunk) == 0:
                    print(f"Recording... {i // (sample_rate // chunk) + 1}/{duration}")
            
            print("Recording complete!")
            
            # Stop and close stream
            stream.stop_stream()
            stream.close()
            
            # Convert frames to numpy array
            audio_data = b''.join(frames)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Convert to float and normalize
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            return audio_float
            
        except Exception as e:
            logger.error(f"Error recording audio: {str(e)}")
            raise
    
    def analyze_recorded_audio(self, audio_data: np.ndarray, sample_rate: int = 22050, mean_pitch: Optional[float] = None) -> Dict[str, Any]:
        """
        Analyze recorded audio data and extract voice metrics
        
        Args:
            audio_data: Recorded audio data as numpy array
            sample_rate: Sample rate of the audio
            mean_pitch: Optional mean pitch from frontend
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            logger.info(f"Starting voice analysis for recorded audio: {len(audio_data)} samples, {sample_rate} Hz")
            
            # Validate audio quality before analysis
            self._validate_audio_quality(audio_data, sample_rate)
            
            # Pre-process audio for better pitch detection
            y_processed = self._preprocess_audio(audio_data, sample_rate)
            
            # Extract pitch using multiple methods for robustness
            pitch_values = self._extract_pitch_advanced(y_processed, sample_rate)
            
            # Lower threshold for short recordings - require at least 3 pitch values
            if not pitch_values or len(pitch_values) < 3:
                raise InsufficientDataError("Insufficient pitch data for analysis. Please try: 1) Speak louder and closer to the microphone, 2) Try singing a sustained note (like 'ahhh') instead of talking, 3) Ensure there's minimal background noise, 4) Try a longer recording duration")
            
            # Calculate advanced metrics
            pitch_values = np.array(pitch_values)
            
            # Extract harmonic features for better voice characterization
            harmonic_features = self._extract_harmonic_features(audio_data, sample_rate)
            
            # Calculate refined metrics for emotional analysis
            analysis_results = {
                "mean_pitch": float(self._calculate_robust_mean_pitch(pitch_values)),
                "vibrato_rate": self._calculate_vibrato_rate_advanced(pitch_values, sample_rate),
                "jitter": self._calculate_jitter_advanced(pitch_values, sample_rate),
                "shimmer": self._calculate_shimmer_advanced(audio_data, sample_rate),
                "voice_type": self._determine_voice_type_advanced(pitch_values, harmonic_features),
                "lowest_note": self._frequency_to_note(self._get_stable_pitch_percentile(pitch_values, 5)),
                "highest_note": self._frequency_to_note(self._get_stable_pitch_percentile(pitch_values, 95)),
                "singing_characteristics": self._analyze_singing_characteristics(pitch_values, audio_data, sample_rate),
                "emotional_indicators": self._extract_emotional_voice_indicators(pitch_values, audio_data, sample_rate),
            }
            
            # Validate and refine results
            analysis_results = self._validate_and_refine_results(analysis_results, mean_pitch)
            
            logger.info(f"Voice analysis completed: {analysis_results}")
            return analysis_results
            
        except (InsufficientDataError, AudioQualityError, VoiceAnalysisError):
            raise
        except Exception as e:
            logger.error(f"Error in voice analysis: {str(e)}", exc_info=True)
            raise VoiceAnalysisError(f"Unexpected error in voice analysis: {str(e)}")
    
    async def analyze_audio_file(self, audio_file_path: str, mean_pitch: Optional[float] = None) -> Dict[str, Any]:
        """
        Analyze audio file and extract voice metrics using advanced algorithms
        
        Args:
            audio_file_path: Path to the audio file
            mean_pitch: Optional mean pitch from frontend
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            logger.info(f"Starting enhanced voice analysis for file: {audio_file_path}")
            
            # Load audio file with optimal sample rate for pitch detection
            target_sr = 22050  # Standard for pitch analysis
            y, sr = librosa.load(audio_file_path, sr=target_sr)
            logger.info(f"Audio loaded: {len(y)} samples, {sr} Hz sample rate, duration: {len(y)/sr:.2f}s")
            
            # Pre-process audio for better pitch detection
            y_processed = self._preprocess_audio(y, sr)
            
            # Extract pitch using multiple methods for robustness
            pitch_values = self._extract_pitch_advanced(y_processed, sr)
            
            if not pitch_values or len(pitch_values) < 3:
                logger.error("Insufficient pitch data for file analysis")
                raise ValueError("Insufficient pitch data in audio file for analysis")
            
            # Calculate advanced metrics
            pitch_values = np.array(pitch_values)
            
            # Extract harmonic features for better voice characterization
            harmonic_features = self._extract_harmonic_features(y, sr)
            
            # Calculate refined metrics for emotional analysis
            analysis_results = {
                "mean_pitch": float(self._calculate_robust_mean_pitch(pitch_values)),
                "vibrato_rate": self._calculate_vibrato_rate_advanced(pitch_values, sr),
                "jitter": self._calculate_jitter_advanced(pitch_values, sr),
                "shimmer": self._calculate_shimmer_advanced(y, sr),
                "voice_type": self._determine_voice_type_advanced(pitch_values, harmonic_features),
                "lowest_note": self._frequency_to_note(self._get_stable_pitch_percentile(pitch_values, 5)),
                "highest_note": self._frequency_to_note(self._get_stable_pitch_percentile(pitch_values, 95)),
                "singing_characteristics": self._analyze_singing_characteristics(pitch_values, y, sr),
                "emotional_indicators": self._extract_emotional_voice_indicators(pitch_values, y, sr),
            }
            
            # Validate and refine results
            analysis_results = self._validate_and_refine_results(analysis_results, mean_pitch)
            
            logger.info(f"Enhanced voice analysis completed: {analysis_results}")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in enhanced voice analysis: {str(e)}", exc_info=True)
            raise
    
    def _preprocess_audio(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Preprocess audio for better pitch detection"""
        try:
            # Apply high-pass filter to remove low-frequency noise
            b, a = signal.butter(4, 80/(sr/2), btype='high')
            y_filtered = signal.filtfilt(b, a, y)
            
            # Normalize
            y_normalized = y_filtered / (np.max(np.abs(y_filtered)) + 1e-8)
            
            return y_normalized
        except Exception as e:
            logger.warning(f"Audio preprocessing failed: {e}")
            return y
    
    def _extract_pitch_advanced(self, y: np.ndarray, sr: int) -> List[float]:
        """Extract pitch using multiple methods for robustness"""
        pitch_results = []
        
        try:
            # Method 1: PYIN with more lenient parameters for short recordings
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, 
                sr=sr,
                fmin=librosa.note_to_hz('C2'),  # 65 Hz
                fmax=librosa.note_to_hz('C7'),  # 2093 Hz
                frame_length=1024,  # Smaller frame for better short recording handling
                hop_length=256     # Smaller hop for more pitch samples
            )
            
            # More lenient confidence threshold for short recordings
            min_confidence = 0.7  # Lowered from 0.9
            confident_pitches = f0[(voiced_flag) & (voiced_probs > min_confidence)]
            if len(confident_pitches) > 0:
                pitch_results.extend(confident_pitches[~np.isnan(confident_pitches)])
            
            # Method 2: Harmonic-percussive separation for cleaner pitch
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            # Use piptrack on harmonic component with more sensitive settings
            pitches, magnitudes = librosa.piptrack(
                y=y_harmonic, 
                sr=sr, 
                threshold=0.02,  # More sensitive threshold
                fmin=60,         # Lower minimum frequency
                fmax=2500        # Higher maximum frequency
            )
            
            # Extract pitches with more lenient magnitude weighting
            for t in range(pitches.shape[1]):
                # Get top 5 pitch candidates (increased from 3)
                mag_sorted_indices = np.argsort(magnitudes[:, t])[-5:]
                for idx in mag_sorted_indices:
                    pitch = pitches[idx, t]
                    mag = magnitudes[idx, t]
                    if pitch > 0 and mag > np.percentile(magnitudes, 60):  # Lowered from 75
                        pitch_results.append(pitch)
            
            # Method 3: Additional pitch extraction using autocorrelation
            try:
                # Use autocorrelation for additional pitch candidates
                autocorr_pitches = self._extract_pitch_autocorr(y, sr)
                if len(autocorr_pitches) > 0:
                    pitch_results.extend(autocorr_pitches)
            except:
                pass  # Continue if autocorr fails
            
            # Remove outliers using more lenient IQR method
            if pitch_results:
                pitch_results = np.array(pitch_results)
                q1, q3 = np.percentile(pitch_results, [25, 75])
                iqr = q3 - q1
                # More lenient outlier removal (2.0 instead of 1.5)
                lower_bound = q1 - 2.0 * iqr
                upper_bound = q3 + 2.0 * iqr
                pitch_results = pitch_results[(pitch_results >= lower_bound) & (pitch_results <= upper_bound)]
            
            return list(pitch_results)
            
        except Exception as e:
            logger.warning(f"Advanced pitch extraction failed: {e}, trying simple method")
            # Fallback to simple piptrack with more lenient settings
            try:
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.01)
                pitch_values = []
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0:
                        pitch_values.append(pitch)
                return pitch_values
            except:
                return []
    
    def _extract_pitch_autocorr(self, y: np.ndarray, sr: int) -> List[float]:
        """Extract pitch using autocorrelation method as additional fallback"""
        try:
            # Apply windowing to reduce edge effects
            windowed = y * signal.windows.hann(len(y))
            
            # Compute autocorrelation
            autocorr = np.correlate(windowed, windowed, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Find peaks in autocorrelation
            peaks, _ = signal.find_peaks(autocorr, height=0.1*np.max(autocorr))
            
            # Convert peak positions to frequencies
            pitches = []
            for peak in peaks[1:]:  # Skip first peak (DC component)
                if peak > 0:
                    freq = sr / peak
                    if 60 <= freq <= 2000:  # Reasonable voice frequency range
                        pitches.append(freq)
            
            return pitches
        except:
            return []
    
    def _validate_audio_quality(self, audio_data: np.ndarray, sample_rate: int) -> None:
        """
        Validate audio quality before analysis
        
        Args:
            audio_data: Audio data to validate
            sample_rate: Sample rate of the audio
            
        Raises:
            ValueError: If audio quality is insufficient for analysis
        """
        # Check if audio has sufficient length
        min_duration = 1.0  # Minimum 1 second
        if len(audio_data) < sample_rate * min_duration:
            raise AudioQualityError(f"Recording too short. Minimum duration: {min_duration} seconds")
        
        # Check if audio has sufficient amplitude (not too quiet)
        rms_energy = np.sqrt(np.mean(audio_data**2))
        min_rms = 0.001  # Minimum RMS energy threshold
        if rms_energy < min_rms:
            raise AudioQualityError("Recording too quiet. Please speak louder and closer to the microphone")
        
        # Check for clipping (too loud)
        max_amplitude = np.max(np.abs(audio_data))
        if max_amplitude > 0.95:
            raise AudioQualityError("Recording too loud (clipping detected). Please speak more quietly")
        
        # Check for silence (all zeros or very low variance)
        if np.var(audio_data) < 1e-8:
            raise AudioQualityError("No audio detected. Please check your microphone and try again")
        
        logger.info(f"Audio quality validation passed: RMS={rms_energy:.4f}, Max={max_amplitude:.4f}")
    
    def _extract_harmonic_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract harmonic features for voice characterization"""
        try:
            # Compute spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            
            # Compute MFCCs for timbre analysis
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            return {
                'spectral_centroid': np.mean(spectral_centroids),
                'spectral_rolloff': np.mean(spectral_rolloff),
                'spectral_bandwidth': np.mean(spectral_bandwidth),
                'mfcc_mean': np.mean(mfccs, axis=1),
                'brightness': np.mean(spectral_centroids) / sr,  # Normalized brightness
            }
        except Exception as e:
            logger.warning(f"Harmonic feature extraction failed: {e}")
            return {
                'spectral_centroid': 1000.0,
                'spectral_rolloff': 2000.0,
                'spectral_bandwidth': 500.0,
                'mfcc_mean': np.zeros(13),
                'brightness': 0.05,
            }
    
    def _calculate_robust_mean_pitch(self, pitch_values: np.ndarray) -> float:
        """Calculate robust mean pitch using trimmed mean"""
        if len(pitch_values) < 3:
            return float(np.mean(pitch_values))
        
        # Use trimmed mean (remove top and bottom 10%)
        trimmed_pitch = np.sort(pitch_values)[int(len(pitch_values)*0.1):int(len(pitch_values)*0.9)]
        return float(np.mean(trimmed_pitch))
    
    def _get_stable_pitch_percentile(self, pitch_values: np.ndarray, percentile: float) -> float:
        """Get stable pitch at given percentile, filtering out transients"""
        # Use only stable portions (remove rapid changes)
        if len(pitch_values) < 5:
            return np.percentile(pitch_values, percentile)
        
        # Calculate pitch derivatives
        pitch_diff = np.abs(np.diff(pitch_values))
        stable_threshold = np.percentile(pitch_diff, 50)
        
        # Get stable pitch indices
        stable_indices = np.where(pitch_diff < stable_threshold)[0]
        if len(stable_indices) > 0:
            stable_pitches = pitch_values[stable_indices]
            return np.percentile(stable_pitches, percentile)
        else:
            return np.percentile(pitch_values, percentile)
    
    def _calculate_vibrato_rate_advanced(self, pitch_values: np.ndarray, sr: int) -> float:
        """Calculate vibrato rate using advanced spectral analysis"""
        try:
            if len(pitch_values) < 20:
                return self._estimate_vibrato_from_variance(pitch_values)
            
            # Convert pitch to cents for better vibrato analysis
            pitch_cents = 1200 * np.log2(pitch_values / np.mean(pitch_values))
            
            # Remove DC component and trend
            pitch_cents = signal.detrend(pitch_cents, type='linear')
            
            # Apply window to reduce edge effects
            window = signal.windows.hamming(len(pitch_cents))
            pitch_windowed = pitch_cents * window
            
            # Compute power spectrum
            nperseg = min(len(pitch_windowed), 256)
            frequencies, psd = signal.welch(pitch_windowed, fs=sr/512, nperseg=nperseg)
            
            # Find peak in vibrato range (3-8 Hz)
            vibrato_mask = (frequencies >= 3) & (frequencies <= 8)
            if np.any(vibrato_mask):
                vibrato_psd = psd[vibrato_mask]
                vibrato_freqs = frequencies[vibrato_mask]
                
                # Find dominant vibrato frequency
                peak_idx = np.argmax(vibrato_psd)
                vibrato_rate = vibrato_freqs[peak_idx]
                
                # Check if vibrato is significant
                vibrato_power = vibrato_psd[peak_idx]
                noise_floor = np.median(psd)
                
                if vibrato_power > 3 * noise_floor:  # Significant vibrato
                    return float(vibrato_rate)
            
            # No significant vibrato detected
            return self._estimate_vibrato_from_variance(pitch_values)
            
        except Exception as e:
            logger.warning(f"Error in advanced vibrato calculation: {e}")
            return 5.0
    
    def _estimate_vibrato_from_variance(self, pitch_values: np.ndarray) -> float:
        """Estimate vibrato rate from pitch variance"""
        pitch_var = np.var(pitch_values)
        pitch_mean = np.mean(pitch_values)
        cv = pitch_var / (pitch_mean ** 2)
        
        # Map coefficient of variation to typical vibrato rates
        if cv < 0.001:
            return 0.0  # No vibrato
        elif cv < 0.005:
            return 4.5  # Light vibrato
        elif cv < 0.01:
            return 5.5  # Medium vibrato
        else:
            return 6.5  # Strong vibrato
    
    def _calculate_jitter_advanced(self, pitch_values: np.ndarray, sr: int) -> float:
        """Calculate jitter using multiple measurement methods"""
        try:
            if len(pitch_values) < 5:
                return 0.015
            
            # Convert to periods for more accurate jitter calculation
            periods = 1.0 / pitch_values
            
            # Method 1: Local (relative) jitter
            period_diff = np.abs(np.diff(periods))
            local_jitter = np.mean(period_diff) / np.mean(periods)
            
            # Method 2: RAP (Relative Average Perturbation) - 3 period
            rap_values = []
            for i in range(1, len(periods) - 1):
                three_period_avg = np.mean(periods[i-1:i+2])
                rap = np.abs(periods[i] - three_period_avg) / three_period_avg
                rap_values.append(rap)
            rap_jitter = np.mean(rap_values) if rap_values else local_jitter
            
            # Method 3: PPQ5 (5-period perturbation quotient)
            ppq5_values = []
            for i in range(2, len(periods) - 2):
                five_period_avg = np.mean(periods[i-2:i+3])
                ppq5 = np.abs(periods[i] - five_period_avg) / five_period_avg
                ppq5_values.append(ppq5)
            ppq5_jitter = np.mean(ppq5_values) if ppq5_values else rap_jitter
            
            # Combine methods with weights
            combined_jitter = 0.4 * local_jitter + 0.3 * rap_jitter + 0.3 * ppq5_jitter
            
            # Apply voice-quality based scaling
            # Higher jitter for less stable voices
            pitch_stability = np.std(pitch_values) / np.mean(pitch_values)
            stability_factor = 1.0 + (pitch_stability * 0.5)
            
            final_jitter = combined_jitter * stability_factor
            
            # Return with realistic bounds
            return np.clip(final_jitter, 0.001, 0.040)
            
        except Exception as e:
            logger.warning(f"Error in advanced jitter calculation: {e}")
            return 0.015
    
    def _calculate_shimmer_advanced(self, audio: np.ndarray, sr: int) -> float:
        """Calculate shimmer using multiple amplitude perturbation methods"""
        try:
            # Extract amplitude peaks corresponding to glottal cycles
            # Using peak detection on the audio signal
            audio_abs = np.abs(audio)
            
            # Find peaks (local maxima) with adaptive threshold
            threshold = np.percentile(audio_abs, 75)
            peaks, properties = signal.find_peaks(audio_abs, height=threshold, distance=int(sr/500))
            
            if len(peaks) < 5:
                return self._simple_shimmer_calculation(audio, sr)
            
            peak_amplitudes = audio_abs[peaks]
            
            # Method 1: Local shimmer
            amp_diff = np.abs(np.diff(peak_amplitudes))
            local_shimmer = np.mean(amp_diff) / np.mean(peak_amplitudes)
            
            # Method 2: APQ3 (3-period amplitude perturbation quotient)
            apq3_values = []
            for i in range(1, len(peak_amplitudes) - 1):
                three_period_avg = np.mean(peak_amplitudes[i-1:i+2])
                apq3 = np.abs(peak_amplitudes[i] - three_period_avg) / three_period_avg
                apq3_values.append(apq3)
            apq3_shimmer = np.mean(apq3_values) if apq3_values else local_shimmer
            
            # Method 3: APQ5
            apq5_values = []
            for i in range(2, len(peak_amplitudes) - 2):
                five_period_avg = np.mean(peak_amplitudes[i-2:i+3])
                apq5 = np.abs(peak_amplitudes[i] - five_period_avg) / five_period_avg
                apq5_values.append(apq5)
            apq5_shimmer = np.mean(apq5_values) if apq5_values else apq3_shimmer
            
            # Combine methods
            combined_shimmer = 0.4 * local_shimmer + 0.3 * apq3_shimmer + 0.3 * apq5_shimmer
            
            # Add dB shimmer component
            peak_amplitudes_db = 20 * np.log10(peak_amplitudes + 1e-10)
            db_shimmer = np.std(peak_amplitudes_db) / 20  # Normalize to 0-1 range
            
            # Final shimmer with dB component
            final_shimmer = 0.7 * combined_shimmer + 0.3 * db_shimmer
            
            return np.clip(final_shimmer, 0.005, 0.050)
            
        except Exception as e:
            logger.warning(f"Error in advanced shimmer calculation: {e}")
            return self._simple_shimmer_calculation(audio, sr)
    
    def _simple_shimmer_calculation(self, audio: np.ndarray, sr: int) -> float:
        """Simple shimmer calculation as fallback"""
        try:
            # RMS energy in short windows
            window_size = int(0.02 * sr)  # 20ms windows
            hop_size = window_size // 2
            
            rms_values = []
            for i in range(0, len(audio) - window_size, hop_size):
                window = audio[i:i+window_size]
                rms = np.sqrt(np.mean(window**2))
                if rms > 0:
                    rms_values.append(rms)
            
            if rms_values:
                shimmer = np.std(rms_values) / np.mean(rms_values)
                return np.clip(shimmer, 0.010, 0.030)
            else:
                return 0.020
        except:
            return 0.020
    
    def _categorize_dynamics_advanced(self, pitch_values: np.ndarray, audio: np.ndarray) -> str:
        """Categorize dynamics using multiple features"""
        try:
            # Pitch-based dynamics
            pitch_cv = np.std(pitch_values) / np.mean(pitch_values)
            pitch_range = np.ptp(pitch_values) / np.mean(pitch_values)
            
            # Amplitude-based dynamics
            amplitude_envelope = np.abs(audio)
            window_size = int(len(amplitude_envelope) / 20)
            amp_windows = []
            
            for i in range(0, len(amplitude_envelope) - window_size, window_size):
                window_amp = np.mean(amplitude_envelope[i:i+window_size])
                amp_windows.append(window_amp)
            
            if amp_windows:
                amp_cv = np.std(amp_windows) / (np.mean(amp_windows) + 1e-8)
                amp_range = np.ptp(amp_windows) / (np.mean(amp_windows) + 1e-8)
            else:
                amp_cv = 0.1
                amp_range = 0.2
            
            # Combine features with weights
            dynamics_score = (0.3 * pitch_cv + 0.2 * pitch_range + 
                            0.3 * amp_cv + 0.2 * amp_range)
            
            # Add temporal variation analysis
            if len(pitch_values) > 50:
                # Check for consistent patterns vs random variation
                pitch_diff = np.diff(pitch_values)
                autocorr = np.correlate(pitch_diff, pitch_diff, mode='valid')
                if len(autocorr) > 0:
                    pattern_strength = np.max(np.abs(autocorr)) / (np.var(pitch_diff) + 1e-8)
                    dynamics_score *= (1 + 0.2 * pattern_strength)
            
            # Categorize based on comprehensive score
            if dynamics_score < 0.08:
                return "stable"
            elif dynamics_score < 0.15:
                return "controlled"
            elif dynamics_score < 0.25:
                return "variable"
            elif dynamics_score < 0.35:
                return "expressive"
            else:
                return "highly expressive"
                
        except Exception as e:
            logger.warning(f"Error in advanced dynamics categorization: {e}")
            return "stable"
    
    def _determine_voice_type_advanced(self, pitch_values: np.ndarray, harmonic_features: Dict[str, float]) -> str:
        """Determine voice type using pitch and harmonic features"""
        try:
            mean_pitch = np.mean(pitch_values)
            median_pitch = np.median(pitch_values)
            
            # Use both mean and median for robustness
            central_pitch = 0.7 * mean_pitch + 0.3 * median_pitch
            
            # Consider spectral brightness for voice type refinement
            brightness = harmonic_features.get('brightness', 0.05)
            
            # Adjust thresholds based on spectral characteristics
            # Brighter voices tend to be perceived as higher
            brightness_adjustment = 1.0 + (brightness - 0.05) * 2
            adjusted_pitch = central_pitch * brightness_adjustment
            
            # More nuanced voice type classification
            if adjusted_pitch < 160:
                return "bass"
            elif adjusted_pitch < 200:
                return "bass-baritone"
            elif adjusted_pitch < 250:
                return "baritone"
            elif adjusted_pitch < 300:
                return "tenor"
            elif adjusted_pitch < 350:
                return "alto"
            elif adjusted_pitch < 450:
                return "mezzo-soprano"
            else:
                return "soprano"
                
        except Exception as e:
            logger.warning(f"Error in advanced voice type determination: {e}")
            # Fallback to simple classification
            mean_pitch = np.mean(pitch_values) if len(pitch_values) > 0 else 250
            if mean_pitch < 250:
                return "bass"
            elif mean_pitch < 300:
                return "baritone"
            elif mean_pitch < 350:
                return "tenor"
            else:
                return "alto"
    
    def _frequency_to_note(self, frequency: float) -> str:
        """Convert frequency to musical note"""
        if frequency <= 0:
            return "C3"  # Default fallback
            
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        a4 = 440
        c0 = a4 * (2 ** -4.75)
        
        half_steps = round(12 * np.log2(frequency / c0))
        octave = (half_steps // 12)
        note_index = half_steps % 12
        
        return f"{note_names[note_index]}{octave}"
    
    def _analyze_singing_characteristics(self, pitch_values: np.ndarray, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze singing characteristics for emotional analysis"""
        try:
            # Pitch stability for singing quality
            pitch_std = np.std(pitch_values)
            pitch_mean = np.mean(pitch_values)
            pitch_cv = pitch_std / pitch_mean if pitch_mean > 0 else 0
            
            # Melodic contour analysis
            pitch_diff = np.diff(pitch_values)
            ascending_movements = np.sum(pitch_diff > 0)
            descending_movements = np.sum(pitch_diff < 0)
            total_movements = len(pitch_diff)
            
            # Singing style classification
            if pitch_cv < 0.05:
                singing_style = "monotone"
            elif pitch_cv < 0.15:
                singing_style = "controlled"
            elif pitch_cv < 0.25:
                singing_style = "expressive"
            else:
                singing_style = "highly_variable"
            
            # Pitch range analysis
            pitch_range = np.ptp(pitch_values)
            range_ratio = pitch_range / pitch_mean if pitch_mean > 0 else 0
            
            # Vibrato presence and quality
            vibrato_present = self._detect_vibrato_presence(pitch_values, sr)
            vibrato_quality = self._assess_vibrato_quality(pitch_values, sr)
            
            return {
                "pitch_stability": float(pitch_cv),
                "melodic_contour": {
                    "ascending_ratio": float(ascending_movements / total_movements) if total_movements > 0 else 0,
                    "descending_ratio": float(descending_movements / total_movements) if total_movements > 0 else 0,
                    "movement_direction": "ascending" if ascending_movements > descending_movements else "descending"
                },
                "singing_style": singing_style,
                "pitch_range_ratio": float(range_ratio),
                "vibrato_present": vibrato_present,
                "vibrato_quality": vibrato_quality,
                "overall_singing_quality": self._assess_singing_quality(pitch_cv, range_ratio, vibrato_present)
            }
            
        except Exception as e:
            logger.warning(f"Error in singing characteristics analysis: {e}")
            return {
                "pitch_stability": 0.1,
                "melodic_contour": {"ascending_ratio": 0.5, "descending_ratio": 0.5, "movement_direction": "neutral"},
                "singing_style": "unknown",
                "pitch_range_ratio": 0.2,
                "vibrato_present": False,
                "vibrato_quality": "none",
                "overall_singing_quality": "unknown"
            }
    
    def _extract_emotional_voice_indicators(self, pitch_values: np.ndarray, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract voice indicators that correlate with emotional states"""
        try:
            # Energy and intensity analysis
            rms_energy = np.sqrt(np.mean(audio**2))
            energy_variance = np.var(np.abs(audio))
            
            # Pitch-based emotional indicators
            pitch_mean = np.mean(pitch_values)
            pitch_std = np.std(pitch_values)
            pitch_trend = self._calculate_pitch_trend(pitch_values)
            
            # Voice quality indicators
            jitter = self._calculate_jitter_advanced(pitch_values, sr)
            shimmer = self._calculate_shimmer_advanced(audio, sr)
            
            # Emotional state indicators
            emotional_indicators = {
                "energy_level": self._categorize_energy_level(rms_energy),
                "intensity_variation": self._categorize_intensity_variation(energy_variance),
                "pitch_trend": pitch_trend,
                "voice_tension": self._assess_voice_tension(jitter, shimmer),
                "emotional_arousal": self._assess_emotional_arousal(pitch_mean, pitch_std, rms_energy),
                "voice_quality": self._assess_emotional_voice_quality(jitter, shimmer, pitch_std),
                "speaking_rate": self._estimate_speaking_rate(audio, sr),
                "breath_control": self._assess_breath_control(pitch_values, audio, sr)
            }
            
            return emotional_indicators
            
        except Exception as e:
            logger.warning(f"Error in emotional voice indicators extraction: {e}")
            return {
                "energy_level": "medium",
                "intensity_variation": "moderate",
                "pitch_trend": "stable",
                "voice_tension": "normal",
                "emotional_arousal": "neutral",
                "voice_quality": "normal",
                "speaking_rate": "normal",
                "breath_control": "normal"
            }
    
    def _detect_vibrato_presence(self, pitch_values: np.ndarray, sr: int) -> bool:
        """Detect if vibrato is present in the voice"""
        try:
            if len(pitch_values) < 20:
                return False
            
            # Convert to cents for vibrato analysis
            pitch_cents = 1200 * np.log2(pitch_values / np.mean(pitch_values))
            pitch_cents = signal.detrend(pitch_cents, type='linear')
            
            # Check for periodic variation in vibrato range (3-8 Hz)
            nperseg = min(len(pitch_cents), 256)
            frequencies, psd = signal.welch(pitch_cents, fs=sr/512, nperseg=nperseg)
            
            vibrato_mask = (frequencies >= 3) & (frequencies <= 8)
            if np.any(vibrato_mask):
                vibrato_power = np.max(psd[vibrato_mask])
                noise_floor = np.median(psd)
                return vibrato_power > 2 * noise_floor
            
            return False
        except:
            return False
    
    def _assess_vibrato_quality(self, pitch_values: np.ndarray, sr: int) -> str:
        """Assess the quality of vibrato if present"""
        try:
            if not self._detect_vibrato_presence(pitch_values, sr):
                return "none"
            
            vibrato_rate = self._calculate_vibrato_rate_advanced(pitch_values, sr)
            
            if vibrato_rate < 4:
                return "slow"
            elif vibrato_rate < 6:
                return "normal"
            else:
                return "fast"
        except Exception as e:
            raise VoiceAnalysisError(f"Error assessing vibrato quality: {str(e)}")
    
    def _assess_singing_quality(self, pitch_cv: float, range_ratio: float, vibrato_present: bool) -> str:
        """Assess overall singing quality"""
        try:
            quality_score = 0
            
            # Pitch stability (lower is better)
            if pitch_cv < 0.05:
                quality_score += 3
            elif pitch_cv < 0.15:
                quality_score += 2
            elif pitch_cv < 0.25:
                quality_score += 1
            
            # Pitch range (moderate is good)
            if 0.2 <= range_ratio <= 0.6:
                quality_score += 2
            elif 0.1 <= range_ratio <= 0.8:
                quality_score += 1
            
            # Vibrato presence
            if vibrato_present:
                quality_score += 1
            
            if quality_score >= 5:
                return "excellent"
            elif quality_score >= 3:
                return "good"
            elif quality_score >= 1:
                return "fair"
            else:
                return "poor"
        except Exception as e:
            raise VoiceAnalysisError(f"Error assessing singing quality: {str(e)}")
    
    def _calculate_pitch_trend(self, pitch_values: np.ndarray) -> str:
        """Calculate overall pitch trend"""
        try:
            if len(pitch_values) < 3:
                return "stable"
            
            # Linear regression to find trend
            x = np.arange(len(pitch_values))
            slope, _, _, _, _ = linregress(x, pitch_values)
            
            if slope > 0.5:
                return "rising"
            elif slope < -0.5:
                return "falling"
            else:
                return "stable"
        except:
            return "stable"
    
    def _categorize_energy_level(self, rms_energy: float) -> str:
        """Categorize energy level based on RMS energy"""
        if rms_energy < 0.01:
            return "low"
        elif rms_energy < 0.05:
            return "medium"
        else:
            return "high"
    
    def _categorize_intensity_variation(self, energy_variance: float) -> str:
        """Categorize intensity variation"""
        if energy_variance < 1e-6:
            return "minimal"
        elif energy_variance < 1e-4:
            return "moderate"
        else:
            return "high"
    
    def _assess_voice_tension(self, jitter: float, shimmer: float) -> str:
        """Assess voice tension based on jitter and shimmer"""
        tension_score = jitter * 1000 + shimmer * 100  # Scale to similar ranges
        
        if tension_score < 20:
            return "relaxed"
        elif tension_score < 40:
            return "normal"
        elif tension_score < 60:
            return "tense"
        else:
            return "very_tense"
    
    def _assess_emotional_arousal(self, pitch_mean: float, pitch_std: float, rms_energy: float) -> str:
        """Assess emotional arousal level"""
        # Higher pitch and energy typically indicate higher arousal
        arousal_score = 0
        
        if pitch_mean > 300:  # Higher pitch
            arousal_score += 1
        if pitch_std > 50:  # More pitch variation
            arousal_score += 1
        if rms_energy > 0.03:  # Higher energy
            arousal_score += 1
        
        if arousal_score >= 2:
            return "high"
        elif arousal_score >= 1:
            return "medium"
        else:
            return "low"
    
    def _assess_emotional_voice_quality(self, jitter: float, shimmer: float, pitch_std: float) -> str:
        """Assess voice quality from emotional perspective"""
        quality_score = 0
        
        # Lower jitter and shimmer are better
        if jitter < 0.01:
            quality_score += 1
        if shimmer < 0.02:
            quality_score += 1
        if pitch_std < 30:  # More stable pitch
            quality_score += 1
        
        if quality_score >= 2:
            return "clear"
        elif quality_score >= 1:
            return "moderate"
        else:
            return "rough"
    
    def _estimate_speaking_rate(self, audio: np.ndarray, sr: int) -> str:
        """Estimate speaking rate from audio characteristics"""
        try:
            # Simple estimation based on energy peaks
            audio_abs = np.abs(audio)
            threshold = np.percentile(audio_abs, 75)
            peaks, _ = signal.find_peaks(audio_abs, height=threshold, distance=int(sr/20))
            
            # Estimate syllables per second
            duration = len(audio) / sr
            syllables_per_second = len(peaks) / duration if duration > 0 else 0
            
            if syllables_per_second < 2:
                return "slow"
            elif syllables_per_second < 4:
                return "normal"
            else:
                return "fast"
        except:
            return "normal"
    
    def _assess_breath_control(self, pitch_values: np.ndarray, audio: np.ndarray, sr: int) -> str:
        """Assess breath control quality"""
        try:
            # Look for consistent energy and pitch over time
            window_size = int(sr * 0.5)  # 0.5 second windows
            energy_windows = []
            pitch_windows = []
            
            for i in range(0, len(audio) - window_size, window_size):
                window_audio = audio[i:i+window_size]
                window_pitch = pitch_values[i//512:(i+window_size)//512] if len(pitch_values) > (i+window_size)//512 else []
                
                if len(window_audio) > 0:
                    energy_windows.append(np.sqrt(np.mean(window_audio**2)))
                if len(window_pitch) > 0:
                    pitch_windows.append(np.mean(window_pitch))
            
            if len(energy_windows) < 2 or len(pitch_windows) < 2:
                raise InsufficientDataError("Insufficient data for breath control assessment")
            
            # Calculate consistency
            energy_cv = np.std(energy_windows) / np.mean(energy_windows)
            pitch_cv = np.std(pitch_windows) / np.mean(pitch_windows)
            
            consistency_score = 1 / (energy_cv + pitch_cv + 1e-8)
            
            if consistency_score > 10:
                return "excellent"
            elif consistency_score > 5:
                return "good"
            elif consistency_score > 2:
                return "fair"
            else:
                return "poor"
        except Exception as e:
            raise VoiceAnalysisError(f"Error assessing breath control: {str(e)}")

    def _validate_and_refine_results(self, results: Dict[str, Any], frontend_pitch: Optional[float]) -> Dict[str, Any]:
        """Validate and refine analysis results"""
        # Validate mean pitch
        if frontend_pitch and frontend_pitch > 0:
            # Trust frontend pitch if backend pitch seems unrealistic
            if results['mean_pitch'] < 50 or results['mean_pitch'] > 1000:
                results['mean_pitch'] = frontend_pitch
                # Recalculate voice type with new pitch
                results['voice_type'] = self._simple_voice_type(frontend_pitch)
        
        # Ensure all values are in realistic ranges
        results['mean_pitch'] = np.clip(results['mean_pitch'], 80, 800)
        results['vibrato_rate'] = np.clip(results['vibrato_rate'], 0, 10)
        results['jitter'] = np.clip(results['jitter'], 0.001, 0.050)
        results['shimmer'] = np.clip(results['shimmer'], 0.005, 0.060)
        
        # Validate new emotional analysis parameters
        if 'singing_characteristics' in results:
            singing = results['singing_characteristics']
            singing['pitch_stability'] = np.clip(singing.get('pitch_stability', 0.1), 0, 1)
            singing['pitch_range_ratio'] = np.clip(singing.get('pitch_range_ratio', 0.2), 0, 2)
        
        if 'emotional_indicators' in results:
            emotional = results['emotional_indicators']
            # Ensure all emotional indicators have valid values
            for key in emotional:
                if isinstance(emotional[key], str):
                    # Keep string values as is
                    continue
                elif isinstance(emotional[key], (int, float)):
                    # Clip numeric values to reasonable ranges
                    if 'ratio' in key or 'level' in key:
                        emotional[key] = np.clip(emotional[key], 0, 1)
        
        # Validate note range
        lowest_freq = librosa.note_to_hz(results['lowest_note'])
        highest_freq = librosa.note_to_hz(results['highest_note'])
        
        # Ensure at least an octave range
        if highest_freq < lowest_freq * 1.5:
            # Adjust range based on voice type
            if 'bass' in results['voice_type']:
                results['lowest_note'] = 'E2'
                results['highest_note'] = 'E4'
            elif 'baritone' in results['voice_type']:
                results['lowest_note'] = 'G2'
                results['highest_note'] = 'G4'
            elif 'tenor' in results['voice_type']:
                results['lowest_note'] = 'C3'
                results['highest_note'] = 'C5'
            elif 'soprano' in results['voice_type']:
                results['lowest_note'] = 'C4'
                results['highest_note'] = 'C6'
            else:  # alto/mezzo
                results['lowest_note'] = 'G3'
                results['highest_note'] = 'G5'
        
        return results
    
    def _simple_voice_type(self, pitch: float) -> str:
        """Simple voice type classification for validation"""
        if pitch < 250:
            return "bass"
        elif pitch < 300:
            return "baritone"
        elif pitch < 350:
            return "tenor"
        else:
            return "alto"
    


def format_analysis_results(results: Dict[str, Any]) -> None:
    """
    Format and display analysis results in a user-friendly way
    
    Args:
        results: Dictionary containing voice analysis results
    """
    print("\n" + "="*60)
    print("           VOICE ANALYSIS RESULTS")
    print("="*60)
    
    # Basic voice characteristics
    print(f"\n🎵 VOICE CHARACTERISTICS:")
    print(f"   Mean Pitch: {results['mean_pitch']:.1f} Hz")
    print(f"   Voice Type: {results['voice_type'].title()}")
    print(f"   Pitch Range: {results['lowest_note']} - {results['highest_note']}")
    
    # Voice quality metrics
    print(f"\n🎤 VOICE QUALITY:")
    print(f"   Vibrato Rate: {results['vibrato_rate']:.1f} Hz")
    print(f"   Jitter: {results['jitter']:.3f} ({'Low' if results['jitter'] < 0.01 else 'Medium' if results['jitter'] < 0.02 else 'High'} stability)")
    print(f"   Shimmer: {results['shimmer']:.3f} ({'Low' if results['shimmer'] < 0.015 else 'Medium' if results['shimmer'] < 0.025 else 'High'} amplitude variation)")
    
    # Singing Characteristics
    if 'singing_characteristics' in results:
        singing = results['singing_characteristics']
        print(f"\n🎵 SINGING CHARACTERISTICS:")
        print(f"   Style: {singing.get('singing_style', 'unknown').title()}")
        print(f"   Quality: {singing.get('overall_singing_quality', 'unknown').title()}")
        print(f"   Pitch Stability: {singing.get('pitch_stability', 0):.3f}")
        print(f"   Vibrato: {'Present' if singing.get('vibrato_present', False) else 'Not detected'}")
        if singing.get('vibrato_present', False):
            print(f"   Vibrato Quality: {singing.get('vibrato_quality', 'unknown').title()}")
    
    # Emotional Voice Indicators
    if 'emotional_indicators' in results:
        emotional = results['emotional_indicators']
        print(f"\n🎭 EMOTIONAL VOICE INDICATORS:")
        print(f"   Energy Level: {emotional.get('energy_level', 'unknown').title()}")
        print(f"   Emotional Arousal: {emotional.get('emotional_arousal', 'unknown').title()}")
        print(f"   Voice Tension: {emotional.get('voice_tension', 'unknown').title()}")
        print(f"   Voice Quality: {emotional.get('voice_quality', 'unknown').title()}")
        print(f"   Speaking Rate: {emotional.get('speaking_rate', 'unknown').title()}")
        print(f"   Breath Control: {emotional.get('breath_control', 'unknown').title()}")
    
    print("\n" + "="*60)
    print("Analysis complete! 🎉")
    print("="*60)


def main():
    """
    Main function to record audio from microphone and analyze it
    """
    print("🎤 Voice Analyzer - Microphone Recording Mode")
    print("=" * 50)
    
    try:
        # Initialize voice analyzer
        analyzer = VoiceAnalyzer()
        
        # Get recording duration from user
        while True:
            try:
                duration = input("\nEnter recording duration in seconds (default: 5): ").strip()
                if not duration:
                    duration = 5
                else:
                    duration = int(duration)
                    if duration < 1 or duration > 30:
                        print("Please enter a duration between 1 and 30 seconds.")
                        continue
                break
            except ValueError:
                print("Please enter a valid number.")
                continue
        
        print(f"\nPreparing to record for {duration} seconds...")
        print("Make sure your microphone is working and positioned correctly.")
        
        # Record audio
        audio_data = analyzer.record_audio(duration=duration)
        
        print("\nAnalyzing your voice... This may take a moment.")
        
        try:
            # Analyze the recorded audio
            results = analyzer.analyze_recorded_audio(audio_data)
            
            # Display results
            format_analysis_results(results)
            
        except ValueError as e:
            print(f"\n❌ Analysis failed: {str(e)}")
            print("\n💡 Tips for better recording:")
            print("   • Speak louder and closer to the microphone")
            print("   • Try singing a sustained note (like 'ahhh') instead of talking")
            print("   • Ensure there's minimal background noise")
            print("   • Try a longer recording duration (8-10 seconds)")
            print("   • Make sure your microphone is working properly")
            return
        except Exception as e:
            print(f"\n❌ Unexpected error during analysis: {str(e)}")
            print("Please try again or check your microphone setup.")
            return
        
    except KeyboardInterrupt:
        print("\n\nRecording cancelled by user.")
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please make sure:")
        print("1. Your microphone is connected and working")
        print("2. PyAudio is installed (pip install pyaudio)")
        print("3. You have granted microphone permissions")
    finally:
        print("\nThank you for using Voice Analyzer! 🎵")


if __name__ == "__main__":
    main()
 