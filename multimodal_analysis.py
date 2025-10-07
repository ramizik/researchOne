"""
Multimodal Emotion and Voice Analysis
Simultaneously records video and audio for 15 seconds and provides comprehensive analysis
"""

import cv2
import numpy as np
import time
from cv2 import dnn
from math import ceil
from pathlib import Path
import sys
from collections import defaultdict, Counter
import threading
from typing import Dict, Any, Optional, List
import warnings
import json
from datetime import datetime
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  google-generativeai not installed. Install with: pip install google-generativeai")

# Import voice analyzer and speech transcriber
from voice_analyzer import VoiceAnalyzer, VoiceAnalysisError, InsufficientDataError, AudioQualityError
from speech_transcriber import SpeechTranscriber

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# ---------- Custom Exceptions ----------
class MultimodalAnalysisError(Exception):
    """Base exception for multimodal analysis errors"""
    pass

class CameraError(MultimodalAnalysisError):
    """Camera-related errors"""
    pass

class AudioError(MultimodalAnalysisError):
    """Audio-related errors"""
    pass

class ModelError(MultimodalAnalysisError):
    """Model-related errors (ONNX, face detection, etc.)"""
    pass

class GeminiError(MultimodalAnalysisError):
    """Gemini AI-related errors"""
    pass

class TranscriptionError(MultimodalAnalysisError):
    """Speech transcription errors"""
    pass

class VoiceAnalysisError(MultimodalAnalysisError):
    """Voice analysis errors"""
    pass

# ---------- Memory Storage System ----------
class AnalysisMemory:
    """Temporary memory storage for analysis results during execution"""
    
    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.emotion_data = {}
        self.voice_data = {}
        self.transcription_data = {}
        self.analysis_metadata = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "recording_duration": 15,
            "analysis_version": "2.0",
            "status": "in_progress"
        }
    
    def store_emotion_data(self, emotion_data: Dict[str, Any]):
        """Store emotion analysis results"""
        self.emotion_data = emotion_data
        self.emotion_data["analysis_timestamp"] = datetime.now().isoformat()
    
    def store_voice_data(self, voice_data: Dict[str, Any]):
        """Store voice analysis results"""
        self.voice_data = voice_data
        self.voice_data["analysis_timestamp"] = datetime.now().isoformat()
    
    def store_transcription_data(self, transcription_data: Dict[str, Any]):
        """Store transcription results"""
        self.transcription_data = transcription_data
        self.transcription_data["analysis_timestamp"] = datetime.now().isoformat()
    
    def get_complete_analysis(self) -> Dict[str, Any]:
        """Get complete analysis data for JSON export"""
        self.analysis_metadata["status"] = "completed"
        self.analysis_metadata["completion_timestamp"] = datetime.now().isoformat()
        
        return {
            "metadata": self.analysis_metadata,
            "emotion_analysis": self.emotion_data,
            "voice_analysis": self.voice_data,
            "transcription_analysis": self.transcription_data,
            "multimodal_insights": self._generate_multimodal_insights()
        }
    
    def _generate_multimodal_insights(self) -> Dict[str, Any]:
        """Generate insights from combined analysis"""
        insights = {
            "emotional_coherence": "unknown",
            "voice_emotion_alignment": "unknown",
            "overall_emotional_state": "unknown",
            "confidence_score": 0.0,
            "key_observations": []
        }
        
        try:
            # Check if we have valid data
            if "error" in self.emotion_data or "error" in self.voice_data:
                insights["key_observations"].append("Incomplete analysis due to errors")
                return insights
            
            # Extract key metrics
            emotions_by_second = self.emotion_data.get("emotions_by_second", {})
            all_emotions = []
            for second, emotions in emotions_by_second.items():
                all_emotions.extend(emotions)
            
            if not all_emotions:
                insights["key_observations"].append("No facial emotions detected")
                return insights
            
            # Calculate dominant emotion
            emotion_counts = Counter(all_emotions)
            dominant_emotion = emotion_counts.most_common(1)[0][0]
            
            # Get voice characteristics
            voice_arousal = self.voice_data.get('emotional_indicators', {}).get('emotional_arousal', 'neutral')
            voice_energy = self.voice_data.get('emotional_indicators', {}).get('energy_level', 'medium')
            mean_pitch = self.voice_data.get('mean_pitch', 0)
            
            # Emotional coherence analysis
            if dominant_emotion in ["happiness", "surprise"] and voice_arousal == "high":
                insights["emotional_coherence"] = "high"
                insights["key_observations"].append("Positive emotions align with high voice arousal")
            elif dominant_emotion in ["sadness", "neutral"] and voice_arousal == "low":
                insights["emotional_coherence"] = "high"
                insights["key_observations"].append("Calm emotions align with low voice arousal")
            elif dominant_emotion in ["anger", "fear"] and voice_arousal == "high":
                insights["emotional_coherence"] = "high"
                insights["key_observations"].append("Intense emotions align with high voice arousal")
            else:
                insights["emotional_coherence"] = "moderate"
                insights["key_observations"].append("Mixed emotional signals detected")
            
            # Voice-emotion alignment
            if mean_pitch > 250 and dominant_emotion in ["happiness", "surprise"]:
                insights["voice_emotion_alignment"] = "high"
                insights["key_observations"].append("High pitch matches positive emotions")
            elif mean_pitch < 200 and dominant_emotion in ["sadness", "neutral"]:
                insights["voice_emotion_alignment"] = "high"
                insights["key_observations"].append("Low pitch matches subdued emotions")
            else:
                insights["voice_emotion_alignment"] = "moderate"
            
            # Overall emotional state
            if insights["emotional_coherence"] == "high" and insights["voice_emotion_alignment"] == "high":
                insights["overall_emotional_state"] = f"coherent_{dominant_emotion}"
                insights["confidence_score"] = 0.9
            elif insights["emotional_coherence"] == "high" or insights["voice_emotion_alignment"] == "high":
                insights["overall_emotional_state"] = f"mostly_{dominant_emotion}"
                insights["confidence_score"] = 0.7
            else:
                insights["overall_emotional_state"] = f"mixed_{dominant_emotion}"
                insights["confidence_score"] = 0.5
            
            # Add transcription insights if available
            if self.transcription_data.get("success", False):
                word_count = self.transcription_data.get("word_count", 0)
                speaking_rate = word_count / RECORDING_DURATION if RECORDING_DURATION > 0 else 0
                insights["key_observations"].append(f"Speaking rate: {speaking_rate:.1f} words/second")
                
                if speaking_rate > 3:
                    insights["key_observations"].append("Fast speaking rate detected")
                elif speaking_rate < 1:
                    insights["key_observations"].append("Slow speaking rate detected")
            
        except Exception as e:
            insights["key_observations"].append(f"Error generating insights: {str(e)}")
        
        return insights

# Global memory instance
analysis_memory = AnalysisMemory()

# ---------- Gemini AI Configuration ----------
def configure_gemini():
    """Configure Google Generative AI"""
    if not GEMINI_AVAILABLE:
        raise GeminiError("Google Generative AI package not installed. Install with: pip install google-generativeai")
    
    try:
        # Try to load API key from environment variable or file
        import os
        api_key = os.getenv('GOOGLE_API_KEY')
        
        if not api_key:
            # Try to load from credentials.json file
            try:
                with open('credentials.json', 'r') as f:
                    creds = json.load(f)
                    api_key = creds.get('api_key') or creds.get('GOOGLE_API_KEY')
            except FileNotFoundError:
                pass
        
        if not api_key:
            raise GeminiError("Google API key not found. Please set GOOGLE_API_KEY environment variable or add 'api_key' to credentials.json")
        
        genai.configure(api_key=api_key)
        return True
        
    except GeminiError:
        raise
    except Exception as e:
        raise GeminiError(f"Error configuring Gemini: {str(e)}")

def get_gemini_analysis(emotion_data: Dict[str, Any], voice_data: Dict[str, Any]) -> str:
    """Get comprehensive analysis from Gemini AI"""
    try:
        # Configure Gemini
        configure_gemini()
        
        # Create the model
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Format the data for Gemini
        analysis_prompt = format_data_for_gemini(emotion_data, voice_data)
        
        # Generate response
        response = model.generate_content(analysis_prompt)
        
        if not response.text:
            raise GeminiError("Empty response received from Gemini AI")
        
        return response.text
        
    except GeminiError:
        raise
    except Exception as e:
        raise GeminiError(f"Error getting Gemini analysis: {str(e)}")

def format_data_for_gemini(emotion_data: Dict[str, Any], voice_data: Dict[str, Any]) -> str:
    """Format analysis data into a comprehensive prompt for Gemini"""
    
    # Extract key information
    emotions_by_second = emotion_data.get("emotions_by_second", {})
    all_emotions = []
    for second, emotions in emotions_by_second.items():
        all_emotions.extend(emotions)
    
    emotion_counts = Counter(all_emotions) if all_emotions else {}
    dominant_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else "unknown"
    
    # Get voice characteristics
    voice_indicators = voice_data.get('emotional_indicators', {})
    singing_characteristics = voice_data.get('singing_characteristics', {})
    
    # Get transcription data
    transcription_data = voice_data.get('transcription', {})
    
    # Build comprehensive prompt
    prompt = f"""
You are an expert psychologist and voice analysis specialist. Please analyze the following multimodal data from a 15-second recording session and provide comprehensive insights about the person's emotional state, vocal characteristics, and overall psychological profile.

## FACIAL EMOTION ANALYSIS
- Dominant Emotion: {dominant_emotion}
- Emotion Distribution: {dict(emotion_counts)}
- Total Emotion Detections: {len(all_emotions)}
- Emotions by Second: {dict(emotions_by_second)}

## VOICE ANALYSIS
### Basic Characteristics:
- Mean Pitch: {voice_data.get('mean_pitch', 'N/A')} Hz
- Voice Type: {voice_data.get('voice_type', 'N/A')}
- Pitch Range: {voice_data.get('lowest_note', 'N/A')} - {voice_data.get('highest_note', 'N/A')}
- Vibrato Rate: {voice_data.get('vibrato_rate', 'N/A')} Hz
- Jitter: {voice_data.get('jitter', 'N/A')}
- Shimmer: {voice_data.get('shimmer', 'N/A')}

### Singing Characteristics:
- Singing Style: {singing_characteristics.get('singing_style', 'N/A')}
- Overall Quality: {singing_characteristics.get('overall_singing_quality', 'N/A')}
- Pitch Stability: {singing_characteristics.get('pitch_stability', 'N/A')}
- Vibrato Present: {singing_characteristics.get('vibrato_present', 'N/A')}
- Vibrato Quality: {singing_characteristics.get('vibrato_quality', 'N/A')}

### Emotional Voice Indicators:
- Energy Level: {voice_indicators.get('energy_level', 'N/A')}
- Emotional Arousal: {voice_indicators.get('emotional_arousal', 'N/A')}
- Voice Tension: {voice_indicators.get('voice_tension', 'N/A')}
- Voice Quality: {voice_indicators.get('voice_quality', 'N/A')}
- Speaking Rate: {voice_indicators.get('speaking_rate', 'N/A')}
- Breath Control: {voice_indicators.get('breath_control', 'N/A')}

## SPEECH TRANSCRIPTION
- Transcription: "{transcription_data.get('transcription', 'No speech detected')}"
- Confidence: {transcription_data.get('confidence', 0):.1%}
- Word Count: {transcription_data.get('word_count', 0)}
- Success: {transcription_data.get('success', False)}

## ENHANCED EMOTIONAL ANALYSIS
{emotion_data.get('emotional_analysis', {})}

## FACIAL EXPRESSION QUALITY
{emotion_data.get('facial_expression_quality', {})}

## EMOTIONAL STABILITY METRICS
{emotion_data.get('emotional_stability_metrics', {})}

---

Please provide a comprehensive analysis covering:

1. **Overall Emotional State Assessment**: What is the person's primary emotional state and how consistent is it?

2. **Voice-Emotion Alignment**: How well do the facial expressions align with voice characteristics? Are there any discrepancies?

3. **Psychological Insights**: What can you infer about the person's psychological state, stress level, and emotional regulation?

4. **Vocal Characteristics Analysis**: What do the voice parameters tell us about the person's emotional state, confidence, and communication style?

5. **Multimodal Coherence**: How coherent are the different modalities (facial expressions, voice, speech content) in expressing emotion?

6. **Potential Concerns or Strengths**: Are there any indicators of stress, anxiety, confidence, or other psychological factors?

7. **Recommendations**: Based on the analysis, what insights or recommendations would you provide?

Please be thorough, professional, and provide specific evidence from the data to support your conclusions.
"""
    
    return prompt

# ---------- Config ----------
RECORDING_DURATION = 15  # seconds to record
SAVE_OUTPUT = True       # write annotated AVI
OUTPUT_FPS = 10

# ---------- Emotion Detection Globals ----------
image_mean = np.array([127, 127, 127])
image_std = 128.0
iou_threshold = 0.3
center_variance = 0.1
size_variance = 0.2
min_boxes = [
    [10.0, 16.0, 24.0],
    [32.0, 48.0],
    [64.0, 96.0],
    [128.0, 192.0, 256.0],
]
strides = [8.0, 16.0, 32.0, 64.0]
threshold = 0.5

# ---------- Utils for SSD ----------
def define_img_size(image_size):
    shrinkage_list = []
    feature_map_w_h_list = []
    for size in image_size:
        feature_map = [int(ceil(size / stride)) for stride in strides]
        feature_map_w_h_list.append(feature_map)
    for _ in range(0, len(image_size)):
        shrinkage_list.append(strides)
    priors = generate_priors(feature_map_w_h_list, shrinkage_list, image_size, min_boxes)
    return priors

def generate_priors(feature_map_list, shrinkage_list, image_size, min_boxes):
    priors = []
    for index in range(0, len(feature_map_list[0])):
        scale_w = image_size[0] / shrinkage_list[0][index]
        scale_h = image_size[1] / shrinkage_list[1][index]
        for j in range(0, feature_map_list[1][index]):
            for i in range(0, feature_map_list[0][index]):
                x_center = (i + 0.5) / scale_w
                y_center = (j + 0.5) / scale_h
                for min_box in min_boxes[index]:
                    w = min_box / image_size[0]
                    h = min_box / image_size[1]
                    priors.append([x_center, y_center, w, h])
    return np.clip(priors, 0.0, 1.0)

def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(rest_boxes, np.expand_dims(current_box, axis=0))
        indexes = indexes[iou <= iou_threshold]
    return box_scores[picked, :]

def area_of(left_top, right_bottom):
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]

def iou_of(boxes0, boxes1, eps=1e-5):
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])
    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs, iou_threshold=iou_threshold, top_k=top_k)
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return (
        picked_box_probs[:, :4].astype(np.int32),
        np.array(picked_labels),
        picked_box_probs[:, 4],
    )

def convert_locations_to_boxes(locations, priors, center_variance, size_variance):
    if len(priors.shape) + 1 == len(locations.shape):
        priors = np.expand_dims(priors, 0)
    return np.concatenate(
        [
            locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
            np.exp(locations[..., 2:] * size_variance) * priors[..., 2:],
        ],
        axis=len(locations.shape) - 1,
    )

def center_form_to_corner_form(locations):
    return np.concatenate(
        [locations[..., :2] - locations[..., 2:] / 2, locations[..., :2] + locations[..., 2:] / 2],
        len(locations.shape) - 1,
    )

# ---------- Video Emotion Detection ----------
def record_video_emotions() -> Dict[str, Any]:
    """
    Record video for 15 seconds and detect emotions
    Returns dictionary with emotion data
    """
    emotion_dict = {
        0: "neutral",
        1: "happiness",
        2: "surprise",
        3: "sadness",
        4: "anger",
        5: "disgust",
        6: "fear",
    }
    
    BASE = Path(__file__).resolve().parent
    onnx_path = BASE / "emotion-ferplus-8.onnx"
    proto_path = BASE / "RFB-320" / "RFB-320.prototxt"
    caffemodel_path = BASE / "RFB-320" / "RFB-320.caffemodel"
    
    if not onnx_path.exists():
        raise ModelError(f"Missing ONNX model: {onnx_path}")
    if not proto_path.exists() or not caffemodel_path.exists():
        raise ModelError(f"Missing face detector files: {proto_path} or {caffemodel_path}")
    
    # Initialize video capture
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows backend
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)  # Try default
    
    if not cap.isOpened():
        raise CameraError("Cannot open camera. Please check if camera is connected and not being used by another application.")
    
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    writer = None
    if SAVE_OUTPUT:
        out_path = BASE / "multimodal-output.avi"
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            OUTPUT_FPS,
            (frame_w, frame_h),
        )
    
    emo_net = cv2.dnn.readNetFromONNX(str(onnx_path))
    face_net = dnn.readNetFromCaffe(str(proto_path), str(caffemodel_path))
    
    input_size = [320, 240]
    width, height = input_size
    priors = define_img_size(input_size)
    
    # Initialize emotion tracking
    emotions_by_second = defaultdict(list)
    start_time = time.time()
    frame_count = 0
    
    # Create window with controlled size
    window_name = "Multimodal Recording"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)  # Set window to 800x600 pixels
    
    print("🎥 Recording video...")
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        
        elapsed_time = time.time() - start_time
        current_second = int(elapsed_time)
        
        if elapsed_time >= RECORDING_DURATION:
            break
        
        rect = cv2.resize(frame, (width, height))
        rect_rgb = cv2.cvtColor(rect, cv2.COLOR_BGR2RGB)
        
        face_net.setInput(dnn.blobFromImage(rect_rgb, 1 / image_std, (width, height), 127))
        boxes_blob, scores_blob = face_net.forward(["boxes", "scores"])
        
        boxes = np.expand_dims(np.reshape(boxes_blob, (-1, 4)), axis=0)
        scores = np.expand_dims(np.reshape(scores_blob, (-1, 2)), axis=0)
        
        boxes = convert_locations_to_boxes(boxes, priors, center_variance, size_variance)
        boxes = center_form_to_corner_form(boxes)
        boxes, labels, probs = predict(frame.shape[1], frame.shape[0], scores, boxes, threshold)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        frame_emotions = []
        
        for (x1, y1, x2, y2) in boxes:
            x1 = max(0, min(int(x1), frame.shape[1] - 1))
            y1 = max(0, min(int(y1), frame.shape[0] - 1))
            x2 = max(0, min(int(x2), frame.shape[1] - 1))
            y2 = max(0, min(int(y2), frame.shape[0] - 1))
            w, h = x2 - x1, y2 - y1
            if w <= 1 or h <= 1:
                continue
            
            face_gray = gray[y1:y2, x1:x2]
            if face_gray.size == 0:
                continue
            
            face_resized = cv2.resize(face_gray, (64, 64), interpolation=cv2.INTER_AREA)
            blob = face_resized.reshape(1, 1, 64, 64).astype(np.float32)
            
            emo_net.setInput(blob)
            output = emo_net.forward()[0]
            pred_idx = int(np.argmax(output))
            pred = emotion_dict.get(pred_idx, "unknown")
            
            frame_emotions.append(pred)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (215, 5, 247), 2, lineType=cv2.LINE_AA)
            cv2.putText(
                frame,
                pred,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (215, 5, 247),
                2,
                lineType=cv2.LINE_AA,
            )
        
        emotions_by_second[current_second].extend(frame_emotions)
        
        remaining = RECORDING_DURATION - int(elapsed_time)
        cv2.putText(
            frame,
            f"Recording: {remaining}s left",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            lineType=cv2.LINE_AA,
        )
        
        if writer is not None:
            writer.write(frame)
        
        cv2.imshow(window_name, frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        frame_count += 1
    
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    
    print(f"✅ Video recording complete! ({frame_count} frames)")
    
    # Enhanced emotional analysis for LLM comprehension
    enhanced_emotion_data = {
        "emotions_by_second": dict(emotions_by_second),
        "duration": RECORDING_DURATION,
        "frame_count": frame_count,
        "emotional_analysis": _analyze_emotional_patterns(emotions_by_second, RECORDING_DURATION),
        "facial_expression_quality": _assess_facial_expression_quality(emotions_by_second, frame_count),
        "emotional_stability_metrics": _calculate_emotional_stability_metrics(emotions_by_second, RECORDING_DURATION)
    }
    
    # Store in memory
    analysis_memory.store_emotion_data(enhanced_emotion_data)
    
    return enhanced_emotion_data

def _analyze_emotional_patterns(emotions_by_second: Dict[int, List[str]], duration: int) -> Dict[str, Any]:
    """Analyze emotional patterns for better LLM comprehension"""
    try:
        all_emotions = []
        for second, emotions in emotions_by_second.items():
            all_emotions.extend(emotions)
        
        if not all_emotions:
            return {
                "dominant_emotion": "unknown",
                "emotion_distribution": {},
                "emotional_intensity": "unknown",
                "emotion_transitions": 0,
                "emotional_consistency": 0.0,
                "emotional_complexity": "unknown"
            }
        
        emotion_counts = Counter(all_emotions)
        total_detections = len(all_emotions)
        dominant_emotion = emotion_counts.most_common(1)[0][0]
        
        # Calculate emotion distribution percentages
        emotion_distribution = {
            emotion: (count / total_detections) * 100 
            for emotion, count in emotion_counts.items()
        }
        
        # Calculate emotional intensity based on distribution
        max_percentage = max(emotion_distribution.values())
        if max_percentage > 70:
            emotional_intensity = "high"
        elif max_percentage > 40:
            emotional_intensity = "medium"
        else:
            emotional_intensity = "low"
        
        # Count emotion transitions
        emotion_transitions = 0
        prev_emotion = None
        for second in sorted(emotions_by_second.keys()):
            if emotions_by_second[second]:
                curr_emotion = Counter(emotions_by_second[second]).most_common(1)[0][0]
                if prev_emotion and curr_emotion != prev_emotion:
                    emotion_transitions += 1
                prev_emotion = curr_emotion
        
        # Calculate emotional consistency
        emotional_consistency = ((duration - emotion_transitions) / duration) * 100 if duration > 0 else 0
        
        # Determine emotional complexity
        unique_emotions = len(emotion_counts)
        if unique_emotions == 1:
            emotional_complexity = "simple"
        elif unique_emotions <= 3:
            emotional_complexity = "moderate"
        else:
            emotional_complexity = "complex"
        
        return {
            "dominant_emotion": dominant_emotion,
            "emotion_distribution": emotion_distribution,
            "emotional_intensity": emotional_intensity,
            "emotion_transitions": emotion_transitions,
            "emotional_consistency": emotional_consistency,
            "emotional_complexity": emotional_complexity,
            "total_emotion_detections": total_detections,
            "unique_emotions_detected": unique_emotions
        }
        
    except Exception as e:
        return {
            "dominant_emotion": "unknown",
            "emotion_distribution": {},
            "emotional_intensity": "unknown",
            "emotion_transitions": 0,
            "emotional_consistency": 0.0,
            "emotional_complexity": "unknown",
            "error": str(e)
        }

def _assess_facial_expression_quality(emotions_by_second: Dict[int, List[str]], frame_count: int) -> Dict[str, Any]:
    """Assess the quality of facial expression detection"""
    try:
        total_detections = sum(len(emotions) for emotions in emotions_by_second.values())
        detection_rate = total_detections / frame_count if frame_count > 0 else 0
        
        # Calculate detection consistency
        seconds_with_detection = len([s for s, emotions in emotions_by_second.items() if emotions])
        detection_consistency = seconds_with_detection / len(emotions_by_second) if emotions_by_second else 0
        
        # Assess quality based on detection metrics
        if detection_rate > 0.8 and detection_consistency > 0.9:
            quality_level = "excellent"
        elif detection_rate > 0.6 and detection_consistency > 0.7:
            quality_level = "good"
        elif detection_rate > 0.4 and detection_consistency > 0.5:
            quality_level = "fair"
        else:
            quality_level = "poor"
        
        return {
            "detection_rate": detection_rate,
            "detection_consistency": detection_consistency,
            "quality_level": quality_level,
            "total_detections": total_detections,
            "seconds_with_detection": seconds_with_detection,
            "average_detections_per_second": total_detections / len(emotions_by_second) if emotions_by_second else 0
        }
        
    except Exception as e:
        return {
            "detection_rate": 0.0,
            "detection_consistency": 0.0,
            "quality_level": "unknown",
            "error": str(e)
        }

def _calculate_emotional_stability_metrics(emotions_by_second: Dict[int, List[str]], duration: int) -> Dict[str, Any]:
    """Calculate detailed emotional stability metrics"""
    try:
        # Calculate stability over time
        emotion_changes = 0
        prev_emotion = None
        stability_timeline = []
        
        for second in sorted(emotions_by_second.keys()):
            if emotions_by_second[second]:
                curr_emotion = Counter(emotions_by_second[second]).most_common(1)[0][0]
                stability_timeline.append(curr_emotion)
                
                if prev_emotion and curr_emotion != prev_emotion:
                    emotion_changes += 1
                prev_emotion = curr_emotion
            else:
                stability_timeline.append("no_detection")
        
        # Calculate various stability metrics
        stability_percentage = ((duration - emotion_changes) / duration) * 100 if duration > 0 else 0
        
        # Calculate emotional volatility (how much emotions change)
        volatility_score = emotion_changes / duration if duration > 0 else 0
        
        # Calculate emotional persistence (how long emotions last)
        emotion_durations = []
        current_emotion = None
        current_duration = 0
        
        for emotion in stability_timeline:
            if emotion == current_emotion:
                current_duration += 1
            else:
                if current_emotion and current_emotion != "no_detection":
                    emotion_durations.append(current_duration)
                current_emotion = emotion
                current_duration = 1
        
        if current_emotion and current_emotion != "no_detection":
            emotion_durations.append(current_duration)
        
        avg_emotion_duration = np.mean(emotion_durations) if emotion_durations else 0
        
        # Determine stability level
        if stability_percentage > 80 and volatility_score < 0.2:
            stability_level = "very_stable"
        elif stability_percentage > 60 and volatility_score < 0.4:
            stability_level = "stable"
        elif stability_percentage > 40 and volatility_score < 0.6:
            stability_level = "moderately_stable"
        elif stability_percentage > 20 and volatility_score < 0.8:
            stability_level = "variable"
        else:
            stability_level = "highly_variable"
        
        return {
            "stability_percentage": stability_percentage,
            "volatility_score": volatility_score,
            "emotion_changes": emotion_changes,
            "average_emotion_duration": avg_emotion_duration,
            "stability_level": stability_level,
            "stability_timeline": stability_timeline
        }
        
    except Exception as e:
        return {
            "stability_percentage": 0.0,
            "volatility_score": 0.0,
            "emotion_changes": 0,
            "average_emotion_duration": 0.0,
            "stability_level": "unknown",
            "error": str(e)
    }

# ---------- Audio Voice Analysis with Transcription ----------
def record_audio_analysis_with_transcription() -> Dict[str, Any]:
    """
    Record audio for 15 seconds, analyze voice, and transcribe speech
    Returns dictionary with voice analysis and transcription data
    """
    try:
        print("🎤 Recording audio...")
        
        analyzer = VoiceAnalyzer()
        
        # Record audio
        audio_data = analyzer.record_audio(duration=RECORDING_DURATION, sample_rate=22050)
        
        # Analyze audio
        print("🔊 Analyzing voice...")
        voice_results = analyzer.analyze_recorded_audio(audio_data)
        
        # Transcribe speech
        print("📝 Transcribing speech...")
        try:
            # Use credentials.json file in the project directory
            transcriber = SpeechTranscriber(credentials_path="credentials.json")
            transcription_results = transcriber.transcribe_audio_data(audio_data, sample_rate=22050)
            
            # Store voice and transcription data separately in memory
            analysis_memory.store_voice_data(voice_results)
            analysis_memory.store_transcription_data(transcription_results)
            
            # Combine voice analysis and transcription results
            combined_results = {
                **voice_results,
                "transcription": transcription_results
            }
            
            print("✅ Audio analysis and transcription complete!")
            return combined_results
            
        except Exception as e:
            print(f"⚠️  Speech transcription failed: {str(e)}")
            print("   Continuing with voice analysis only...")
            
            # Store voice data and empty transcription in memory
            analysis_memory.store_voice_data(voice_results)
            analysis_memory.store_transcription_data({
                "transcription": "",
                "confidence": 0.0,
                "success": False,
                "error": str(e),
                "word_count": 0
            })
            
            # Return voice analysis without transcription
            return {
                **voice_results,
                "transcription": {
                    "transcription": "",
                    "confidence": 0.0,
                    "success": False,
                    "error": str(e),
                    "word_count": 0
                }
            }
        
    except Exception as e:
        raise AudioError(f"Audio analysis failed: {str(e)}")

# ---------- Combined Analysis ----------
def format_combined_results(emotion_data: Dict[str, Any], voice_data: Dict[str, Any]):
    """
    Display combined emotion, voice analysis, and transcription results
    """
    print("\n" + "="*70)
    print("           MULTIMODAL ANALYSIS RESULTS")
    print("           (Emotion + Voice + Speech Transcription)")
    print("="*70)
    
    # ========== EMOTION ANALYSIS ==========
    if "error" not in emotion_data:
        emotions_by_second = emotion_data.get("emotions_by_second", {})
        
        all_emotions = []
        for second, emotions in emotions_by_second.items():
            all_emotions.extend(emotions)
        
        if all_emotions:
            emotion_counts = Counter(all_emotions)
            total_detections = len(all_emotions)
            dominant_emotion = emotion_counts.most_common(1)[0][0]
            
            emotion_emojis = {
                "neutral": "😐",
                "happiness": "😊",
                "surprise": "😲",
                "sadness": "😢",
                "anger": "😠",
                "disgust": "🤢",
                "fear": "😨"
            }
            
            print(f"\n{'='*70}")
            print("🎭 FACIAL EMOTION ANALYSIS")
            print(f"{'='*70}")
            
            print(f"\n📊 STATISTICS:")
            print(f"   Total Detections: {total_detections}")
            print(f"   Dominant Emotion: {emotion_emojis.get(dominant_emotion, '🎭')} {dominant_emotion.title()}")
            
            print(f"\n🎭 EMOTION DISTRIBUTION:")
            for emotion, count in emotion_counts.most_common():
                percentage = (count / total_detections) * 100
                emoji = emotion_emojis.get(emotion, "🎭")
                bar_length = int(percentage / 2.5)
                bar = "█" * bar_length
                print(f"   {emoji} {emotion.title():12} {bar} {percentage:5.1f}%")
            
            # Calculate emotional stability
            emotion_changes = 0
            prev_emotion = None
            for second in sorted(emotions_by_second.keys()):
                if emotions_by_second[second]:
                    curr_emotion = Counter(emotions_by_second[second]).most_common(1)[0][0]
                    if prev_emotion and curr_emotion != prev_emotion:
                        emotion_changes += 1
                    prev_emotion = curr_emotion
            
            stability_percentage = ((RECORDING_DURATION - emotion_changes) / RECORDING_DURATION) * 100
            print(f"\n💡 EMOTIONAL STABILITY: {stability_percentage:.0f}%")
            if stability_percentage > 80:
                print(f"   Very stable emotional state")
            elif stability_percentage > 60:
                print(f"   Moderately stable emotions")
            else:
                print(f"   Dynamic emotional expression")
        else:
            print(f"\n🎭 FACIAL EMOTION ANALYSIS")
            print("   ⚠️  No emotions detected")
    else:
        print(f"\n🎭 FACIAL EMOTION ANALYSIS")
        print(f"   ❌ Error: {emotion_data['error']}")
    
    # ========== VOICE ANALYSIS ==========
    print(f"\n{'='*70}")
    print("🎤 VOICE ANALYSIS")
    print(f"{'='*70}")
    
    if "error" not in voice_data:
        print(f"\n🎵 VOICE CHARACTERISTICS:")
        print(f"   Mean Pitch: {voice_data['mean_pitch']:.1f} Hz")
        print(f"   Voice Type: {voice_data['voice_type'].title()}")
        print(f"   Pitch Range: {voice_data['lowest_note']} - {voice_data['highest_note']}")
        
        print(f"\n🎤 VOICE QUALITY:")
        print(f"   Vibrato Rate: {voice_data['vibrato_rate']:.1f} Hz")
        jitter_level = 'Low' if voice_data['jitter'] < 0.01 else 'Medium' if voice_data['jitter'] < 0.02 else 'High'
        print(f"   Jitter: {voice_data['jitter']:.3f} ({jitter_level} stability)")
        shimmer_level = 'Low' if voice_data['shimmer'] < 0.015 else 'Medium' if voice_data['shimmer'] < 0.025 else 'High'
        print(f"   Shimmer: {voice_data['shimmer']:.3f} ({shimmer_level} variation)")
        
        # Singing Characteristics
        if 'singing_characteristics' in voice_data:
            singing = voice_data['singing_characteristics']
            print(f"\n🎵 SINGING CHARACTERISTICS:")
            print(f"   Style: {singing.get('singing_style', 'unknown').title()}")
            print(f"   Quality: {singing.get('overall_singing_quality', 'unknown').title()}")
            print(f"   Pitch Stability: {singing.get('pitch_stability', 0):.3f}")
            print(f"   Vibrato: {'Present' if singing.get('vibrato_present', False) else 'Not detected'}")
        
        # Emotional Voice Indicators
        if 'emotional_indicators' in voice_data:
            emotional = voice_data['emotional_indicators']
            print(f"\n🎭 EMOTIONAL VOICE INDICATORS:")
            print(f"   Energy Level: {emotional.get('energy_level', 'unknown').title()}")
            print(f"   Emotional Arousal: {emotional.get('emotional_arousal', 'unknown').title()}")
            print(f"   Voice Tension: {emotional.get('voice_tension', 'unknown').title()}")
            print(f"   Speaking Rate: {emotional.get('speaking_rate', 'unknown').title()}")
    else:
        print(f"   ❌ Error: {voice_data['error']}")
    
    # ========== SPEECH TRANSCRIPTION ==========
    print(f"\n{'='*70}")
    print("📝 SPEECH TRANSCRIPTION")
    print(f"{'='*70}")
    
    transcription_data = voice_data.get("transcription", {})
    if transcription_data.get("success", False):
        print(f"\n📝 TRANSCRIPTION:")
        print(f"   Text: \"{transcription_data['transcription']}\"")
        print(f"   Confidence: {transcription_data['confidence']:.1%}")
        print(f"   Word Count: {transcription_data['word_count']}")
        print(f"   Language: {transcription_data.get('language_code', 'en-US')}")
        
        if transcription_data.get("enhanced_model"):
            print(f"   Model: Enhanced")
        
        # Show word-level timing if available
        if "words" in transcription_data and transcription_data["words"]:
            print(f"\n⏱️  WORD TIMING:")
            for i, word_info in enumerate(transcription_data["words"][:10]):  # Show first 10 words
                start_time = word_info.get("start_time", 0)
                end_time = word_info.get("end_time", 0)
                confidence = word_info.get("confidence", 0)
                print(f"   {i+1:2d}. '{word_info['word']}' ({start_time:.1f}s-{end_time:.1f}s, {confidence:.1%})")
            
            if len(transcription_data["words"]) > 10:
                print(f"   ... and {len(transcription_data['words']) - 10} more words")
    else:
        print(f"\n❌ TRANSCRIPTION FAILED:")
        print(f"   Error: {transcription_data.get('error', 'Unknown error')}")
        print(f"   Success: {transcription_data.get('success', False)}")
    
    # ========== GEMINI AI ANALYSIS ==========
    if "error" not in emotion_data and "error" not in voice_data and all_emotions:
        print(f"\n{'='*70}")
        print("🤖 GEMINI AI COMPREHENSIVE ANALYSIS")
        print(f"{'='*70}")
        
        try:
            print("\n🔄 Generating AI-powered analysis...")
            print("   This may take a few moments...")
            
            # Get Gemini analysis
            gemini_response = get_gemini_analysis(emotion_data, voice_data)
            
            print(f"\n📊 GEMINI AI ANALYSIS RESULTS:")
            print("="*70)
            print(gemini_response)
            print("="*70)
            
        except GeminiError as e:
            print(f"\n❌ GEMINI AI ERROR: {str(e)}")
            print("\n💡 To enable AI-powered analysis:")
            print("   1. Install: pip install google-generativeai")
            print("   2. Set up API key (see GEMINI_SETUP.md)")
            print("   3. Check internet connection")
            print("   4. Restart the application")
            print("\n   Basic analysis results are still available above.")
            # Don't raise the exception here, just show the error and continue
    
    print("\n" + "="*70)
    print("Analysis complete! 🎉")
    print("="*70)
    
    # Generate and save JSON output for LLM integration
    generate_json_output()

def generate_json_output():
    """Generate comprehensive JSON output for LLM integration"""
    try:
        # Get complete analysis data
        complete_analysis = analysis_memory.get_complete_analysis()
        
        # Create output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"multimodal_analysis_{timestamp}.json"
        
        # Save to file
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(complete_analysis, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 JSON output saved to: {output_filename}")
        print("   This file contains all analysis data for LLM integration")
        
        # Display summary for user
        print(f"\n📊 ANALYSIS SUMMARY FOR LLM:")
        print(f"   Session ID: {complete_analysis['metadata']['session_id']}")
        print(f"   Status: {complete_analysis['metadata']['status']}")
        
        if 'multimodal_insights' in complete_analysis:
            insights = complete_analysis['multimodal_insights']
            print(f"   Overall Emotional State: {insights.get('overall_emotional_state', 'unknown')}")
            print(f"   Confidence Score: {insights.get('confidence_score', 0):.2f}")
            print(f"   Emotional Coherence: {insights.get('emotional_coherence', 'unknown')}")
            print(f"   Voice-Emotion Alignment: {insights.get('voice_emotion_alignment', 'unknown')}")
        
        return output_filename
        
    except Exception as e:
        print(f"❌ Error generating JSON output: {str(e)}")
        return None

# ---------- Main Function ----------
def run_multimodal_analysis():
    """
    Run simultaneous video and audio recording with analysis and transcription
    """
    try:
        # Reset memory for new session
        global analysis_memory
        analysis_memory = AnalysisMemory()
        
        print("\n" + "="*70)
        print("      🎭🎤📝 MULTIMODAL EMOTION, VOICE & SPEECH ANALYSIS 📝🎤🎭")
        print("="*70)
        print("\nThis tool will simultaneously:")
        print("  • Record video for 15 seconds and detect facial emotions")
        print("  • Record audio for 15 seconds and analyze voice characteristics")
        print("  • Transcribe your speech using Google Speech-to-Text")
        print("  • Provide comprehensive multimodal analysis")
        print("  • Generate AI-powered insights using Google Gemini")
        print("  • Generate JSON output for LLM integration")
        print("\nMake sure your webcam and microphone are ready!")
        print("="*70)
        
        input("\nPress ENTER to start recording...")
        
        print("\n" + "="*70)
        print("🎬 STARTING MULTIMODAL RECORDING...")
        print("="*70)
        print("Recording will start in 3 seconds...")
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        print("🔴 RECORDING NOW!\n")
        
        # Storage for results
        emotion_results = {}
        voice_results = {}
        audio_error = None
        
        # Run video recording in main thread (needs GUI)
        # Run audio recording with transcription in separate thread
        def audio_worker():
            nonlocal voice_results, audio_error
            try:
                voice_results.update(record_audio_analysis_with_transcription())
            except Exception as e:
                audio_error = e
        
        audio_thread = threading.Thread(target=audio_worker)
        
        # Start audio thread
        audio_thread.start()
        
        # Run video in main thread (OpenCV needs main thread for GUI)
        try:
            emotion_results = record_video_emotions()
        except Exception as e:
            print(f"\n❌ Video recording failed: {str(e)}")
            raise
        
        # Wait for audio thread to complete
        audio_thread.join()
        
        # Check if audio thread had errors
        if audio_error:
            print(f"\n❌ Audio recording failed: {str(audio_error)}")
            raise audio_error
        
        print("\n" + "="*70)
        print("📊 Processing and analyzing results...")
        print("="*70)
        
        # Display combined results
        format_combined_results(emotion_results, voice_results)
        
    except CameraError as e:
        print(f"\n❌ CAMERA ERROR: {str(e)}")
        print("\n💡 Troubleshooting steps:")
        print("   • Check if camera is connected")
        print("   • Close other applications using the camera")
        print("   • Check camera permissions")
        print("   • Try restarting the application")
        sys.exit(1)
        
    except ModelError as e:
        print(f"\n❌ MODEL ERROR: {str(e)}")
        print("\n💡 Troubleshooting steps:")
        print("   • Ensure emotion-ferplus-8.onnx is in the project directory")
        print("   • Ensure RFB-320 folder with .prototxt and .caffemodel files exists")
        print("   • Check file permissions")
        sys.exit(1)
        
    except AudioError as e:
        print(f"\n❌ AUDIO ERROR: {str(e)}")
        print("\n💡 Troubleshooting steps:")
        print("   • Check if microphone is connected and working")
        print("   • Check microphone permissions")
        print("   • Close other applications using the microphone")
        print("   • Try speaking louder and closer to the microphone")
        sys.exit(1)
        
    except GeminiError as e:
        print(f"\n❌ GEMINI AI ERROR: {str(e)}")
        print("\n💡 Troubleshooting steps:")
        print("   • Install: pip install google-generativeai")
        print("   • Set up API key (see GEMINI_SETUP.md)")
        print("   • Check internet connection")
        print("   • Verify API key is valid")
        sys.exit(1)
        
    except TranscriptionError as e:
        print(f"\n❌ TRANSCRIPTION ERROR: {str(e)}")
        print("\n💡 Troubleshooting steps:")
        print("   • Check Google Cloud credentials")
        print("   • Verify internet connection")
        print("   • Check microphone quality")
        print("   • Try speaking more clearly")
        sys.exit(1)
        
    except (VoiceAnalysisError, InsufficientDataError, AudioQualityError) as e:
        print(f"\n❌ VOICE ANALYSIS ERROR: {str(e)}")
        print("\n💡 Troubleshooting steps:")
        print("   • Check audio quality")
        print("   • Try speaking louder")
        print("   • Ensure minimal background noise")
        print("   • Try a longer recording")
        print("   • Check microphone connection and permissions")
        sys.exit(1)
        
    except MultimodalAnalysisError as e:
        print(f"\n❌ ANALYSIS ERROR: {str(e)}")
        print("\n💡 Please check the error message above for specific guidance.")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n❌ Analysis interrupted by user.")
        print("👋 Thank you for using Multimodal Analysis Tool!")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {str(e)}")
        print("\n💡 This is an unexpected error. Please:")
        print("   • Check all dependencies are installed")
        print("   • Verify all required files are present")
        print("   • Try restarting the application")
        print("   • Contact support if the issue persists")
        sys.exit(1)

if __name__ == "__main__":
    run_multimodal_analysis()

