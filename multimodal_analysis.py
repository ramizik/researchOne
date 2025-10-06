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
from typing import Dict, Any, Optional
import warnings

# Import voice analyzer and speech transcriber
from voice_analyzer import VoiceAnalyzer
from speech_transcriber import SpeechTranscriber

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

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
        print(f"[ERROR] Missing ONNX model: {onnx_path}")
        return {"error": "Missing ONNX model"}
    if not proto_path.exists() or not caffemodel_path.exists():
        print(f"[ERROR] Missing face detector files")
        return {"error": "Missing face detector files"}
    
    # Initialize video capture
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows backend
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)  # Try default
    
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return {"error": "Cannot open camera"}
    
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
        
        cv2.imshow("Multimodal Recording", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        frame_count += 1
    
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    
    print(f"✅ Video recording complete! ({frame_count} frames)")
    
    return {
        "emotions_by_second": dict(emotions_by_second),
        "duration": RECORDING_DURATION,
        "frame_count": frame_count
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
        print(f"❌ Audio analysis failed: {str(e)}")
        return {"error": str(e)}

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
        
        dynamics_emoji = {
            'stable': '🔒',
            'controlled': '🎯',
            'variable': '📈',
            'expressive': '🎭',
            'highly expressive': '🌟'
        }
        print(f"\n🎨 DYNAMICS:")
        print(f"   Style: {dynamics_emoji.get(voice_data['dynamics'], '🎵')} {voice_data['dynamics'].title()}")
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
    
    # ========== CORRELATION INSIGHTS ==========
    if "error" not in emotion_data and "error" not in voice_data and all_emotions:
        print(f"\n{'='*70}")
        print("🔗 MULTIMODAL INSIGHTS")
        print(f"{'='*70}")
        
        # Analyze correlation between emotion and voice dynamics
        voice_dynamics = voice_data.get('dynamics', 'stable')
        
        print(f"\n💭 OBSERVATIONS:")
        
        # Match emotions with voice characteristics
        if dominant_emotion == "happiness" and voice_data['vibrato_rate'] > 5:
            print(f"   • Happy emotional state matches expressive voice quality")
        elif dominant_emotion in ["sadness", "neutral"] and voice_dynamics in ["stable", "controlled"]:
            print(f"   • Calm emotional state aligns with controlled voice dynamics")
        elif dominant_emotion in ["anger", "surprise"] and voice_dynamics in ["expressive", "highly expressive"]:
            print(f"   • Intense emotions correlate with dynamic voice expression")
        
        # Pitch and emotion correlation
        if voice_data['mean_pitch'] > 250 and dominant_emotion in ["happiness", "surprise"]:
            print(f"   • Higher pitch frequency matches positive/excited emotions")
        elif voice_data['mean_pitch'] < 200 and dominant_emotion in ["sadness", "anger", "neutral"]:
            print(f"   • Lower pitch frequency aligns with subdued emotions")
        
        # Stability analysis
        if stability_percentage > 70 and voice_dynamics in ["stable", "controlled"]:
            print(f"   • Consistent emotional and vocal presentation")
        elif stability_percentage < 50 and voice_dynamics in ["expressive", "highly expressive"]:
            print(f"   • Variable emotions match expressive vocal style")
        
        # Speech content analysis
        if transcription_data.get("success", False):
            transcript = transcription_data['transcription'].lower()
            word_count = transcription_data['word_count']
            
            print(f"\n🗣️  SPEECH CONTENT ANALYSIS:")
            print(f"   • Spoke {word_count} words in {RECORDING_DURATION} seconds")
            print(f"   • Speaking rate: {word_count / RECORDING_DURATION:.1f} words/second")
            
            # Analyze emotional content in speech
            positive_words = ['happy', 'good', 'great', 'wonderful', 'amazing', 'excellent', 'fantastic']
            negative_words = ['sad', 'bad', 'terrible', 'awful', 'horrible', 'disappointed', 'angry']
            
            positive_count = sum(1 for word in positive_words if word in transcript)
            negative_count = sum(1 for word in negative_words if word in transcript)
            
            if positive_count > negative_count:
                print(f"   • Speech content appears positive ({positive_count} positive words)")
            elif negative_count > positive_count:
                print(f"   • Speech content appears negative ({negative_count} negative words)")
            else:
                print(f"   • Speech content appears neutral")
        
        print(f"\n   Overall: Your facial expressions, voice characteristics, and speech content")
        print(f"   show {'strong alignment' if stability_percentage > 60 else 'dynamic variation'}")
    
    print("\n" + "="*70)
    print("Analysis complete! 🎉")
    print("="*70)

# ---------- Main Function ----------
def run_multimodal_analysis():
    """
    Run simultaneous video and audio recording with analysis and transcription
    """
    print("\n" + "="*70)
    print("      🎭🎤📝 MULTIMODAL EMOTION, VOICE & SPEECH ANALYSIS 📝🎤🎭")
    print("="*70)
    print("\nThis tool will simultaneously:")
    print("  • Record video for 15 seconds and detect facial emotions")
    print("  • Record audio for 15 seconds and analyze voice characteristics")
    print("  • Transcribe your speech using Google Speech-to-Text")
    print("  • Provide comprehensive multimodal analysis")
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
    
    # Run video recording in main thread (needs GUI)
    # Run audio recording with transcription in separate thread
    audio_thread = threading.Thread(target=lambda: voice_results.update(record_audio_analysis_with_transcription()))
    
    # Start audio thread
    audio_thread.start()
    
    # Run video in main thread (OpenCV needs main thread for GUI)
    emotion_results = record_video_emotions()
    
    # Wait for audio thread to complete
    audio_thread.join()
    
    print("\n" + "="*70)
    print("📊 Processing and analyzing results...")
    print("="*70)
    
    # Display combined results
    format_combined_results(emotion_results, voice_results)

if __name__ == "__main__":
    try:
        run_multimodal_analysis()
    except KeyboardInterrupt:
        print("\n\n❌ Analysis interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n👋 Thank you for using Multimodal Analysis Tool!")

