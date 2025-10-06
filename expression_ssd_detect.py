"""
Emotion Detection - 15 Second Recording Mode
Records video for 15 seconds and analyzes emotions per second
"""

import cv2
import numpy as np
import time
from cv2 import dnn
from math import ceil
from pathlib import Path
import sys
from collections import defaultdict, Counter

# ---------- Config ----------
USE_WEBCAM = True              # True = webcam, False = video file
VIDEO_FILE = "video1.mp4"      # used if USE_WEBCAM is False
SAVE_OUTPUT = True             # write annotated AVI next to this script
OUTPUT_FPS = 10
RECORDING_DURATION = 15        # seconds to record

# ---------- Globals ----------
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

# ---------------face detection ends here-----------

def format_emotion_results(emotions_by_second, total_duration):
    """
    Format and display emotion analysis results in a user-friendly way
    
    Args:
        emotions_by_second: Dictionary mapping second -> list of detected emotions
        total_duration: Total recording duration in seconds
    """
    print("\n" + "="*60)
    print("           EMOTION ANALYSIS RESULTS")
    print("="*60)
    
    # Flatten all emotions into a single list
    all_emotions = []
    for second, emotions in emotions_by_second.items():
        all_emotions.extend(emotions)
    
    if not all_emotions:
        print("\n⚠️  No emotions detected during the recording.")
        print("   Please ensure:")
        print("   • Your face is clearly visible")
        print("   • There is adequate lighting")
        print("   • You're facing the camera")
        print("\n" + "="*60)
        return
    
    # Calculate statistics
    emotion_counts = Counter(all_emotions)
    total_detections = len(all_emotions)
    dominant_emotion = emotion_counts.most_common(1)[0][0]
    
    # Emotion emojis
    emotion_emojis = {
        "neutral": "😐",
        "happiness": "😊",
        "surprise": "😲",
        "sadness": "😢",
        "anger": "😠",
        "disgust": "🤢",
        "fear": "😨"
    }
    
    # Display overall statistics
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   Recording Duration: {total_duration} seconds")
    print(f"   Total Detections: {total_detections}")
    print(f"   Dominant Emotion: {emotion_emojis.get(dominant_emotion, '🎭')} {dominant_emotion.title()}")
    
    # Display emotion distribution
    print(f"\n🎭 EMOTION DISTRIBUTION:")
    for emotion, count in emotion_counts.most_common():
        percentage = (count / total_detections) * 100
        emoji = emotion_emojis.get(emotion, "🎭")
        bar_length = int(percentage / 2)  # Scale bar to max 50 chars
        bar = "█" * bar_length
        print(f"   {emoji} {emotion.title():12} {bar} {percentage:5.1f}% ({count})")
    
    # Display timeline
    print(f"\n⏱️  EMOTION TIMELINE:")
    for second in sorted(emotions_by_second.keys()):
        emotions = emotions_by_second[second]
        if emotions:
            # Get most common emotion for this second
            second_emotion = Counter(emotions).most_common(1)[0][0]
            emoji = emotion_emojis.get(second_emotion, "🎭")
            print(f"   Second {second:2d}: {emoji} {second_emotion.title()}")
        else:
            print(f"   Second {second:2d}: ❌ No face detected")
    
    # Display insights
    print(f"\n💡 INSIGHTS:")
    unique_emotions = len(emotion_counts)
    if unique_emotions == 1:
        print(f"   • Consistent emotional state throughout recording")
    elif unique_emotions <= 3:
        print(f"   • Relatively stable emotions with {unique_emotions} different states")
    else:
        print(f"   • Highly variable emotions with {unique_emotions} different states")
    
    # Calculate emotional stability
    emotion_changes = 0
    prev_emotion = None
    for second in sorted(emotions_by_second.keys()):
        if emotions_by_second[second]:
            curr_emotion = Counter(emotions_by_second[second]).most_common(1)[0][0]
            if prev_emotion and curr_emotion != prev_emotion:
                emotion_changes += 1
            prev_emotion = curr_emotion
    
    stability_percentage = ((total_duration - emotion_changes) / total_duration) * 100
    if stability_percentage > 80:
        print(f"   • Very stable emotional state ({stability_percentage:.0f}% stability)")
    elif stability_percentage > 60:
        print(f"   • Moderately stable emotions ({stability_percentage:.0f}% stability)")
    else:
        print(f"   • Dynamic emotional expression ({stability_percentage:.0f}% stability)")
    
    print("\n" + "="*60)
    print("Analysis complete! 🎉")
    print("="*60)

# ---------- Main ----------
def FER_live_cam():
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
        sys.exit(1)
    if not proto_path.exists() or not caffemodel_path.exists():
        print(f"[ERROR] Missing face detector files:\n - {proto_path}\n - {caffemodel_path}")
        sys.exit(1)

    if USE_WEBCAM:
        # macOS backend; try default; if fails, try index 1
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(str(BASE / VIDEO_FILE))

    if not cap.isOpened():
        print("[ERROR] Cannot open camera/video")
        sys.exit(1)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if SAVE_OUTPUT:
        out_path = BASE / "infer2-test.avi"
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            OUTPUT_FPS,
            (frame_w, frame_h),
        )

    emo_net = cv2.dnn.readNetFromONNX(str(onnx_path))
    face_net = dnn.readNetFromCaffe(str(proto_path), str(caffemodel_path))

    input_size = [320, 240]  # width, height
    width, height = input_size
    priors = define_img_size(input_size)

    # Initialize emotion tracking
    emotions_by_second = defaultdict(list)
    start_time = time.time()
    frame_count = 0
    
    print("\n" + "="*60)
    print(f"🎥 Recording for {RECORDING_DURATION} seconds...")
    print("Please look at the camera and show your emotions!")
    print("="*60 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("No frame received. Exiting...")
            break

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        current_second = int(elapsed_time)
        
        # Stop after recording duration
        if elapsed_time >= RECORDING_DURATION:
            break

        start_t = time.time()

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

        # Track emotions for this frame
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
            output = emo_net.forward()[0]  # FER+ returns 8 emotions in some variants; mapping used here is 7
            pred_idx = int(np.argmax(output))
            pred = emotion_dict.get(pred_idx, "unknown")
            
            # Store emotion for this second
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
        
        # Add detected emotions to the current second
        emotions_by_second[current_second].extend(frame_emotions)

        # Display recording progress
        remaining = RECORDING_DURATION - int(elapsed_time)
        fps = 1.0 / max(1e-6, (time.time() - start_t))
        
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
        
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            lineType=cv2.LINE_AA,
        )

        if writer is not None:
            writer.write(frame)

        cv2.imshow("Emotion Recording", frame)
        
        # Allow early exit with 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\nRecording stopped by user.")
            break
        
        frame_count += 1
        
        # Print progress to console
        if current_second > 0 and frame_count % 30 == 0:  # Print every ~1 second
            print(f"⏱️  Recording... {current_second}/{RECORDING_DURATION} seconds")

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Recording complete! Captured {frame_count} frames.")
    print("📊 Processing results...\n")
    
    # Format and display results
    format_emotion_results(emotions_by_second, RECORDING_DURATION)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("      🎭 EMOTION DETECTION - 15 SECOND ANALYSIS 🎭")
    print("="*60)
    print("\nThis tool will:")
    print("  • Record video for 15 seconds from your webcam")
    print("  • Detect facial emotions in real-time")
    print("  • Track emotions per second")
    print("  • Provide detailed analysis and insights")
    print("\nMake sure your webcam is connected and ready!")
    print("="*60)
    
    input("\nPress ENTER to start recording...")
    
    FER_live_cam()

