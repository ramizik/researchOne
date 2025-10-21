"""
Analysis Engine - Wrapper for multimodal analysis components
Handles video and audio processing for the API server
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

# ==================== Emotion Detection Engine ====================

class EmotionAnalysisEngine:
    """Handles facial emotion detection from video files"""

    def __init__(self):
        """Initialize emotion detection models"""
        self.emotion_dict = {
            0: "neutral",
            1: "happiness",
            2: "surprise",
            3: "sadness",
            4: "anger",
            5: "disgust",
            6: "fear",
        }

        # Get model paths
        BASE = Path(__file__).resolve().parent
        self.onnx_path = BASE / "emotion-ferplus-8.onnx"
        self.proto_path = BASE / "RFB-320" / "RFB-320.prototxt"
        self.caffemodel_path = BASE / "RFB-320" / "RFB-320.caffemodel"

        # Validate model files
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")
        if not self.proto_path.exists() or not self.caffemodel_path.exists():
            raise FileNotFoundError("Face detection model files not found")

        # Load models
        self.emo_net = cv2.dnn.readNetFromONNX(str(self.onnx_path))
        self.face_net = cv2.dnn.readNetFromCaffe(str(self.proto_path), str(self.caffemodel_path))

        # SSD configuration
        self.input_size = [320, 240]
        self.width, self.height = self.input_size
        self.priors = self._define_img_size(self.input_size)

        self.center_variance = 0.1
        self.size_variance = 0.2
        self.threshold = 0.5

        logger.info("Emotion detection models loaded successfully")

    def _define_img_size(self, image_size):
        """Define image size for SSD"""
        from math import ceil

        strides = [8.0, 16.0, 32.0, 64.0]
        min_boxes = [
            [10.0, 16.0, 24.0],
            [32.0, 48.0],
            [64.0, 96.0],
            [128.0, 192.0, 256.0],
        ]

        shrinkage_list = []
        feature_map_w_h_list = []
        for size in image_size:
            feature_map = [int(ceil(size / stride)) for stride in strides]
            feature_map_w_h_list.append(feature_map)
        for _ in range(0, len(image_size)):
            shrinkage_list.append(strides)
        priors = self._generate_priors(feature_map_w_h_list, shrinkage_list, image_size, min_boxes)
        return priors

    def _generate_priors(self, feature_map_list, shrinkage_list, image_size, min_boxes):
        """Generate prior boxes for SSD"""
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

    def _convert_locations_to_boxes(self, locations, priors):
        """Convert locations to boxes"""
        if len(priors.shape) + 1 == len(locations.shape):
            priors = np.expand_dims(priors, 0)
        return np.concatenate(
            [
                locations[..., :2] * self.center_variance * priors[..., 2:] + priors[..., :2],
                np.exp(locations[..., 2:] * self.size_variance) * priors[..., 2:],
            ],
            axis=len(locations.shape) - 1,
        )

    def _center_form_to_corner_form(self, locations):
        """Convert center form to corner form"""
        return np.concatenate(
            [locations[..., :2] - locations[..., 2:] / 2, locations[..., :2] + locations[..., 2:] / 2],
            len(locations.shape) - 1,
        )

    def _predict(self, width, height, confidences, boxes):
        """Predict faces in image"""
        from multimodal_analysis import hard_nms, iou_of, area_of

        boxes = boxes[0]
        confidences = confidences[0]
        picked_box_probs = []
        picked_labels = []

        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > self.threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = hard_nms(box_probs, iou_threshold=0.3, top_k=-1)
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

    def analyze_video(self, video_path: Path) -> Dict[str, Any]:
        """
        Analyze video file for emotions

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with emotion analysis results
        """
        try:
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise Exception(f"Cannot open video file: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            emotions_by_second = defaultdict(list)
            frame_count = 0

            logger.info(f"Processing video at {fps} FPS")

            # Process frames
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                current_second = int(frame_count / fps)

                # Resize frame
                rect = cv2.resize(frame, (self.width, self.height))
                rect_rgb = cv2.cvtColor(rect, cv2.COLOR_BGR2RGB)

                # Face detection
                self.face_net.setInput(cv2.dnn.blobFromImage(rect_rgb, 1 / 128.0, (self.width, self.height), 127))
                boxes_blob, scores_blob = self.face_net.forward(["boxes", "scores"])

                boxes = np.expand_dims(np.reshape(boxes_blob, (-1, 4)), axis=0)
                scores = np.expand_dims(np.reshape(scores_blob, (-1, 2)), axis=0)

                boxes = self._convert_locations_to_boxes(boxes, self.priors)
                boxes = self._center_form_to_corner_form(boxes)
                boxes, labels, probs = self._predict(frame.shape[1], frame.shape[0], scores, boxes)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Process detected faces
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

                    self.emo_net.setInput(blob)
                    output = self.emo_net.forward()[0]
                    pred_idx = int(np.argmax(output))
                    pred = self.emotion_dict.get(pred_idx, "unknown")

                    emotions_by_second[current_second].append(pred)

                frame_count += 1

            cap.release()

            # Import analysis functions
            from multimodal_analysis import (
                _analyze_emotional_patterns,
                _assess_facial_expression_quality,
                _calculate_emotional_stability_metrics
            )

            # Build comprehensive results
            duration = max(emotions_by_second.keys()) + 1 if emotions_by_second else 0

            results = {
                "emotions_by_second": dict(emotions_by_second),
                "duration": duration,
                "frame_count": frame_count,
                "emotional_analysis": _analyze_emotional_patterns(emotions_by_second, duration),
                "facial_expression_quality": _assess_facial_expression_quality(emotions_by_second, frame_count),
                "emotional_stability_metrics": _calculate_emotional_stability_metrics(emotions_by_second, duration),
                "analysis_timestamp": np.datetime64('now').astype(str)
            }

            logger.info(f"Video analysis complete: {frame_count} frames, {duration} seconds")
            return results

        except Exception as e:
            logger.error(f"Error analyzing video: {e}", exc_info=True)
            raise

# ==================== Singleton Instance ====================

_emotion_engine = None

def get_emotion_engine() -> EmotionAnalysisEngine:
    """Get or create the emotion analysis engine singleton"""
    global _emotion_engine
    if _emotion_engine is None:
        _emotion_engine = EmotionAnalysisEngine()
    return _emotion_engine
