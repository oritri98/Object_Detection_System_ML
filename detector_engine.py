"""
Project HYDRA Cam - Detection Engine
MediaPipe Object Detection with Luxury Black Diamond & Electrum Gold theme.
"""

import os
import sys
import time
import urllib.request
from typing import Optional, Dict, Any, List
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite"
DEFAULT_MODEL_PATH = "efficientdet_lite0.tflite"

# BGR Colors for OpenCV
THEME_COLORS = {
    "electrum_gold": {
        "box": (156, 203, 231),       # #E7CB9C Electrum Gold in BGR
        "badge_bg": (156, 203, 231),  # Gold badge fill
        "text": (25, 23, 23),         # #171719 Obsidian text for max readability
        "accent": (255, 253, 232)     # #E8FDFF Milky Quartz accent
    },
    "ocean_cyan": {
        "box": (254, 242, 0),
        "badge_bg": (254, 242, 0),
        "text": (10, 15, 30),
        "accent": (255, 200, 0)
    }
}


def ensure_model_exists(model_path: str = DEFAULT_MODEL_PATH) -> bool:
    if not os.path.exists(model_path):
        print(f"[*] Downloading MediaPipe model '{model_path}'...")
        try:
            urllib.request.urlretrieve(MODEL_URL, model_path)
            print(f"[+] Download complete ({os.path.getsize(model_path) / (1024 * 1024):.1f} MB).")
            return True
        except Exception as e:
            print(f"[!] Error downloading model: {e}")
            return False
    return True


class HydraDetector:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, threshold: float = 0.5, max_results: int = 15):
        self.model_path = model_path
        self.threshold = float(threshold)
        self.max_results = int(max_results)
        self.detector: Optional[vision.ObjectDetector] = None

        # Display Preferences
        self.label_mode = "numbered_hybrid"
        self.color_theme = "electrum_gold"
        self.show_boxes = True
        self.show_labels = True
        self.show_scores = True
        self.show_reticles = True

        ensure_model_exists(self.model_path)
        self._init_detector()

    def _init_detector(self):
        try:
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.ObjectDetectorOptions(
                base_options=base_options,
                score_threshold=self.threshold,
                max_results=self.max_results,
                running_mode=vision.RunningMode.IMAGE,
            )
            self.detector = vision.ObjectDetector.create_from_options(options)
        except Exception as e:
            print(f"[!] Error initializing detector: {e}")

    def update_config(self, threshold: float = None, max_results: int = None,
                      label_mode: str = None, color_theme: str = None,
                      show_boxes: bool = None, show_labels: bool = None,
                      show_scores: bool = None, show_reticles: bool = None):
        reinit_needed = False
        if threshold is not None and abs(threshold - self.threshold) > 0.01:
            self.threshold = float(np.clip(threshold, 0.05, 0.99))
            reinit_needed = True

        if max_results is not None and max_results != self.max_results:
            self.max_results = max(1, min(50, int(max_results)))
            reinit_needed = True

        if label_mode is not None:
            self.label_mode = label_mode
        if color_theme is not None and color_theme in THEME_COLORS:
            self.color_theme = color_theme
        if show_boxes is not None:
            self.show_boxes = bool(show_boxes)
        if show_labels is not None:
            self.show_labels = bool(show_labels)
        if show_scores is not None:
            self.show_scores = bool(show_scores)
        if show_reticles is not None:
            self.show_reticles = bool(show_reticles)

        if reinit_needed:
            self._init_detector()

    def detect_and_annotate(self, frame: np.ndarray):
        if frame is None or frame.size == 0 or self.detector is None:
            return frame, []

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        try:
            detection_result = self.detector.detect(mp_image)
        except Exception as e:
            print(f"[!] Detection error: {e}")
            return frame, []

        annotated_frame = frame.copy()
        detections_data = []

        if not detection_result or not detection_result.detections:
            return annotated_frame, detections_data

        theme = THEME_COLORS.get(self.color_theme, THEME_COLORS["electrum_gold"])
        box_color = theme["box"]
        badge_color = theme["badge_bg"]
        text_color = theme["text"]
        accent_color = theme["accent"]

        for idx, detection in enumerate(detection_result.detections, start=1):
            bbox = detection.bounding_box
            x = max(0, int(bbox.origin_x))
            y = max(0, int(bbox.origin_y))
            box_w = min(int(bbox.width), w - x)
            box_h = min(int(bbox.height), h - y)

            if box_w <= 0 or box_h <= 0:
                continue

            category_name = "Object"
            score = 0.0
            if detection.categories:
                cat = detection.categories[0]
                category_name = (cat.category_name or "Object").capitalize()
                score = float(cat.score)

            if self.label_mode == "numbered_simple":
                display_label = f"Object {idx}"
            elif self.label_mode == "numbered_category":
                display_label = f"Object {idx}: {category_name}"
            elif self.label_mode == "standard":
                display_label = f"{category_name} {int(score * 100)}%"
            else:
                if self.show_scores:
                    display_label = f"Object {idx}: {category_name} ({int(score * 100)}%)"
                else:
                    display_label = f"Object {idx}: {category_name}"

            detections_data.append({
                "id": idx,
                "raw_label": category_name,
                "display_label": display_label,
                "score": round(score * 100, 1),
                "box": [x, y, box_w, box_h]
            })

            # Bounding Box with luxury gold styling
            if self.show_boxes:
                cv2.rectangle(annotated_frame, (x-1, y-1), (x + box_w + 1, y + box_h + 1), (15, 15, 18), 1)
                cv2.rectangle(annotated_frame, (x, y), (x + box_w, y + box_h), box_color, 2)

                if self.show_reticles:
                    corner_len = min(16, box_w // 4, box_h // 4)
                    if corner_len > 3:
                        thick = 3
                        cv2.line(annotated_frame, (x, y), (x + corner_len, y), accent_color, thick)
                        cv2.line(annotated_frame, (x, y), (x, y + corner_len), accent_color, thick)
                        cv2.line(annotated_frame, (x + box_w, y), (x + box_w - corner_len, y), accent_color, thick)
                        cv2.line(annotated_frame, (x + box_w, y), (x + box_w, y + corner_len), accent_color, thick)
                        cv2.line(annotated_frame, (x, y + box_h), (x + corner_len, y + box_h), accent_color, thick)
                        cv2.line(annotated_frame, (x, y + box_h), (x, y + box_h - corner_len), accent_color, thick)
                        cv2.line(annotated_frame, (x + box_w, y + box_h), (x + box_w - corner_len, y + box_h), accent_color, thick)
                        cv2.line(annotated_frame, (x + box_w, y + box_h), (x + box_w, y + box_h - corner_len), accent_color, thick)

            if self.show_labels:
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.52
                text_thickness = 1
                (text_w, text_h), baseline = cv2.getTextSize(display_label, font, font_scale, text_thickness)

                badge_h = text_h + 10
                badge_w = text_w + 14
                badge_y1 = max(0, y - badge_h) if (y - badge_h) > 0 else y
                badge_y2 = badge_y1 + badge_h
                badge_x2 = min(w, x + badge_w)

                cv2.rectangle(annotated_frame, (x, badge_y1), (badge_x2, badge_y2), badge_color, -1)
                cv2.rectangle(annotated_frame, (x, badge_y1), (badge_x2, badge_y2), (255, 253, 232), 1)

                cv2.putText(
                    annotated_frame,
                    display_label,
                    (x + 7, badge_y1 + text_h + 4),
                    font,
                    font_scale,
                    text_color,
                    text_thickness,
                    cv2.LINE_AA,
                )

        return annotated_frame, detections_data
