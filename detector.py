"""
Real-time Object Detection System using OpenCV and MediaPipe.
Detects objects from webcam feed and marks them with green bounding boxes.
"""

import argparse
import os
import sys
import time
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite"
DEFAULT_MODEL_PATH = "efficientdet_lite0.tflite"


def ensure_model_exists(model_path: str = DEFAULT_MODEL_PATH):
    """Checks if the TFLite model file exists locally; downloads it if missing."""
    if not os.path.exists(model_path):
        print(f"[*] Model '{model_path}' not found. Downloading from MediaPipe repository...")
        try:
            urllib.request.urlretrieve(MODEL_URL, model_path)
            print(f"[+] Successfully downloaded '{model_path}' ({os.path.getsize(model_path) / (1024 * 1024):.1f} MB).")
        except Exception as e:
            print(f"[!] Error downloading model: {e}")
            print(f"[!] Please manually download from: {MODEL_URL}")
            sys.exit(1)


def create_detector(model_path: str, score_threshold: float, max_results: int = 10):
    """Initializes and returns a MediaPipe ObjectDetector instance."""
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        score_threshold=score_threshold,
        max_results=max_results,
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.ObjectDetector.create_from_options(options)


def draw_detections(frame, detection_result, box_color=(0, 255, 0)):
    """
    Draws green bounding boxes and label tags on the frame for each detected object.
    
    Args:
        frame: The OpenCV BGR image frame.
        detection_result: MediaPipe ObjectDetector detection result.
        box_color: BGR tuple for bounding box (default is vibrant green: (0, 255, 0)).
    
    Returns:
        frame with rendered detections and the total object count.
    """
    h, w, _ = frame.shape
    num_objects = 0

    if not detection_result or not detection_result.detections:
        return frame, 0

    for detection in detection_result.detections:
        num_objects += 1
        bbox = detection.bounding_box
        
        # Coordinates in pixels
        x = max(0, int(bbox.origin_x))
        y = max(0, int(bbox.origin_y))
        box_w = min(int(bbox.width), w - x)
        box_h = min(int(bbox.height), h - y)

        # Draw main green bounding box
        cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), box_color, 2)
        
        # Corner accents for a modern high-tech look
        corner_len = min(15, box_w // 4, box_h // 4)
        if corner_len > 0:
            thickness = 3
            # Top-left
            cv2.line(frame, (x, y), (x + corner_len, y), box_color, thickness)
            cv2.line(frame, (x, y), (x, y + corner_len), box_color, thickness)
            # Top-right
            cv2.line(frame, (x + box_w, y), (x + box_w - corner_len, y), box_color, thickness)
            cv2.line(frame, (x + box_w, y), (x + box_w, y + corner_len), box_color, thickness)
            # Bottom-left
            cv2.line(frame, (x, y + box_h), (x + corner_len, y + box_h), box_color, thickness)
            cv2.line(frame, (x, y + box_h), (x, y + box_h - corner_len), box_color, thickness)
            # Bottom-right
            cv2.line(frame, (x + box_w, y + box_h), (x + box_w - corner_len, y + box_h), box_color, thickness)
            cv2.line(frame, (x + box_w, y + box_h), (x + box_w, y + box_h - corner_len), box_color, thickness)

        # Get top category details
        if detection.categories:
            category = detection.categories[0]
            label = category.category_name or "Object"
            score = category.score
            text = f"{label} {int(score * 100)}%"

            # Calculate text size for badge background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            text_thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
            
            # Position label badge above box if space permits, otherwise inside
            badge_y1 = max(0, y - text_h - 10) if y - text_h - 10 > 0 else y
            badge_y2 = badge_y1 + text_h + 8
            badge_x2 = min(w, x + text_w + 10)

            # Draw green label badge background
            cv2.rectangle(frame, (x, badge_y1), (badge_x2, badge_y2), box_color, -1)
            # Draw label text in black for high contrast over green
            cv2.putText(
                frame,
                text,
                (x + 5, badge_y1 + text_h + 3),
                font,
                font_scale,
                (0, 0, 0),
                text_thickness,
                cv2.LINE_AA,
            )

    return frame, num_objects


def draw_hud(frame, fps: float, count: int, threshold: float):
    """Draws a semi-transparent HUD showing FPS, detection count, and active threshold."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (280, 85), (20, 20, 20), -1)
    # Blend semi-transparent HUD background
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    # Status texts
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 32), font, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Detected Objects: {count}", (20, 54), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Threshold: {int(threshold * 100)}% (u/d to tune)", (20, 74), font, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


def run_system(camera_id=0, model_path=DEFAULT_MODEL_PATH, initial_threshold=0.5, max_results=10):
    """Main loop for camera capture, detection, and display."""
    ensure_model_exists(model_path)

    current_threshold = initial_threshold
    detector = create_detector(model_path, current_threshold, max_results)

    print(f"[*] Initializing camera (ID: {camera_id})...")
    # Using DirectShow on Windows if available for faster startup
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"[!] Fallback: retrying camera default backend...")
        cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"[!] Error: Could not open camera {camera_id}.")
        print("[!] Please verify your webcam is connected, enabled in privacy settings, and not used by another app.")
        sys.exit(1)

    # Set preferred resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "MediaPipe Object Detection (Press 'q' to Exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n==============================================")
    print("  MediaPipe Real-Time Object Detection System  ")
    print("==============================================")
    print("  Controls:")
    print("    'q' or ESC : Quit application")
    print("    's'        : Save screenshot with detections")
    print("    'u'        : Increase confidence threshold")
    print("    'd'        : Decrease confidence threshold")
    print("==============================================\n")

    fps = 0.0
    frame_counter = 0
    start_time = time.time()

    save_dir = os.environ.get("HYDRA_CAPTURES_DIR", r"E:\HYDRA REC")
    os.makedirs(save_dir, exist_ok=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[!] Failed to grab frame from camera.")
                break

            # Calculate FPS
            frame_counter += 1
            elapsed = time.time() - start_time
            if elapsed >= 0.5:
                fps = frame_counter / elapsed
                frame_counter = 0
                start_time = time.time()

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Perform detection
            detection_result = detector.detect(mp_image)

            # Draw green bounding boxes and labels
            frame, obj_count = draw_detections(frame, detection_result, box_color=(0, 255, 0))

            # Draw HUD
            draw_hud(frame, fps, obj_count, current_threshold)

            # Display output
            cv2.imshow(window_name, frame)

            # Handle key events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("[*] Exiting system...")
                break
            elif key == ord('s'):
                filename = os.path.join(save_dir, f"detection_{int(time.time())}.jpg")
                cv2.imwrite(filename, frame)
                print(f"[+] Screenshot saved to: {filename}")
            elif key in (ord('u'), ord('U')):
                current_threshold = min(0.95, current_threshold + 0.05)
                detector = create_detector(model_path, current_threshold, max_results)
                print(f"[*] Confidence threshold increased to: {int(current_threshold * 100)}%")
            elif key in (ord('d'), ord('D')):
                current_threshold = max(0.10, current_threshold - 0.05)
                detector = create_detector(model_path, current_threshold, max_results)
                print(f"[*] Confidence threshold decreased to: {int(current_threshold * 100)}%")

    except KeyboardInterrupt:
        print("\n[*] Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[+] Camera released and windows closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Object Detection with OpenCV and MediaPipe")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to .tflite model file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold (0.0 - 1.0, default: 0.5)")
    parser.add_argument("--max_results", type=int, default=10, help="Max detected objects per frame (default: 10)")
    args = parser.parse_args()

    run_system(
        camera_id=args.camera,
        model_path=args.model,
        initial_threshold=args.threshold,
        max_results=args.max_results,
    )
