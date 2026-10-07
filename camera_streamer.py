"""
Project HYDRA Cam - Camera Streamer & Pipeline Manager
Threaded multi-source video capture for Phone / IP cameras, USB webcams, and Demo streams.
Zero-lag buffered capture with real-time MediaPipe inference and snapshot/recording capabilities.
"""

import os
import sys
import time
import threading
import datetime
import cv2
import numpy as np
from typing import Optional, Dict, Any, List
from detector_engine import HydraDetector

CAPTURES_DIR = os.environ.get("HYDRA_CAPTURES_DIR", r"E:\HYDRA REC")
os.makedirs(CAPTURES_DIR, exist_ok=True)


class CameraStreamer:
    def __init__(self, detector: HydraDetector):
        self.detector = detector
        self.cap: Optional[cv2.VideoCapture] = None
        self.source_type: str = "demo"  # "webcam" | "ip_cam" | "demo"
        self.source_value: Any = 0  # int for webcam, str for url, "demo"
        self.is_running: bool = False
        self.is_connected: bool = False
        self.connection_error: str = ""

        self.latest_raw_frame: Optional[np.ndarray] = None
        self.latest_annotated_frame: Optional[np.ndarray] = None
        self.latest_detections: List[Dict[str, Any]] = []

        self.lock = threading.Lock()
        self.worker_thread: Optional[threading.Thread] = None

        # Telemetry metrics
        self.fps: float = 0.0
        self.frame_count: int = 0
        self.stream_width: int = 1280
        self.stream_height: int = 720
        self.last_frame_time: float = time.time()

        # Recording state
        self.is_recording: bool = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.current_recording_filename: str = ""
        self.recording_start_time: float = 0.0

        # Start demo stream initially
        self.connect(source_type="demo", source_value="demo")

    def connect(self, source_type: str, source_value: Any):
        """Switches or reconnects the active video source."""
        with self.lock:
            self.is_running = False

        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.5)

        with self.lock:
            self.source_type = source_type
            self.source_value = source_value
            self.is_connected = False
            self.connection_error = ""
            self.is_running = True

            # Clean up old capture device
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

        self.worker_thread = threading.Thread(target=self._capture_and_process_loop, daemon=True)
        self.worker_thread.start()

    def disconnect(self):
        """Stops active camera and switches to offline idle."""
        self.is_running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        with self.lock:
            if self.is_recording:
                self.stop_recording()
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
            self.is_connected = False
            self.source_type = "offline"

    def _open_capture(self) -> bool:
        """Attempts to open the specified camera source."""
        if self.source_type == "demo":
            self.is_connected = True
            self.stream_width = 1280
            self.stream_height = 720
            return True

        if self.source_type == "webcam":
            try:
                cam_idx = int(self.source_value)
            except ValueError:
                cam_idx = 0

            # Prefer DirectShow backend on Windows for rapid initialization
            if sys.platform.startswith("win"):
                self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(cam_idx)

            if not self.cap.isOpened():
                # Fallback standard backend
                self.cap = cv2.VideoCapture(cam_idx)

        elif self.source_type == "ip_cam":
            url = str(self.source_value).strip()
            # Set OpenCV FFmpeg / MJPEG options for minimal network buffering
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|buffer_size;1024000"
            self.cap = cv2.VideoCapture(url)

        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Try setting resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.is_connected = True
            self.connection_error = ""
            return True
        else:
            self.is_connected = False
            self.connection_error = f"Unable to open {self.source_type} stream: {self.source_value}"
            return False

    def _generate_demo_frame(self, t: float) -> np.ndarray:
        """Creates a high-tech ocean simulation frame with moving targets."""
        w, h = self.stream_width, self.stream_height
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Deep ocean gradient background
        for y in range(h):
            ratio = y / h
            b = int(25 + 35 * ratio + 5 * np.sin(t * 0.5 + ratio * 3))
            g = int(12 + 18 * ratio)
            r = int(5 + 10 * ratio)
            frame[y, :] = (b, g, r)

        # Draw grid lines
        grid_spacing = 60
        grid_color = (45, 30, 15)
        for gx in range(0, w, grid_spacing):
            cv2.line(frame, (gx, 0), (gx, h), grid_color, 1)
        for gy in range(0, h, grid_spacing):
            cv2.line(frame, (0, gy), (w, gy), grid_color, 1)

        # Ocean sonar radar sweep circle
        cx, cy = w // 2, h // 2
        sweep_angle = (t * 60) % 360
        radar_rad = 220
        cv2.circle(frame, (cx, cy), radar_rad, (80, 50, 20), 1)
        cv2.circle(frame, (cx, cy), radar_rad // 2, (60, 40, 15), 1)
        rad = np.radians(sweep_angle)
        sx = int(cx + radar_rad * np.cos(rad))
        sy = int(cy + radar_rad * np.sin(rad))
        cv2.line(frame, (cx, cy), (sx, sy), (200, 150, 0), 2)

        # Simulated moving items for detection demonstration
        # Target 1 (Simulated Person / Target A)
        p1_x = int(cx - 240 + 80 * np.cos(t * 0.8))
        p1_y = int(cy - 120 + 40 * np.sin(t * 0.6))
        cv2.rectangle(frame, (p1_x, p1_y), (p1_x + 130, p1_y + 240), (140, 80, 20), -1)
        cv2.circle(frame, (p1_x + 65, p1_y + 40), 30, (180, 100, 30), -1)
        cv2.putText(frame, "SIMULATED PERSON", (p1_x - 10, p1_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Target 2 (Simulated Object / Target B)
        p2_x = int(cx + 160 + 50 * np.sin(t * 1.1))
        p2_y = int(cy + 20 + 30 * np.cos(t * 0.9))
        cv2.rectangle(frame, (p2_x, p2_y), (p2_x + 90, p2_y + 90), (110, 90, 40), -1)
        cv2.putText(frame, "SIMULATED ITEM", (p2_x - 5, p2_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Watermark & Guidance
        cv2.putText(frame, "PROJECT HYDRA CAM - DEMO TEST PATTERN", (30, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (254, 242, 0), 2)
        cv2.putText(frame, "Select 'Phone / IP Camera' or 'Local Webcam' from the side panel to connect real feed", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 220, 255), 1)

        return frame

    def _annotate_demo_frame(self, frame: np.ndarray):
        """Draws dynamic cyber HUD annotations on demo targets."""
        annotated = frame.copy()
        w, h = self.stream_width, self.stream_height
        cx, cy = w // 2, h // 2
        t = time.time()

        p1_x = int(cx - 240 + 80 * np.cos(t * 0.8))
        p1_y = int(cy - 120 + 40 * np.sin(t * 0.6))
        p2_x = int(cx + 160 + 50 * np.sin(t * 1.1))
        p2_y = int(cy + 20 + 30 * np.cos(t * 0.9))

        raw_targets = [
            {"id": 1, "label": "Person", "score": 94.2, "box": [p1_x, p1_y, 130, 240]},
            {"id": 2, "label": "Device", "score": 88.6, "box": [p2_x, p2_y, 90, 90]}
        ]

        detections = []
        for t_info in raw_targets:
            score_val = t_info["score"] / 100.0
            if score_val < self.detector.threshold:
                continue

            idx = t_info["id"]
            cat = t_info["label"]
            x, y, bw, bh = t_info["box"]

            if self.detector.label_mode == "numbered_simple":
                disp = f"Object {idx}"
            elif self.detector.label_mode == "numbered_category":
                disp = f"Object {idx}: {cat}"
            elif self.detector.label_mode == "standard":
                disp = f"{cat} {int(t_info['score'])}%"
            else:
                disp = f"Object {idx}: {cat} ({int(t_info['score'])}%)"

            detections.append({
                "id": idx,
                "raw_label": cat,
                "display_label": disp,
                "score": t_info["score"],
                "box": [x, y, bw, bh]
            })

            if self.detector.show_boxes:
                # Main box
                cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (254, 242, 0), 2)
                # Reticles
                if self.detector.show_reticles:
                    cv2.line(annotated, (x, y), (x + 15, y), (255, 200, 0), 3)
                    cv2.line(annotated, (x, y), (x, y + 15), (255, 200, 0), 3)
                    cv2.line(annotated, (x + bw, y), (x + bw - 15, y), (255, 200, 0), 3)
                    cv2.line(annotated, (x + bw, y), (x + bw, y + 15), (255, 200, 0), 3)
                    cv2.line(annotated, (x, y + bh), (x + 15, y + bh), (255, 200, 0), 3)
                    cv2.line(annotated, (x, y + bh), (x, y + bh - 15), (255, 200, 0), 3)
                    cv2.line(annotated, (x + bw, y + bh), (x + bw - 15, y + bh), (255, 200, 0), 3)
                    cv2.line(annotated, (x + bw, y + bh), (x + bw, y + bh - 15), (255, 200, 0), 3)

            if self.detector.show_labels:
                font = cv2.FONT_HERSHEY_DUPLEX
                (tw, th), _ = cv2.getTextSize(disp, font, 0.52, 1)
                by1 = max(0, y - th - 10) if y - th - 10 > 0 else y
                by2 = by1 + th + 10
                bx2 = min(w, x + tw + 14)
                cv2.rectangle(annotated, (x, by1), (bx2, by2), (254, 242, 0), -1)
                cv2.rectangle(annotated, (x, by1), (bx2, by2), (255, 255, 255), 1)
                cv2.putText(annotated, disp, (x + 7, by1 + th + 4), font, 0.52, (10, 15, 30), 1, cv2.LINE_AA)

        return annotated, detections


    def _generate_connecting_frame(self) -> np.ndarray:
        """Generates an ocean connecting splash card."""
        w, h = 1280, 720
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (30, 15, 6)

        # Cyber Grid
        for gx in range(0, w, 40):
            cv2.line(frame, (gx, 0), (gx, h), (45, 25, 12), 1)
        for gy in range(0, h, 40):
            cv2.line(frame, (0, gy), (w, gy), (45, 25, 12), 1)

        cx, cy = w // 2, h // 2
        # Pulsing circle
        t = time.time()
        pulse_r = int(70 + 20 * np.sin(t * 3))
        cv2.circle(frame, (cx, cy), pulse_r, (254, 242, 0), 2)
        cv2.circle(frame, (cx, cy), 8, (254, 242, 0), -1)

        title = "CONNECTING TO HYDRA VIDEO STREAM..."
        font = cv2.FONT_HERSHEY_DUPLEX
        (tw, th), _ = cv2.getTextSize(title, font, 0.85, 2)
        cv2.putText(frame, title, (cx - tw // 2, cy - 100), font, 0.85, (254, 242, 0), 2, cv2.LINE_AA)

        src_txt = f"Source: [{self.source_type.upper()}] {self.source_value}"
        (sw, sh), _ = cv2.getTextSize(src_txt, font, 0.55, 1)
        cv2.putText(frame, src_txt, (cx - sw // 2, cy + 120), font, 0.55, (220, 240, 255), 1, cv2.LINE_AA)

        if self.connection_error:
            err_txt = f"Status: {self.connection_error}"
            (ew, eh), _ = cv2.getTextSize(err_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(frame, err_txt, (cx - ew // 2, cy + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1, cv2.LINE_AA)

        return frame

    def _capture_and_process_loop(self):
        """Continuous background thread capturing frames and running MediaPipe inference."""
        t_start = time.time()
        frame_counter = 0
        reconnect_delay = 2.0
        last_connect_try = 0.0

        while self.is_running:
            raw_frame = None

            if self.source_type == "demo":
                t = time.time()
                raw_frame = self._generate_demo_frame(t)
                self.is_connected = True
                self.connection_error = ""
                time.sleep(0.033)  # ~30 FPS
            else:
                if self.cap is None or not self.cap.isOpened():
                    if time.time() - last_connect_try > reconnect_delay:
                        last_connect_try = time.time()
                        self._open_capture()

                    if not self.is_connected:
                        raw_frame = self._generate_connecting_frame()
                        time.sleep(0.05)
                else:
                    ret, frame = self.cap.read()
                    if ret and frame is not None and frame.size > 0:
                        raw_frame = frame
                        self.stream_height, self.stream_width = raw_frame.shape[:2]
                        self.is_connected = True
                    else:
                        # Stream interrupted, trigger reconnect on next loop
                        self.is_connected = False
                        self.connection_error = "Frame read failed or connection dropped."
                        try:
                            self.cap.release()
                        except Exception:
                            pass
                        self.cap = None
                        raw_frame = self._generate_connecting_frame()
                        time.sleep(0.1)

            if raw_frame is not None:
                # 1. Run detection engine
                if self.source_type == "demo":
                    annotated_frame, detections = self._annotate_demo_frame(raw_frame)
                else:
                    annotated_frame, detections = self.detector.detect_and_annotate(raw_frame)

                # 2. Update FPS counter
                frame_counter += 1
                now = time.time()
                dt = now - t_start
                if dt >= 0.5:
                    self.fps = round(frame_counter / dt, 1)
                    frame_counter = 0
                    t_start = now

                # 3. Handle video recording if active
                if self.is_recording and self.video_writer is not None:
                    try:
                        self.video_writer.write(annotated_frame)
                    except Exception as e:
                        print(f"[!] Recording write error: {e}")

                # 4. Save thread-safe outputs
                with self.lock:
                    self.latest_raw_frame = raw_frame
                    self.latest_annotated_frame = annotated_frame
                    self.latest_detections = detections
                    self.last_frame_time = now

        # Cleanup on loop exit
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def get_latest_jpeg(self, annotated: bool = True, quality: int = 85) -> Optional[bytes]:
        """Encodes the latest frame as JPEG bytes for MJPEG streaming."""
        with self.lock:
            frame = self.latest_annotated_frame if annotated else self.latest_raw_frame

        if frame is None:
            return None

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, encoded = cv2.imencode(".jpg", frame, encode_param)
        if success:
            return encoded.tobytes()
        return None

    def capture_snapshot(self) -> Dict[str, Any]:
        """Saves current frame as an HD snapshot in captures directory."""
        with self.lock:
            frame = self.latest_annotated_frame

        if frame is None:
            return {"success": False, "error": "No active frame available"}

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hydra_capture_{timestamp_str}.jpg"
        filepath = os.path.join(CAPTURES_DIR, filename)

        success = cv2.imwrite(filepath, frame)
        if success:
            return {
                "success": True,
                "filename": filename,
                "filepath": filepath,
                "timestamp": timestamp_str,
                "object_count": len(self.latest_detections)
            }
        else:
            return {"success": False, "error": "Failed to write image file"}

    def toggle_recording(self) -> Dict[str, Any]:
        """Toggles video recording on / off."""
        if self.is_recording:
            return self.stop_recording()
        else:
            return self.start_recording()

    def start_recording(self) -> Dict[str, Any]:
        """Starts saving annotated video stream to MP4."""
        with self.lock:
            if self.is_recording:
                return {"success": True, "recording": True, "filename": self.current_recording_filename}

            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hydra_record_{timestamp_str}.mp4"
            filepath = os.path.join(CAPTURES_DIR, filename)

            h, w = self.stream_height, self.stream_width
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = max(15.0, self.fps if self.fps > 0 else 24.0)

            self.video_writer = cv2.VideoWriter(filepath, fourcc, fps, (w, h))
            if self.video_writer.isOpened():
                self.is_recording = True
                self.current_recording_filename = filename
                self.recording_start_time = time.time()
                return {"success": True, "recording": True, "filename": filename}
            else:
                self.video_writer = None
                return {"success": False, "error": "Failed to initialize video writer"}

    def stop_recording(self) -> Dict[str, Any]:
        """Stops active recording and finalizes MP4 file."""
        with self.lock:
            if not self.is_recording:
                return {"success": True, "recording": False}

            self.is_recording = False
            filename = self.current_recording_filename
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None

            duration = round(time.time() - self.recording_start_time, 1)
            return {"success": True, "recording": False, "filename": filename, "duration": duration}

    def get_status(self) -> Dict[str, Any]:
        """Returns comprehensive telemetry dictionary."""
        with self.lock:
            detections = list(self.latest_detections)
            fps = self.fps
            is_connected = self.is_connected
            src_type = self.source_type
            src_val = str(self.source_value)
            w, h = self.stream_width, self.stream_height
            is_rec = self.is_recording
            rec_file = self.current_recording_filename

        return {
            "connected": is_connected,
            "source_type": src_type,
            "source_value": src_val,
            "fps": fps,
            "resolution": f"{w}x{h}",
            "object_count": len(detections),
            "detections": detections,
            "is_recording": is_rec,
            "recording_filename": rec_file,
            "detector_config": {
                "threshold": self.detector.threshold,
                "max_results": self.detector.max_results,
                "label_mode": self.detector.label_mode,
                "color_theme": self.detector.color_theme,
                "show_boxes": self.detector.show_boxes,
                "show_labels": self.detector.show_labels,
                "show_scores": self.detector.show_scores,
                "show_reticles": self.detector.show_reticles
            }
        }
