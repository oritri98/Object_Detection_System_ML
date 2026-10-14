"""
=============================================================================
                     PROJECT HYDRA CAM - OBJECT DETECTION
=============================================================================
A clean, lightweight, and attractive Desktop App with Ocean Dark Blue & White Theme.
Supports Phone / IP Cameras (IP Webcam, DroidCam) and Laptop Webcams.
Detects and numbers objects: "Object 1: Person", "Object 2: Cup", etc.
=============================================================================
"""

import os
import sys
import time
import threading
import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# MediaPipe Model Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = "efficientdet_lite0.tflite"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite"
CAPTURES_DIR = os.environ.get("HYDRA_CAPTURES_DIR", r"E:\HYDRA REC")
os.makedirs(CAPTURES_DIR, exist_ok=True)


def check_and_download_model():
    """Ensure the object detection model is downloaded."""
    if not os.path.exists(MODEL_PATH):
        print("[*] Downloading object detection model file...")
        import urllib.request
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("[+] Model ready.")
        except Exception as e:
            print(f"[!] Error downloading model: {e}")


# ---------------------------------------------------------------------------
# Project HYDRA Cam - Main App GUI
# ---------------------------------------------------------------------------
class HydraCamApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Project HYDRA Cam")
        self.root.geometry("1100x750")
        self.root.minsize(850, 600)
        self.root.configure(bg="#07111E")  # Deep Ocean Abyss Navy

        # Core State
        self.cap = None
        self.is_running = False
        self.latest_frame = None
        self.threshold = 0.50
        self.detected_count = 0
        self.fps = 0.0
        self.lock = threading.Lock()

        # Initialize MediaPipe Detector
        check_and_download_model()
        self.detector = self._create_detector(self.threshold)

        # Build Clean Ocean UI
        self._build_header()
        self._build_video_screen()
        self._build_simple_controls()

        # Start Video Rendering Loop
        self.root.after(40, self._render_loop)

    def _create_detector(self, score_thresh: float):
        """Initializes the MediaPipe Object Detector."""
        try:
            base_opts = python.BaseOptions(model_asset_path=MODEL_PATH)
            opts = vision.ObjectDetectorOptions(
                base_options=base_opts,
                score_threshold=score_thresh,
                max_results=15,
                running_mode=vision.RunningMode.IMAGE,
            )
            return vision.ObjectDetector.create_from_options(opts)
        except Exception as e:
            print(f"[!] Detector error: {e}")
            return None

    # -----------------------------------------------------------------------
    # UI Components
    # -----------------------------------------------------------------------
    def _build_header(self):
        """Top Header with Ocean Theme & Live Status."""
        header = tk.Frame(self.root, bg="#0B1A30", height=60, highlightbackground="#00D4FF", highlightthickness=1)
        header.pack(fill=tk.X, padx=14, pady=(12, 6))

        # Title
        title_box = tk.Frame(header, bg="#0B1A30")
        title_box.pack(side=tk.LEFT, padx=18, pady=10)

        lbl_title = tk.Label(
            title_box,
            text="PROJECT HYDRA CAM",
            font=("Segoe UI", 14, "bold"),
            fg="#FFFFFF",
            bg="#0B1A30"
        )
        lbl_title.pack(side=tk.LEFT)

        lbl_badge = tk.Label(
            title_box,
            text=" OCEAN VISION ",
            font=("Segoe UI", 8, "bold"),
            fg="#00D4FF",
            bg="#07111E",
            padx=6,
            pady=2
        )
        lbl_badge.pack(side=tk.LEFT, padx=10)

        # Right Telemetry
        tele_box = tk.Frame(header, bg="#0B1A30")
        tele_box.pack(side=tk.RIGHT, padx=18)

        self.lbl_status = tk.Label(
            tele_box,
            text="● Offline",
            font=("Segoe UI", 10, "bold"),
            fg="#FF5E7E",
            bg="#0B1A30"
        )
        self.lbl_status.pack(side=tk.LEFT, padx=12)

        self.lbl_fps = tk.Label(
            tele_box,
            text="0.0 FPS",
            font=("Consolas", 10),
            fg="#8BA3C0",
            bg="#0B1A30"
        )
        self.lbl_fps.pack(side=tk.LEFT, padx=8)

        self.lbl_count = tk.Label(
            tele_box,
            text="Objects: 0",
            font=("Segoe UI", 10, "bold"),
            fg="#00D4FF",
            bg="#07111E",
            padx=10,
            pady=3
        )
        self.lbl_count.pack(side=tk.LEFT, padx=8)

    def _build_video_screen(self):
        """Central Video Screen Container."""
        self.video_frame = tk.Frame(self.root, bg="#030810", highlightbackground="#162E50", highlightthickness=1)
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=6)

        self.lbl_video = tk.Label(
            self.video_frame,
            text="Select your camera below and click 'Start Camera'\n\n🌊 Tip: For Phone Camera, open 'IP Webcam' on your phone\nand enter the IP address shown on your phone screen.",
            font=("Segoe UI", 11),
            fg="#6C8CAE",
            bg="#030810",
            justify=tk.CENTER
        )
        self.lbl_video.pack(fill=tk.BOTH, expand=True)

    def _build_simple_controls(self):
        """Simple and Clean Bottom Control Panel."""
        panel = tk.Frame(self.root, bg="#0B1A30", highlightbackground="#162E50", highlightthickness=1)
        panel.pack(fill=tk.X, padx=14, pady=(6, 14))

        # Row 1: Camera Type Selector & IP input
        row1 = tk.Frame(panel, bg="#0B1A30")
        row1.pack(fill=tk.X, padx=16, pady=(12, 6))

        tk.Label(row1, text="Camera:", font=("Segoe UI", 10, "bold"), fg="#FFFFFF", bg="#0B1A30").pack(side=tk.LEFT, padx=(0, 8))

        self.cam_source_var = tk.StringVar(value="phone")

        rb_phone = tk.Radiobutton(
            row1, text="📱 Phone Camera (Wi-Fi)", variable=self.cam_source_var, value="phone",
            font=("Segoe UI", 9), fg="#FFFFFF", bg="#0B1A30", selectcolor="#07111E",
            activebackground="#0B1A30", activeforeground="#00D4FF",
            command=self._on_source_mode_changed
        )
        rb_phone.pack(side=tk.LEFT, padx=6)

        rb_webcam = tk.Radiobutton(
            row1, text="💻 Laptop Webcam", variable=self.cam_source_var, value="webcam",
            font=("Segoe UI", 9), fg="#FFFFFF", bg="#0B1A30", selectcolor="#07111E",
            activebackground="#0B1A30", activeforeground="#00D4FF",
            command=self._on_source_mode_changed
        )
        rb_webcam.pack(side=tk.LEFT, padx=6)

        # Phone IP Input Box
        self.lbl_url_title = tk.Label(row1, text="Phone IP / URL:", font=("Segoe UI", 9), fg="#8BA3C0", bg="#0B1A30")
        self.lbl_url_title.pack(side=tk.LEFT, padx=(20, 6))

        self.entry_url = tk.Entry(
            row1,
            font=("Consolas", 10),
            bg="#061122",
            fg="#00D4FF",
            insertbackground="#00D4FF",
            highlightbackground="#00D4FF",
            highlightthickness=1,
            relief="flat",
            width=32
        )
        self.entry_url.insert(0, "http://192.168.1.100:8080/video")
        self.entry_url.pack(side=tk.LEFT, padx=4)

        # Row 2: Action Buttons & Confidence Slider
        row2 = tk.Frame(panel, bg="#0B1A30")
        row2.pack(fill=tk.X, padx=16, pady=(6, 12))

        # Start / Stop Button
        self.btn_toggle_cam = tk.Button(
            row2,
            text="▶ Start Camera",
            font=("Segoe UI", 10, "bold"),
            bg="#00D4FF",
            fg="#040B14",
            activebackground="#4FACFE",
            activeforeground="#FFFFFF",
            relief="flat",
            padx=18,
            pady=6,
            cursor="hand2",
            command=self._toggle_camera
        )
        self.btn_toggle_cam.pack(side=tk.LEFT, padx=(0, 10))

        # Snapshot Button
        self.btn_snapshot = tk.Button(
            row2,
            text="📸 Save Photo",
            font=("Segoe UI", 10, "bold"),
            bg="#162E50",
            fg="#FFFFFF",
            activebackground="#204374",
            activeforeground="#FFFFFF",
            relief="flat",
            padx=14,
            pady=6,
            cursor="hand2",
            command=self._take_snapshot
        )
        self.btn_snapshot.pack(side=tk.LEFT, padx=6)

        # Confidence Slider
        slider_box = tk.Frame(row2, bg="#0B1A30")
        slider_box.pack(side=tk.RIGHT)

        tk.Label(slider_box, text="Confidence Gate:", font=("Segoe UI", 9), fg="#8BA3C0", bg="#0B1A30").pack(side=tk.LEFT, padx=(0, 6))

        self.scale_thresh = tk.Scale(
            slider_box,
            from_=10,
            to=90,
            orient=tk.HORIZONTAL,
            bg="#0B1A30",
            fg="#FFFFFF",
            troughcolor="#061122",
            highlightthickness=0,
            length=140,
            command=self._on_threshold_change
        )
        self.scale_thresh.set(50)
        self.scale_thresh.pack(side=tk.LEFT, padx=4)

        self.lbl_thresh_val = tk.Label(slider_box, text="50%", font=("Consolas", 9, "bold"), fg="#00D4FF", bg="#0B1A30")
        self.lbl_thresh_val.pack(side=tk.LEFT, padx=(4, 0))

    # -----------------------------------------------------------------------
    # Camera Logic & Object Detection
    # -----------------------------------------------------------------------
    def _on_source_mode_changed(self):
        mode = self.cam_source_var.get()
        if mode == "phone":
            self.lbl_url_title.pack(side=tk.LEFT, padx=(20, 6))
            self.entry_url.pack(side=tk.LEFT, padx=4)
        else:
            self.lbl_url_title.pack_forget()
            self.entry_url.pack_forget()

    def _on_threshold_change(self, val):
        self.threshold = float(val) / 100.0
        self.lbl_thresh_val.config(text=f"{int(val)}%")
        self.detector = self._create_detector(self.threshold)

    def _toggle_camera(self):
        if self.is_running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        mode = self.cam_source_var.get()
        if mode == "phone":
            source = self.entry_url.get().strip()
            if not source.startswith("http"):
                messagebox.showwarning(
                    "Invalid URL",
                    "Please enter a valid Phone Camera stream URL\nExample: http://192.168.1.5:8080/video"
                )
                return
        else:
            source = 0  # Default Webcam Index

        self.lbl_status.config(text="● Connecting...", fg="#FFAA00")
        self.btn_toggle_cam.config(text="Connecting...", state=tk.DISABLED)

        # Start capture thread
        threading.Thread(target=self._camera_worker, args=(source,), daemon=True).start()

    def _camera_worker(self, source):
        """Background worker reading frames from camera."""
        try:
            if sys.platform.startswith("win") and isinstance(source, int):
                cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(source)

            if not cap.isOpened():
                cap = cv2.VideoCapture(source)

            if not cap.isOpened():
                self.root.after(0, self._on_camera_failed, f"Could not connect to camera source:\n{source}")
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self.cap = cap
            self.is_running = True

            self.root.after(0, self._on_camera_connected)

            t_start = time.time()
            f_count = 0

            while self.is_running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.02)
                    continue

                # Run Object Detection with MediaPipe
                annotated_frame, count = self._detect_objects(frame)

                # Update FPS
                f_count += 1
                dt = time.time() - t_start
                if dt >= 0.5:
                    self.fps = f_count / dt
                    f_count = 0
                    t_start = time.time()

                with self.lock:
                    self.latest_frame = annotated_frame
                    self.detected_count = count

            cap.release()

        except Exception as e:
            self.root.after(0, self._on_camera_failed, str(e))

    def _on_camera_connected(self):
        self.lbl_status.config(text="● Connected & Tracking", fg="#00FF80")
        self.btn_toggle_cam.config(text="⏹ Stop Camera", bg="#FF4757", fg="#FFFFFF", state=tk.NORMAL)

    def _on_camera_failed(self, error_msg):
        self.lbl_status.config(text="● Disconnected", fg="#FF5E7E")
        self.btn_toggle_cam.config(text="▶ Start Camera", bg="#00D4FF", fg="#040B14", state=tk.NORMAL)
        messagebox.showerror("Camera Error", f"Connection failed:\n{error_msg}\n\nMake sure phone and laptop are on the same Wi-Fi.")

    def _stop_camera(self):
        self.is_running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        self.lbl_status.config(text="● Offline", fg="#FF5E7E")
        self.lbl_fps.config(text="0.0 FPS")
        self.lbl_count.config(text="Objects: 0")
        self.btn_toggle_cam.config(text="▶ Start Camera", bg="#00D4FF", fg="#040B14", state=tk.NORMAL)

    def _detect_objects(self, frame):
        """
        Runs detection and draws clear 'Object 1: Person (94%)' bounding boxes.
        Ocean color scheme: Cyan BGR (254, 212, 0).
        """
        if self.detector is None:
            return frame, 0

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            results = self.detector.detect(mp_img)
        except Exception:
            return frame, 0

        annotated = frame.copy()
        if not results or not results.detections:
            return annotated, 0

        box_color = (254, 212, 0)      # Ocean Cyan in BGR
        badge_color = (254, 212, 0)    # Cyan Badge Fill
        text_color = (8, 16, 30)       # Dark Navy Text for max readability

        count = len(results.detections)

        for idx, det in enumerate(results.detections, start=1):
            bbox = det.bounding_box
            x = max(0, int(bbox.origin_x))
            y = max(0, int(bbox.origin_y))
            bw = min(int(bbox.width), w - x)
            bh = min(int(bbox.height), h - y)

            if bw <= 0 or bh <= 0:
                continue

            category_name = "Object"
            score = 0.0
            if det.categories:
                category_name = (det.categories[0].category_name or "Object").capitalize()
                score = det.categories[0].score

            # Label strictly as "Object 1", "Object 2", "Object 3", etc.
            label = f"Object {idx}"

            # 1. Main Bounding Box
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), box_color, 2)

            # 2. Corner High-Tech Brackets
            corner_len = min(16, bw // 4, bh // 4)
            if corner_len > 3:
                cv2.line(annotated, (x, y), (x + corner_len, y), (255, 255, 255), 3)
                cv2.line(annotated, (x, y), (x, y + corner_len), (255, 255, 255), 3)
                cv2.line(annotated, (x + bw, y), (x + bw - corner_len, y), (255, 255, 255), 3)
                cv2.line(annotated, (x + bw, y), (x + bw, y + corner_len), (255, 255, 255), 3)
                cv2.line(annotated, (x, y + bh), (x + corner_len, y + bh), (255, 255, 255), 3)
                cv2.line(annotated, (x, y + bh), (x, y + bh - corner_len), (255, 255, 255), 3)
                cv2.line(annotated, (x + bw, y + bh), (x + bw - corner_len, y + bh), (255, 255, 255), 3)
                cv2.line(annotated, (x + bw, y + bh), (x + bw, y + bh - corner_len), (255, 255, 255), 3)

            # 3. Label Badge
            font = cv2.FONT_HERSHEY_DUPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.52, 1)
            by1 = max(0, y - th - 10) if y - th - 10 > 0 else y
            by2 = by1 + th + 10
            bx2 = min(w, x + tw + 14)

            cv2.rectangle(annotated, (x, by1), (bx2, by2), badge_color, -1)
            cv2.rectangle(annotated, (x, by1), (bx2, by2), (255, 255, 255), 1)
            cv2.putText(annotated, label, (x + 6, by1 + th + 4), font, 0.52, text_color, 1, cv2.LINE_AA)

        return annotated, count

    # -----------------------------------------------------------------------
    # Rendering Loop & Snapshot
    # -----------------------------------------------------------------------
    def _render_loop(self):
        """Refreshes video frame on Tkinter canvas smoothly."""
        with self.lock:
            frame = self.latest_frame
            count = self.detected_count
            fps = self.fps

        if frame is not None and self.is_running:
            # Scale frame to fit window
            vw = max(320, self.video_frame.winfo_width() - 8)
            vh = max(240, self.video_frame.winfo_height() - 8)

            h, w = frame.shape[:2]
            scale = min(vw / w, vh / h)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))

            resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            self.lbl_video.imgtk = imgtk
            self.lbl_video.config(image=imgtk, text="")

            self.lbl_fps.config(text=f"{fps:.1f} FPS")
            self.lbl_count.config(text=f"Objects: {count}")

        self.root.after(33, self._render_loop)

    def _take_snapshot(self):
        with self.lock:
            frame = self.latest_frame
            count = self.detected_count

        if frame is None or not self.is_running:
            messagebox.showinfo("Project HYDRA Cam", "Please start camera before taking a snapshot.")
            return

        filename = f"hydra_capture_{int(time.time())}.jpg"
        filepath = os.path.join(CAPTURES_DIR, filename)
        cv2.imwrite(filepath, frame)
        messagebox.showinfo(
            "Snapshot Saved!",
            f"📸 HD Snapshot saved successfully with {count} detected object(s)!\n\nSaved to:\n{filepath}"
        )

    def on_closing(self):
        self._stop_camera()
        self.root.destroy()


# ---------------------------------------------------------------------------
# App Launch Entry Point
# ---------------------------------------------------------------------------
def main():
    root = tk.Tk()
    app = HydraCamApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
