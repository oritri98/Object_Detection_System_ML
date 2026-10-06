"""
Project HYDRA Cam - Native Desktop Application Window
Standalone Tkinter-based Ocean Dark UI with real-time MediaPipe Object Detection,
supporting Phone / IP cameras, USB webcams, and Demo stream.
"""

import os
import sys
import time
import datetime
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

from detector_engine import HydraDetector
from camera_streamer import CameraStreamer, CAPTURES_DIR


class HydraDesktopApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Project HYDRA Cam - Desktop Vision Matrix")
        self.root.geometry("1240x820")
        self.root.minsize(980, 680)
        self.root.configure(bg="#050B18")

        # Initialize detector and streamer
        self.detector = HydraDetector(threshold=0.50, max_results=15)
        self.streamer = CameraStreamer(detector=self.detector)

        self._init_styles()
        self._build_ui()
        self._start_video_loop()

    def _init_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Configure dark ocean styles
        self.style.configure("TFrame", background="#050B18")
        self.style.configure("Card.TFrame", background="#0B192C", relief="flat")
        self.style.configure("TLabel", background="#050B18", foreground="#F0F8FF", font=("Segoe UI", 10))
        self.style.configure("Card.TLabel", background="#0B192C", foreground="#F0F8FF", font=("Segoe UI", 10))
        self.style.configure("Title.TLabel", background="#050B18", foreground="#00F2FE", font=("Segoe UI", 14, "bold"))
        self.style.configure("Section.TLabel", background="#0B192C", foreground="#00F2FE", font=("Segoe UI", 11, "bold"))
        self.style.configure("Telemetry.TLabel", background="#0B192C", foreground="#4FACFE", font=("Consolas", 10, "bold"))
        self.style.configure("TCombobox", fieldbackground="#061124", background="#0B192C", foreground="#FFFFFF")

    def _build_ui(self):
        # 1. Top Header
        header_frame = tk.Frame(self.root, bg="#071025", height=54, highlightbackground="#00F2FE", highlightthickness=1)
        header_frame.pack(fill=tk.X, padx=12, pady=(10, 8))

        # Brand Title
        lbl_title = tk.Label(header_frame, text="🔱 PROJECT HYDRA CAM", font=("Segoe UI", 13, "bold"), fg="#00F2FE", bg="#071025")
        lbl_title.pack(side=tk.LEFT, padx=16, pady=10)

        lbl_sub = tk.Label(header_frame, text="• Desktop Neural Vision Terminal", font=("Segoe UI", 9), fg="#8CA3BA", bg="#071025")
        lbl_sub.pack(side=tk.LEFT, padx=4)

        # Telemetry Labels
        self.lbl_fps = tk.Label(header_frame, text="FPS: 0.0", font=("Consolas", 10, "bold"), fg="#00FF80", bg="#071025")
        self.lbl_fps.pack(side=tk.RIGHT, padx=16)

        self.lbl_count = tk.Label(header_frame, text="OBJECTS: 0", font=("Consolas", 10, "bold"), fg="#00F2FE", bg="#071025")
        self.lbl_count.pack(side=tk.RIGHT, padx=14)

        # 2. Main Content Grid (Video Left + Sidebar Right)
        main_container = tk.Frame(self.root, bg="#050B18")
        main_container.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

        # Left Video Card
        video_card = tk.Frame(main_container, bg="#02050E", highlightbackground="#1E3E62", highlightthickness=1)
        video_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.video_canvas = tk.Label(video_card, bg="#01040A", text="Connecting to HYDRA Vision...", fg="#00F2FE", font=("Segoe UI", 12))
        self.video_canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Bottom Bar under Video
        video_bottom = tk.Frame(video_card, bg="#081426", height=38)
        video_bottom.pack(fill=tk.X, side=tk.BOTTOM)

        btn_snap = tk.Button(video_bottom, text="📸 Snapshot", font=("Segoe UI", 9, "bold"), bg="#0072FF", fg="#FFFFFF",
                             activebackground="#00F2FE", activeforeground="#000000", relief="flat", padx=14, pady=4,
                             command=self._on_snapshot)
        btn_snap.pack(side=tk.LEFT, padx=10, pady=6)

        self.btn_rec = tk.Button(video_bottom, text="🔴 Record", font=("Segoe UI", 9, "bold"), bg="#1E3E62", fg="#FF6B81",
                                activebackground="#FF4757", activeforeground="#FFFFFF", relief="flat", padx=14, pady=4,
                                command=self._on_record_toggle)
        self.btn_rec.pack(side=tk.LEFT, padx=6, pady=6)

        self.lbl_rec_status = tk.Label(video_bottom, text="Standby", font=("Consolas", 9), fg="#8CA3BA", bg="#081426")
        self.lbl_rec_status.pack(side=tk.RIGHT, padx=12)

        # Right Controls Sidebar
        sidebar = tk.Frame(main_container, bg="#0B192C", width=340, highlightbackground="#00F2FE", highlightthickness=1)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 0))
        sidebar.pack_propagate(False)

        # Sidebar Title
        tk.Label(sidebar, text="⚙️ CAMERA & ML MATRIX", font=("Segoe UI", 11, "bold"), fg="#00F2FE", bg="#0B192C").pack(anchor=tk.W, padx=14, pady=(14, 8))

        # Camera Source Section
        tk.Label(sidebar, text="Camera Stream Input:", font=("Segoe UI", 9, "bold"), fg="#F0F8FF", bg="#0B192C").pack(anchor=tk.W, padx=14, pady=(6, 2))
        
        self.source_var = tk.StringVar(value="ip_cam")
        r_phone = tk.Radiobutton(sidebar, text="📱 Phone / IP Camera URL", variable=self.source_var, value="ip_cam",
                                 bg="#0B192C", fg="#FFFFFF", selectcolor="#050B18", activebackground="#0B192C", activeforeground="#00F2FE",
                                 command=self._update_source_mode)
        r_phone.pack(anchor=tk.W, padx=14)

        r_webcam = tk.Radiobutton(sidebar, text="💻 Laptop Webcam (0)", variable=self.source_var, value="webcam",
                                  bg="#0B192C", fg="#FFFFFF", selectcolor="#050B18", activebackground="#0B192C", activeforeground="#00F2FE",
                                  command=self._update_source_mode)
        r_webcam.pack(anchor=tk.W, padx=14)

        r_demo = tk.Radiobutton(sidebar, text="🌊 Ocean Demo Pattern", variable=self.source_var, value="demo",
                                bg="#0B192C", fg="#FFFFFF", selectcolor="#050B18", activebackground="#0B192C", activeforeground="#00F2FE",
                                command=self._update_source_mode)
        r_demo.pack(anchor=tk.W, padx=14)

        # URL Input Box
        self.lbl_url = tk.Label(sidebar, text="Phone Stream URL (IP Webcam):", font=("Segoe UI", 8), fg="#8CA3BA", bg="#0B192C")
        self.lbl_url.pack(anchor=tk.W, padx=14, pady=(8, 2))

        self.entry_url = tk.Entry(sidebar, bg="#061124", fg="#00F2FE", insertbackground="#00F2FE",
                                  font=("Consolas", 9), relief="flat", highlightthickness=1, highlightbackground="#1E3E62")
        self.entry_url.insert(0, "http://192.168.1.100:8080/video")
        self.entry_url.pack(fill=tk.X, padx=14, pady=(0, 8))

        # Connect / Disconnect Buttons
        btn_connect = tk.Button(sidebar, text="⚡ Connect Stream", font=("Segoe UI", 9, "bold"), bg="#00F2FE", fg="#040814",
                                activebackground="#4FACFE", relief="flat", pady=4, command=self._on_connect_stream)
        btn_connect.pack(fill=tk.X, padx=14, pady=4)

        # Separator
        tk.Frame(sidebar, bg="#1E3E62", height=1).pack(fill=tk.X, padx=14, pady=12)

        # ML Detection Settings
        tk.Label(sidebar, text="🎯 Detection Tuning:", font=("Segoe UI", 9, "bold"), fg="#F0F8FF", bg="#0B192C").pack(anchor=tk.W, padx=14, pady=(0, 2))

        # Threshold Slider
        thresh_row = tk.Frame(sidebar, bg="#0B192C")
        thresh_row.pack(fill=tk.X, padx=14, pady=(4, 0))
        tk.Label(thresh_row, text="Confidence Gate:", font=("Segoe UI", 8), fg="#8CA3BA", bg="#0B192C").pack(side=tk.LEFT)
        self.lbl_thresh_val = tk.Label(thresh_row, text="50%", font=("Consolas", 8, "bold"), fg="#00F2FE", bg="#0B192C")
        self.lbl_thresh_val.pack(side=tk.RIGHT)

        self.scale_thresh = tk.Scale(sidebar, from_=10, to=95, orient=tk.HORIZONTAL, bg="#0B192C", fg="#FFFFFF",
                                     troughcolor="#061124", highlightthickness=0, command=self._on_threshold_change)
        self.scale_thresh.set(50)
        self.scale_thresh.pack(fill=tk.X, padx=14, pady=(0, 6))

        # Label Convention Dropdown
        tk.Label(sidebar, text="Object Labeling Style:", font=("Segoe UI", 8), fg="#8CA3BA", bg="#0B192C").pack(anchor=tk.W, padx=14, pady=(4, 2))
        self.combo_label_mode = ttk.Combobox(sidebar, values=[
            "Object 1: Person (94%)",
            "Object 1: Person",
            "Object 1, Object 2",
            "Person 94%"
        ], state="readonly")
        self.combo_label_mode.current(0)
        self.combo_label_mode.pack(fill=tk.X, padx=14, pady=(0, 10))
        self.combo_label_mode.bind("<<ComboboxSelected>>", self._on_label_mode_change)

        # Live Target List Box
        tk.Label(sidebar, text="Active Detected Objects:", font=("Segoe UI", 9, "bold"), fg="#00F2FE", bg="#0B192C").pack(anchor=tk.W, padx=14, pady=(6, 2))
        
        self.listbox_targets = tk.Listbox(sidebar, bg="#061124", fg="#00FF80", font=("Consolas", 9),
                                          selectbackground="#1E3E62", relief="flat", highlightthickness=1,
                                          highlightbackground="#1E3E62", height=8)
        self.listbox_targets.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 12))

    def _update_source_mode(self):
        mode = self.source_var.get()
        if mode == "ip_cam":
            self.lbl_url.config(text="Phone Stream URL (IP Webcam):")
            self.entry_url.config(state=tk.NORMAL)
        elif mode == "webcam":
            self.lbl_url.config(text="Webcam Device Index (0 or 1):")
            self.entry_url.delete(0, tk.END)
            self.entry_url.insert(0, "0")
        else:
            self.lbl_url.config(text="Demo Simulation Pattern:")
            self.entry_url.delete(0, tk.END)
            self.entry_url.insert(0, "demo")

    def _on_connect_stream(self):
        mode = self.source_var.get()
        val = self.entry_url.get().strip()
        if mode == "webcam":
            try:
                val = int(val)
            except ValueError:
                val = 0
        self.streamer.connect(source_type=mode, source_value=val)

    def _on_threshold_change(self, val):
        thresh = float(val) / 100.0
        self.lbl_thresh_val.config(text=f"{int(val)}%")
        self.detector.update_config(threshold=thresh)

    def _on_label_mode_change(self, event):
        idx = self.combo_label_mode.current()
        modes = ["numbered_hybrid", "numbered_category", "numbered_simple", "standard"]
        selected = modes[idx] if idx < len(modes) else "numbered_hybrid"
        self.detector.update_config(label_mode=selected)

    def _on_snapshot(self):
        res = self.streamer.capture_snapshot()
        if res["success"]:
            messagebox.showinfo("HYDRA Snapshot", f"HD Snapshot saved successfully!\n\nFile: {res['filename']}\nFolder: {CAPTURES_DIR}")
        else:
            messagebox.showerror("Error", f"Failed to capture snapshot: {res.get('error')}")

    def _on_record_toggle(self):
        res = self.streamer.toggle_recording()
        if res.get("recording"):
            self.btn_rec.config(text="⏹ Stop Rec", bg="#FF4757", fg="#FFFFFF")
            self.lbl_rec_status.config(text="🔴 RECORDING...", fg="#FF6B81")
        else:
            self.btn_rec.config(text="🔴 Record", bg="#1E3E62", fg="#FF6B81")
            self.lbl_rec_status.config(text=f"Saved {res.get('filename', '')}", fg="#8CA3BA")

    def _start_video_loop(self):
        def update_frame():
            with self.streamer.lock:
                frame = self.streamer.latest_annotated_frame
                fps = self.streamer.fps
                detections = list(self.streamer.latest_detections)

            if frame is not None:
                # Resize frame to fit canvas
                canvas_w = max(320, self.video_canvas.winfo_width())
                canvas_h = max(240, self.video_canvas.winfo_height())
                
                h, w = frame.shape[:2]
                scale = min(canvas_w / w, canvas_h / h)
                nw, nh = max(1, int(w * scale)), max(1, int(h * scale))

                resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                imgtk = ImageTk.PhotoImage(image=img)

                self.video_canvas.imgtk = imgtk
                self.video_canvas.config(image=imgtk, text="")

            # Update Telemetry Readouts
            self.lbl_fps.config(text=f"FPS: {fps:.1f}")
            self.lbl_count.config(text=f"OBJECTS: {len(detections)}")

            # Update Listbox Targets
            self.listbox_targets.delete(0, tk.END)
            if not detections:
                self.listbox_targets.insert(tk.END, " Scanning scene for objects...")
            else:
                for d in detections:
                    self.listbox_targets.insert(tk.END, f" #{d['id']} {d['display_label']}")

            self.root.after(33, update_frame)

        self.root.after(100, update_frame)

    def on_closing(self):
        self.streamer.disconnect()
        self.root.destroy()


def run_desktop_app():
    root = tk.Tk()
    app = HydraDesktopApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    run_desktop_app()
