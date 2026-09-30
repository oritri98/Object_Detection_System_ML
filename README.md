# Real-Time Object Detection with OpenCV & MediaPipe

A real-time, lightweight computer vision system built with **OpenCV** and **Google MediaPipe**. It captures video from your webcam, detects standard objects in real-time, and highlights them using high-visibility **green bounding boxes** and formatted labels with confidence percentages.

---

## 🚀 Features

- **Live Webcam Stream**: Automatically connects to your default camera and streams live video.
- **MediaPipe Object Detector**: Uses Google's fast and efficient TFLite models (efficientdet_lite0.tflite).
- **Green Bounding Boxes**: Clearly demarcates detected objects with green boxes and corner accents.
- **High-Contrast Labels**: Displays the detected object name and confidence score (e.g., person 89%, laptop 94%, cup 75%).
- **Interactive Controls**:
  - q / ESC : Exit the application safely.
  - s : Save an instant screenshot to the screenshots/ directory.
  - u / d : Dynamically raise (u) or lower (d) the confidence threshold on the fly.
- **HUD (Heads-Up Display)**: Displays live FPS counter and object detection tally.
- **Auto-Model Management**: Automatically downloads the required TFLite model on first run if not already present.

---