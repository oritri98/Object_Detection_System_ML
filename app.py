"""
Project HYDRA Cam - Main Application Server
FastAPI web server providing live MJPEG streaming, real-time object telemetry,
camera controls (Phone/IP Cam & USB Webcams), and modern ocean dark UI.
"""

import os
import glob
import time
import webbrowser
import threading
from typing import Optional, Dict, Any
from fastapi import FastAPI, Request, HTTPException, Body
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

from detector_engine import HydraDetector
from camera_streamer import CameraStreamer, CAPTURES_DIR

app = FastAPI(title="Project HYDRA Cam", version="1.0.0")

# Mount static and captures directory
STATIC_DIR = os.path.abspath("static")
TEMPLATES_DIR = os.path.abspath("templates")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/captures", StaticFiles(directory=CAPTURES_DIR), name="captures")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Initialize Core Services
detector = HydraDetector(threshold=0.50, max_results=15)
streamer = CameraStreamer(detector=detector)


class ConfigUpdateModel(BaseModel):
    threshold: Optional[float] = None
    max_results: Optional[int] = None
    label_mode: Optional[str] = None
    color_theme: Optional[str] = None
    show_boxes: Optional[bool] = None
    show_labels: Optional[bool] = None
    show_scores: Optional[bool] = None
    show_reticles: Optional[bool] = None


class CameraConnectModel(BaseModel):
    source_type: str  # "ip_cam" | "webcam" | "demo"
    source_value: Any  # "http://...", 0, "demo"


# IP Camera Presets for Phone Cam Apps
PHONE_CAM_PRESETS = [
    {
        "id": "ip_webcam_android",
        "name": "IP Webcam (Android)",
        "app_name": "IP Webcam by Pavel Khlebovich",
        "url_template": "http://<PHONE_IP>:8080/video",
        "description": "Standard Android IP Webcam app. Enter phone IP shown on screen with port 8080."
    },
    {
        "id": "droidcam",
        "name": "DroidCam (Android / iOS)",
        "app_name": "DroidCam Wireless Webcam",
        "url_template": "http://<PHONE_IP>:4747/video",
        "description": "Popular low-latency phone webcam app. Default port is 4747."
    },
    {
        "id": "iriun_webcam",
        "name": "Iriun / Local USB Webcam",
        "app_name": "Iriun Webcam",
        "url_template": "0",
        "description": "If using Iriun or USB cable mode, it appears as a standard local webcam index (0 or 1)."
    },
    {
        "id": "generic_rtsp",
        "name": "RTSP Security / IP Camera",
        "app_name": "RTSP Stream",
        "url_template": "rtsp://admin:123456@<IP>:554/stream1",
        "description": "Standard RTSP video feed from security or smart home cameras."
    },
    {
        "id": "esp32_cam",
        "name": "ESP32-CAM / IoT Cam",
        "app_name": "ESP32 MJPEG Stream",
        "url_template": "http://<IP>:81/stream",
        "description": "ESP32 Camera module MJPEG web stream."
    }
]


@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    """Renders the Project HYDRA Cam ocean dashboard."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "app_title": "Project HYDRA Cam",
            "presets": PHONE_CAM_PRESETS
        }
    )


def mjpeg_generator(annotated: bool = True):
    """Generator yielding multipart MJPEG frames for browser streaming."""
    while True:
        frame_bytes = streamer.get_latest_jpeg(annotated=annotated)
        if frame_bytes is not None:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        else:
            time.sleep(0.02)
        time.sleep(0.01)  # small throttle for yield


@app.get("/video_feed")
async def video_feed():
    """Live MJPEG video stream with real-time HYDRA bounding boxes & labels."""
    return StreamingResponse(
        mjpeg_generator(annotated=True),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/raw_feed")
async def raw_feed():
    """Clean live video stream without annotation overlays."""
    return StreamingResponse(
        mjpeg_generator(annotated=False),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/status")
async def get_status():
    """Returns real-time telemetry: FPS, resolution, detections, camera state."""
    return JSONResponse(content=streamer.get_status())


@app.post("/api/config")
async def update_config(config: ConfigUpdateModel):
    """Updates detector parameters and display preferences."""
    detector.update_config(
        threshold=config.threshold,
        max_results=config.max_results,
        label_mode=config.label_mode,
        color_theme=config.color_theme,
        show_boxes=config.show_boxes,
        show_labels=config.show_labels,
        show_scores=config.show_scores,
        show_reticles=config.show_reticles
    )
    return JSONResponse(content={"success": True, "config": streamer.get_status()["detector_config"]})


@app.post("/api/camera/connect")
async def connect_camera(payload: CameraConnectModel):
    """Switches active camera source (Phone / IP Camera, Webcam, or Demo)."""
    streamer.connect(source_type=payload.source_type, source_value=payload.source_value)
    return JSONResponse(content={"success": True, "message": f"Connected to {payload.source_type}"})


@app.post("/api/camera/disconnect")
async def disconnect_camera():
    """Disconnects the active camera stream."""
    streamer.disconnect()
    return JSONResponse(content={"success": True, "message": "Camera disconnected"})


@app.get("/api/camera/presets")
async def get_presets():
    """Returns popular Phone Cam & IP Camera connection presets."""
    return JSONResponse(content={"presets": PHONE_CAM_PRESETS})


@app.post("/api/snapshot")
async def take_snapshot():
    """Captures and saves current frame in HD to captures folder."""
    result = streamer.capture_snapshot()
    if result["success"]:
        result["url"] = f"/captures/{result['filename']}"
    return JSONResponse(content=result)


@app.post("/api/record/toggle")
async def toggle_recording():
    """Starts or stops video recording."""
    result = streamer.toggle_recording()
    if result.get("filename"):
        result["url"] = f"/captures/{result['filename']}"
    return JSONResponse(content=result)


@app.get("/api/captures")
async def list_captures():
    """Lists all captured snapshots and recordings."""
    files = []
    for f in sorted(os.listdir(CAPTURES_DIR), reverse=True):
        full_path = os.path.join(CAPTURES_DIR, f)
        if os.path.isfile(full_path):
            stat = os.stat(full_path)
            is_video = f.endswith(".mp4")
            files.append({
                "filename": f,
                "url": f"/captures/{f}",
                "size_kb": round(stat.st_size / 1024, 1),
                "created_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
                "is_video": is_video
            })
    return JSONResponse(content={"captures": files})


@app.delete("/api/captures/{filename}")
async def delete_capture(filename: str):
    """Deletes a captured file."""
    # Prevent directory traversal
    clean_name = os.path.basename(filename)
    target_path = os.path.join(CAPTURES_DIR, clean_name)
    if os.path.exists(target_path):
        os.remove(target_path)
        return JSONResponse(content={"success": True, "message": f"Deleted {clean_name}"})
    raise HTTPException(status_code=404, detail="File not found")


def get_local_ip() -> str:
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def start_server(host: str = "0.0.0.0", port: int = 8000, auto_open: bool = True):
    """Starts Uvicorn server accessible on local network and opens browser."""
    local_ip = get_local_ip()
    
    if auto_open:
        def open_browser():
            time.sleep(1.2)
            webbrowser.open(f"http://127.0.0.1:{port}")
        threading.Thread(target=open_browser, daemon=True).start()

    print("\n" + "=" * 58)
    print("       >>> PROJECT HYDRA CAM - OBJECT DETECTION SYSTEM <<<")
    print("=" * 58)
    print(f"  * Laptop Access:   http://localhost:{port}")
    print(f"  * Phone Access:    http://{local_ip}:{port}")
    print(f"  * Captures Saved:  {CAPTURES_DIR}")
    print("=" * 58 + "\n")

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    start_server(host="0.0.0.0", port=8000, auto_open=True)
