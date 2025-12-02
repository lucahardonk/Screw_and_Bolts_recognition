#!/usr/bin/env python3
''' 
This node applies the fisheye camera calibration to incoming frames and
serves the latest undistorted, center-cropped image as a single JPEG.

Pipeline:
  http://172.27.96.1:8000/frame        (raw camera, single JPEG per request)
      -> undistort + center-crop at 5 Hz (internal timer)
      -> latest calibrated frame available at:
         http://0.0.0.0:8000/frame_calibrated  (single JPEG per GET)

- /frame_calibrated: ONE full-quality JPEG (no MJPEG)
- /: HTML viewer that refreshes the image at 10 Hz via JavaScript
'''

import os
import yaml
import numpy as np
import cv2
import subprocess
import tempfile
from flask import Flask, Response, make_response, jsonify
from threading import Thread, Lock
import time

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
CAMERA_URL = "http://172.27.96.1:8000/frame"

# FIXED CENTER CROP SIZE (pixels)
CROP_W = 1500   # width
CROP_H = 1500   # height

# Undistortion balance (0.0 = strongest crop / least distortion)
BALANCE = 0.0

# Frame fetch rate (Hz)
FETCH_RATE = 5.0  # internal pull from camera

# JPEG quality settings
JPEG_QUALITY = 100

# ---------------------------------------------------------------
# FLASK APP
# ---------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------
# GLOBAL STATE
# ---------------------------------------------------------------
latest_calibrated_frame = None
frame_lock = Lock()

stats = {
    "frames_fetched": 0,
    "frames_processed": 0,
    "errors": 0
}

# ---------------------------------------------------------------
# LOAD CALIBRATION FILE
# ---------------------------------------------------------------
def load_calibration():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Walk up to find ros2_ws
    current = script_dir
    workspace_root = None
    while current != "/":
        if os.path.basename(current) == "ros2_ws":
            workspace_root = current
            break
        current = os.path.dirname(current)

    if workspace_root is None:
        base_dir = os.path.dirname(script_dir)
    else:
        base_dir = os.path.join(workspace_root, "src", "camera_pkg")

    calib_file = os.path.join(base_dir, "fisheye_camera.yaml")
    print(f"Loading calibration: {calib_file}")

    with open(calib_file, "r") as f:
        calib = yaml.safe_load(f)

    K = np.array(calib["camera_matrix"]["data"]).reshape(3, 3)
    D = np.array(calib["distortion_coefficients"]["data"]).reshape(4, 1)
    W = calib["image_width"]
    H = calib["image_height"]

    return K, D, W, H, calib_file


K, D, W, H, YAML_PATH = load_calibration()

# ---------------------------------------------------------------
# PRECOMPUTE UNDISTORT MAPS
# ---------------------------------------------------------------
print("Preparing undistortion maps...")

new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
    K, D, (W, H), np.eye(3), balance=BALANCE
)

map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K, D, np.eye(3), new_K, (W, H), cv2.CV_16SC2
)

print("✓ Undistortion maps ready")

# ---------------------------------------------------------------
# CAPTURE ONE FRAME
# ---------------------------------------------------------------
def get_frame():
    """Capture a single JPEG frame from the camera via HTTP."""
    global stats
    
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        subprocess.run(
            ["curl", "-s", "-o", tmp_path, CAMERA_URL],
            timeout=5,
            capture_output=True
        )

        img = cv2.imread(tmp_path)
        os.unlink(tmp_path)

        if img is None:
            stats["errors"] += 1
            return None

        # Ensure size matches calibration
        if img.shape[1] != W or img.shape[0] != H:
            img = cv2.resize(img, (W, H))

        stats["frames_fetched"] += 1
        return img
    
    except Exception as e:
        stats["errors"] += 1
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        print(f"Error fetching frame: {e}")
        return None

# ---------------------------------------------------------------
# UNDISTORT + FIXED CENTER CROP
# ---------------------------------------------------------------
def undistort_and_center_crop(img):
    """Apply undistortion and center crop to the image."""
    global stats
    
    if img is None:
        return None
    
    und = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

    cx = W // 2
    cy = H // 2

    x1 = cx - CROP_W // 2
    x2 = cx + CROP_W // 2
    y1 = cy - CROP_H // 2
    y2 = cy + CROP_H // 2

    # safety clamp
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)

    cropped = und[y1:y2, x1:x2]
    
    stats["frames_processed"] += 1
    return cropped

# ---------------------------------------------------------------
# BACKGROUND FRAME FETCHER
# ---------------------------------------------------------------
def frame_fetcher_loop():
    """Continuously fetch and process frames at FETCH_RATE Hz."""
    global latest_calibrated_frame
    
    print(f"Starting frame fetcher at {FETCH_RATE} Hz")
    
    while True:
        start_time = time.time()
        
        frame = get_frame()
        
        if frame is not None:
            calibrated = undistort_and_center_crop(frame)
            with frame_lock:
                latest_calibrated_frame = calibrated
        
        elapsed = time.time() - start_time
        sleep_time = max(0, (1.0 / FETCH_RATE) - elapsed)
        time.sleep(sleep_time)

# ---------------------------------------------------------------
# SINGLE IMAGE ENDPOINT
# ---------------------------------------------------------------
@app.route("/frame_calibrated")
def get_calibrated_image():
    """
    Return the latest calibrated frame as a single JPEG image.
    NOT a stream.
    """
    global latest_calibrated_frame

    with frame_lock:
        if latest_calibrated_frame is None:
            return make_response("No frame available yet", 503)
        frame = latest_calibrated_frame.copy()

    # Encode to JPEG with high quality
    encode_params = [
        int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY
    ]
    ret, jpeg = cv2.imencode(".jpg", frame, encode_params)
    if not ret:
        return make_response("Failed to encode image", 500)

    response = make_response(jpeg.tobytes())
    response.headers["Content-Type"] = "image/jpeg"
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response

# ---------------------------------------------------------------
# STATUS + ROOT (HTML VIEWER 10 Hz)
# ---------------------------------------------------------------
@app.route("/status")
def status():
    """Status endpoint with statistics."""
    with frame_lock:
        has_frame = latest_calibrated_frame is not None
    
    return jsonify({
        "status": "ok",
        "has_frame": has_frame,
        "fetch_rate_hz": FETCH_RATE,
        "crop_size": f"{CROP_W}x{CROP_H}",
        "yaml_path": YAML_PATH,
        "statistics": stats
    })

@app.route("/")
def root():
    """HTML page that refreshes the image at 10 Hz."""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Calibrated Center-Crop Viewer</title>
        <style>
            body {{
                font-family: sans-serif;
                background: #111;
                color: #eee;
                text-align: center;
            }}
            img {{
                max-width: 100%;
                height: auto;
                border: 2px solid #444;
                background: #000;
            }}
            .info {{
                margin-top: 10px;
                font-size: 14px;
                color: #aaa;
            }}
        </style>
    </head>
    <body>
        <h1>Calibrated Center-Crop Viewer</h1>
        <p>Resolution: {CROP_W} × {CROP_H}</p>
        <p>Internal fetch rate: {FETCH_RATE} Hz</p>
        <p>Image endpoint: <code>/frame_calibrated</code></p>

        <img id="calibratedImage" src="/frame_calibrated" alt="Calibrated frame">

        <div class="info">
            <p>Webpage refresh rate: 10 Hz (every 100 ms)</p>
            <p>Last update: <span id="timestamp">never</span></p>
        </div>

        <script>
            const img = document.getElementById('calibratedImage');
            const ts  = document.getElementById('timestamp');

            function updateImage() {{
                // cache-busting query param
                const url = '/frame_calibrated?rand=' + Date.now();
                img.src = url;
                ts.textContent = new Date().toLocaleTimeString();
            }}

            // 10 Hz refresh (every 100 ms)
            setInterval(updateImage, 100);
        </script>
    </body>
    </html>
    """

# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Calibrated Center-Crop Image Server")
    print("=" * 60)
    print(f"Camera URL:  {CAMERA_URL}")
    print(f"Fetch Rate:  {FETCH_RATE} Hz")
    print(f"Crop Size:   {CROP_W} × {CROP_H}")
    print("=" * 60)
    print("Single image endpoint: http://0.0.0.0:8000/frame_calibrated")
    print("Viewer page:          http://0.0.0.0:8000/")
    print("=" * 60)
    
    fetcher_thread = Thread(target=frame_fetcher_loop, daemon=True)
    fetcher_thread.start()
    
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)