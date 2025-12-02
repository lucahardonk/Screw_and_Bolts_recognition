#!/usr/bin/env python3

import os
import yaml
import numpy as np
import cv2
import subprocess
import tempfile
from flask import Flask, Response

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
CAMERA_URL = "http://172.27.96.1:8000/frame"

# FIXED CENTER CROP SIZE (pixels)
CROP_W = 1500   # width
CROP_H = 1500   # height

# Undistortion balance (0.0 = strongest crop / least distortion)
BALANCE = 0.0

# ---------------------------------------------------------------
# FLASK APP
# ---------------------------------------------------------------
app = Flask(__name__)

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
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    result = subprocess.run(
        ["curl", "-s", "-o", tmp_path, CAMERA_URL],
        timeout=5,
        capture_output=True
    )

    img = cv2.imread(tmp_path)
    os.unlink(tmp_path)

    if img is None:
        return None

    # Ensure size matches calibration
    if img.shape[1] != W or img.shape[0] != H:
        img = cv2.resize(img, (W, H))

    return img

# ---------------------------------------------------------------
# UNDISTORT + FIXED CENTER CROP
# ---------------------------------------------------------------
def undistort_and_center_crop(img):
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

    return cropped

# ---------------------------------------------------------------
# MJPEG STREAM
# ---------------------------------------------------------------
def mjpeg_stream():
    while True:
        frame = get_frame()
        if frame is None:
            continue

        calibrated = undistort_and_center_crop(frame)

        ret, jpeg = cv2.imencode(".jpg", calibrated)
        if not ret:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               jpeg.tobytes() +
               b"\r\n")

# ---------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------
@app.route("/frame_calibrated")
def stream_output():
    return Response(
        mjpeg_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/")
def root():
    return f"""
    <h1>Calibrated Center-Crop Stream</h1>
    <p>Resolution: {CROP_W} × {CROP_H}</p>
    <p>YAML: {YAML_PATH}</p>
    <p>URL: <a href="/frame_calibrated">/frame_calibrated</a></p>
    """

# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("Streaming at: http://0.0.0.0:8000/frame_calibrated")
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
