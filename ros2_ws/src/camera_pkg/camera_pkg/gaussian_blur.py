#!/usr/bin/env python3
"""
Gaussian Blur Filter Node

This node fetches calibrated frames from the calibration node and applies Gaussian blur.

Input:  http://127.0.0.1:8000/frame_calibrated (single JPEG from calibration node)
Output: http://127.0.0.1:8001/gaussian_blurred (single JPEG with blur applied)

PARAMETERS:
- INPUT_URL: URL to fetch calibrated frames from
- OUTPUT_PORT: Port for this Flask server
- FETCH_RATE: Rate to pull frames from input (Hz)
- GAUSSIAN_KERNEL_SIZE: Size of Gaussian kernel (must be odd: 3, 5, 7, 9, 11, etc.)
                        Larger = more blur. Typical range: 3-15
- GAUSSIAN_SIGMA: Standard deviation for Gaussian kernel
                  0 = auto-calculate from kernel size
                  Larger = more blur. Typical range: 0.5-5.0
"""

import cv2
import numpy as np
import requests
import time
from flask import Flask, Response, make_response, jsonify
from threading import Thread, Lock

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
INPUT_URL = "http://127.0.0.1:8000/frame_calibrated"
OUTPUT_PORT = 8001

# Frame fetch rate (Hz)
FETCH_RATE = 5.0  # 5 Hz = fetch frame every 0.2 seconds

# Gaussian Blur Parameters
GAUSSIAN_KERNEL_SIZE = 15  # Must be odd (3, 5, 7, 9, 11, etc.)
GAUSSIAN_SIGMA = 0        # 0 = auto-calculate

# JPEG quality settings
JPEG_QUALITY = 100

# ---------------------------------------------------------------
# FLASK APP
# ---------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------
# GLOBAL STATE
# ---------------------------------------------------------------
latest_blurred_frame = None
frame_lock = Lock()

stats = {
    "frames_fetched": 0,
    "frames_processed": 0,
    "errors": 0
}

# ---------------------------------------------------------------
# FETCH ONE FRAME FROM CALIBRATION NODE
# ---------------------------------------------------------------
def get_calibrated_frame():
    """Fetch a single JPEG frame from the calibration node."""
    global stats
    
    try:
        response = requests.get(INPUT_URL, timeout=2)
        
        if response.status_code != 200:
            stats["errors"] += 1
            return None
        
        # Decode JPEG
        arr = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        if img is None:
            stats["errors"] += 1
            return None
        
        stats["frames_fetched"] += 1
        return img
    
    except Exception as e:
        stats["errors"] += 1
        print(f"Error fetching frame: {e}")
        return None

# ---------------------------------------------------------------
# APPLY GAUSSIAN BLUR
# ---------------------------------------------------------------
def apply_gaussian_blur(img):
    """Apply Gaussian blur to the image."""
    global stats
    
    if img is None:
        return None

    # Ensure kernel size is odd
    kernel_size = GAUSSIAN_KERNEL_SIZE
    if kernel_size % 2 == 0:
        kernel_size += 1
        print(f"Warning: Kernel size must be odd. Adjusted to {kernel_size}")

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(
        img,
        (kernel_size, kernel_size),
        GAUSSIAN_SIGMA
    )

    stats["frames_processed"] += 1
    return blurred

# ---------------------------------------------------------------
# BACKGROUND FRAME FETCHER
# ---------------------------------------------------------------
def frame_fetcher_loop():
    """Continuously fetch and process frames at FETCH_RATE Hz."""
    global latest_blurred_frame
    
    print(f"Starting frame fetcher at {FETCH_RATE} Hz")
    
    while True:
        start_time = time.time()
        
        # Fetch calibrated frame
        frame = get_calibrated_frame()
        
        if frame is not None:
            # Apply Gaussian blur
            blurred = apply_gaussian_blur(frame)
            
            # Store the latest blurred frame
            with frame_lock:
                latest_blurred_frame = blurred
        
        # Sleep to maintain the desired rate
        elapsed = time.time() - start_time
        sleep_time = max(0, (1.0 / FETCH_RATE) - elapsed)
        time.sleep(sleep_time)

# ---------------------------------------------------------------
# SINGLE IMAGE ENDPOINT
# ---------------------------------------------------------------
@app.route("/gaussian_blurred")
def get_blurred_image():
    """
    Return the latest Gaussian blurred frame as a single JPEG image.
    NOT a stream.
    """
    global latest_blurred_frame

    with frame_lock:
        if latest_blurred_frame is None:
            return make_response("No frame available yet", 503)
        frame = latest_blurred_frame.copy()

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
    """Health check and statistics endpoint."""
    with frame_lock:
        has_frame = latest_blurred_frame is not None
    
    return jsonify({
        "status": "ok",
        "has_frame": has_frame,
        "input_url": INPUT_URL,
        "output_port": OUTPUT_PORT,
        "fetch_rate_hz": FETCH_RATE,
        "gaussian_kernel_size": GAUSSIAN_KERNEL_SIZE,
        "gaussian_sigma": GAUSSIAN_SIGMA,
        "statistics": stats
    })

@app.route("/")
def root():
    """HTML page that refreshes the blurred image at 10 Hz."""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Gaussian Blur Filter Viewer</title>
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
        <h1>Gaussian Blur Filter Viewer</h1>
        <p><strong>Input:</strong> <a href="{INPUT_URL}" target="_blank">{INPUT_URL}</a></p>
        <p><strong>Kernel Size:</strong> {GAUSSIAN_KERNEL_SIZE}</p>
        <p><strong>Sigma:</strong> {GAUSSIAN_SIGMA}</p>
        <p>Internal fetch rate: {FETCH_RATE} Hz</p>
        <p>Image endpoint: <code>/gaussian_blurred</code></p>

        <img id="blurredImage" src="/gaussian_blurred" alt="Gaussian blurred frame">

        <div class="info">
            <p>Webpage refresh rate: 10 Hz (every 100 ms)</p>
            <p>Last update: <span id="timestamp">never</span></p>
            <hr>
            <h3>Statistics</h3>
            <p>Frames Fetched: {stats["frames_fetched"]}</p>
            <p>Frames Processed: {stats["frames_processed"]}</p>
            <p>Errors: {stats["errors"]}</p>
        </div>

        <script>
            const img = document.getElementById('blurredImage');
            const ts  = document.getElementById('timestamp');

            function updateImage() {{
                // cache-busting query param
                const url = '/gaussian_blurred?rand=' + Date.now();
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
    print("Gaussian Blur Filter Node")
    print("=" * 60)
    print(f"Input URL:       {INPUT_URL}")
    print(f"Output Port:     {OUTPUT_PORT}")
    print(f"Fetch Rate:      {FETCH_RATE} Hz")
    print(f"Kernel Size:     {GAUSSIAN_KERNEL_SIZE}")
    print(f"Sigma:           {GAUSSIAN_SIGMA}")
    print("=" * 60)
    print(f"Single image endpoint: http://0.0.0.0:{OUTPUT_PORT}/gaussian_blurred")
    print(f"Viewer page:          http://0.0.0.0:{OUTPUT_PORT}/")
    print("=" * 60)
    
    # Start background frame fetcher thread
    fetcher_thread = Thread(target=frame_fetcher_loop, daemon=True)
    fetcher_thread.start()
    
    # Start Flask server
    app.run(host="0.0.0.0", port=OUTPUT_PORT, debug=False, threaded=True)