#!/usr/bin/env python3
import cv2
import time
from flask import Flask, Response
from threading import Thread, Lock

# -------------------------
# CONFIG
# -------------------------
CAM_INDEX = 1
PORT = 8000
WIDTH  = 3264
HEIGHT = 2448
TARGET_FPS = 15

# -------------------------
# CAMERA INIT
# -------------------------
cap = cv2.VideoCapture(CAM_INDEX)

# Critical: Set FOURCC before resolution
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

# Image quality settings (reduce noise)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
cap.set(cv2.CAP_PROP_EXPOSURE, -4)
cap.set(cv2.CAP_PROP_GAIN, 0)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 140)
cap.set(cv2.CAP_PROP_CONTRAST, 130)
cap.set(cv2.CAP_PROP_SATURATION, 128)
cap.set(cv2.CAP_PROP_SHARPNESS, 255)

if not cap.isOpened():
    raise RuntimeError(f"Could not open camera at index {CAM_INDEX}")

print("[INFO] Camera opened successfully!")

# Verify camera configuration
actual_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
fps_setting = cap.get(cv2.CAP_PROP_FPS)

fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

print(f"[INFO] Actual Resolution: {actual_width}x{actual_height}")
print(f"[INFO] FOURCC: '{fourcc_str}'")
print(f"[INFO] Reported FPS: {fps_setting}")

# Get camera settings
exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
gain = cap.get(cv2.CAP_PROP_GAIN)
brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)

print(f"[INFO] Exposure: {exposure}")
print(f"[INFO] Gain: {gain}")
print(f"[INFO] Brightness: {brightness}")

# Warning if MJPEG not active
if "MJPG" not in fourcc_str and "JPEG" not in fourcc_str:
    print(f"[WARN] MJPEG not active! Got '{fourcc_str}' instead.")

# -------------------------
# THREADED FRAME CAPTURE
# -------------------------
class CameraThread:
    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.jpeg_frame = None  # Store pre-encoded JPEG
        self.lock = Lock()
        self.running = True
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()
        
        self.thread = Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
    
    def _capture_loop(self):
        """Continuously capture and encode frames in background thread."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Encode to JPEG in capture thread (faster)
                ret_encode, buffer = cv2.imencode(
                    '.jpg', 
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 85,
                     cv2.IMWRITE_JPEG_OPTIMIZE, 1]
                )
                
                if ret_encode:
                    with self.lock:
                        self.jpeg_frame = buffer.tobytes()
                        self.frame_count += 1
                    
                    # Calculate FPS
                    now = time.time()
                    if now - self.last_time >= 1.0:
                        self.fps = self.frame_count / (now - self.last_time)
                        print(f"[INFO] Capture FPS: {self.fps:.2f}")
                        self.frame_count = 0
                        self.last_time = now
            else:
                time.sleep(0.001)  # Small delay on failure
    
    def read_jpeg(self):
        """Get the latest pre-encoded JPEG frame."""
        with self.lock:
            return self.jpeg_frame
    
    def stop(self):
        """Stop the capture thread."""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)

camera_thread = CameraThread(cap)

# Wait for first frame
print("[INFO] Waiting for first frame...")
timeout = 5
start = time.time()
while camera_thread.read_jpeg() is None and (time.time() - start) < timeout:
    time.sleep(0.1)

if camera_thread.read_jpeg() is None:
    print("[ERROR] No frames received from camera!")
    camera_thread.stop()
    cap.release()
    exit(1)

print("[INFO] First frame received!")

# -------------------------
# OPTIMIZED MJPEG GENERATOR
# -------------------------
def generate_frames():
    """Generate MJPEG stream with minimal latency."""
    last_frame = None
    
    while True:
        frame_bytes = camera_thread.read_jpeg()
        
        if frame_bytes is None:
            # No new frame, wait a bit
            time.sleep(0.01)
            continue
        
        # Only send if we have a new frame
        if frame_bytes != last_frame:
            last_frame = frame_bytes
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n'
                   b'\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.01)

# -------------------------
# WEB APP
# -------------------------
app = Flask(__name__)

@app.route('/')
def index():
    return f"""
    <html>
      <head>
        <title>SVPRO Camera Stream</title>
        <style>
          body {{ font-family: Arial; margin: 20px; background: #1a1a1a; color: white; }}
          .info {{ background: #2a2a2a; padding: 15px; margin: 10px 0; border-radius: 5px; }}
          img {{ border: 2px solid #444; border-radius: 5px; }}
        </style>
      </head>
      <body>
        <h1>🎥 SVPRO 8MP Fisheye – Live Stream</h1>
        <div class="info">
          <strong>Resolution:</strong> {actual_width}x{actual_height}<br>
          <strong>Format:</strong> {fourcc_str}<br>
          <strong>Target FPS:</strong> {TARGET_FPS}
        </div>
        <img src="/video" style="width: 90%; max-width: 1200px;">
        <p><a href="/frame" style="color: #4CAF50;">Single Frame (for calibration)</a></p>
      </body>
    </html>
    """

@app.route('/video')
def video():
    """MJPEG stream endpoint."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/frame')
def frame():
    """Single JPEG frame endpoint (for calibration)."""
    frame_bytes = camera_thread.read_jpeg()
    
    if frame_bytes is None:
        return "No frame available", 503
    
    return Response(frame_bytes, mimetype='image/jpeg')

# -------------------------
# MAIN
# -------------------------
if __name__ == '__main__':
    print(f"\n[INFO] Streaming on http://0.0.0.0:{PORT}")
    print(f"[INFO] MJPEG stream: http://localhost:{PORT}/video")
    print(f"[INFO] Single frame: http://localhost:{PORT}/frame")
    print(f"[INFO] Press Ctrl+C to stop\n")
    
    try:
        app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    finally:
        camera_thread.stop()
        cap.release()
        print("[INFO] Camera released")