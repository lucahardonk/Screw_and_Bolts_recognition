#!/usr/bin/env python3
import cv2
import time
from flask import Flask, Response
from threading import Thread, Lock

# -------------------------
# CONFIG
# -------------------------
CAM_INDEX = 0
PORT = 8000

# Full resolution for snapshots
FULL_WIDTH  = 3264
FULL_HEIGHT = 2448

# Lower resolution for video stream
STREAM_WIDTH  = 1280
STREAM_HEIGHT = 720

TARGET_FPS = 15

# -------------------------
# CAMERA INIT WITH AGGRESSIVE MODE FORCING
# -------------------------
print("[INFO] Attempting to open camera with maximum resolution...")

# Try Method 1: DirectShow with explicit mode setting
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError(f"Could not open camera at index {CAM_INDEX}")

print("[INFO] Camera opened, forcing settings...")

# Set FOURCC to MJPEG first (critical for high res)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))

# Set FULL resolution MULTIPLE times (some drivers need this)
for _ in range(3):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FULL_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FULL_HEIGHT)
    time.sleep(0.1)

# Set FPS
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

# Set buffer size
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Read back what we got
actual_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
fps_setting = cap.get(cv2.CAP_PROP_FPS)

fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

print(f"[INFO] First attempt - Resolution: {actual_width}x{actual_height}")
print(f"[INFO] FOURCC: '{fourcc_str}'")
print(f"[INFO] FPS: {fps_setting}")

# If we didn't get the resolution we want, try reopening with different backend
if actual_width != FULL_WIDTH or actual_height != FULL_HEIGHT:
    print("[WARN] Desired resolution not achieved, trying alternative method...")
    cap.release()
    time.sleep(0.5)
    
    # Try without specifying backend (let OpenCV choose)
    cap = cv2.VideoCapture(CAM_INDEX)
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not reopen camera at index {CAM_INDEX}")
    
    # Force MJPEG first
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    
    # Try setting resolution multiple times again
    for _ in range(5):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FULL_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FULL_HEIGHT)
        time.sleep(0.05)
    
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Read back again
    actual_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps_setting = cap.get(cv2.CAP_PROP_FPS)
    fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    
    print(f"[INFO] Second attempt - Resolution: {actual_width}x{actual_height}")
    print(f"[INFO] FOURCC: '{fourcc_str}'")
    print(f"[INFO] FPS: {fps_setting}")

# Image quality settings (brighter)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # manual mode on many Windows drivers
cap.set(cv2.CAP_PROP_EXPOSURE, -3)
cap.set(cv2.CAP_PROP_GAIN, 4)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 160)
cap.set(cv2.CAP_PROP_CONTRAST, 130)
cap.set(cv2.CAP_PROP_SATURATION, 128)
cap.set(cv2.CAP_PROP_SHARPNESS, 255)

print("[INFO] Camera configuration complete!")
print(f"[INFO] Final Resolution: {actual_width}x{actual_height}")
print(f"[INFO] Final FOURCC: '{fourcc_str}'")
print(f"[INFO] Final FPS: {fps_setting}")

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

if actual_width != FULL_WIDTH or actual_height != FULL_HEIGHT:
    print(f"[WARN] Could not achieve desired resolution {FULL_WIDTH}x{FULL_HEIGHT}")
    print(f"[WARN] Camera/driver only supports {actual_width}x{actual_height} via OpenCV")
    print(f"[WARN] This may be a driver limitation. Check camera settings in Windows Camera app or OBS.")

# -------------------------
# THREADED FRAME CAPTURE
# -------------------------
class CameraThread:
    def __init__(self, cap):
        self.cap = cap
        self.full_frame = None          # Store latest full-res frame (BGR)
        self.stream_jpeg = None         # Store downscaled JPEG for streaming
        self.lock = Lock()
        self.running = True
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()
        
        self.thread = Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
    
    def _capture_loop(self):
        """Continuously capture frames in background thread."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Store full-res frame for on-demand snapshots
                with self.lock:
                    self.full_frame = frame.copy()
                
                # Downscale for streaming
                stream_frame = cv2.resize(frame, (STREAM_WIDTH, STREAM_HEIGHT), interpolation=cv2.INTER_AREA)
                
                # Encode downscaled frame to JPEG (lower quality for streaming)
                ret_encode, buffer = cv2.imencode(
                    '.jpg', 
                    stream_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 60]  # Lower quality for stream
                )
                
                if ret_encode:
                    with self.lock:
                        self.stream_jpeg = buffer.tobytes()
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
    
    def read_stream_jpeg(self):
        """Get the latest downscaled JPEG for streaming."""
        with self.lock:
            return self.stream_jpeg

    
    def read_full_frame_jpeg(self):
        """
        Return the most recent FULL-RESOLUTION frame as a JPEG byte buffer.

        This method is used for the /frame endpoint, which serves a single
        high-quality snapshot (not the live low-res stream).

        Flow:
        1. Lock the thread to safely read the latest full-resolution frame.
        2. If no frame has been captured yet, return None.
        3. Encode the stored full-resolution frame (self.full_frame)
        into a JPEG at the highest possible quality (100).
        4. Convert the encoded image into raw bytes for HTTP serving.
        5. Return the byte array, or None if encoding failed.

        Notes:
        - The frame resolution is whatever the camera capture thread
        is running at (typically 3264x2448 for your SVPro camera).
        - JPEG encoding is lossy even at quality=100, but this produces
        the highest quality snapshot practical for HTTP.
        - This function is NOT used for video streaming; it's designed
        only for on-demand full-resolution capture.
        """
        with self.lock:
            if self.full_frame is None:
                return None
            
            # Encode full-res frame to high-quality JPEG on-demand
            ret_encode, buffer = cv2.imencode(
                '.jpg',
                self.full_frame,
                [cv2.IMWRITE_JPEG_QUALITY, 100]  # High quality for snapshots
            )
            
            if ret_encode:
                return buffer.tobytes()
            return None
    
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
while camera_thread.read_stream_jpeg() is None and (time.time() - start) < timeout:
    time.sleep(0.1)

if camera_thread.read_stream_jpeg() is None:
    print("[ERROR] No frames received from camera!")
    camera_thread.stop()
    cap.release()
    exit(1)

print("[INFO] First frame received!")

# -------------------------
# OPTIMIZED MJPEG GENERATOR
# -------------------------
def generate_frames():
    """Generate MJPEG stream with minimal latency (downscaled)."""
    last_frame = None
    
    while True:
        frame_bytes = camera_thread.read_stream_jpeg()
        
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
          <strong>Camera Resolution:</strong> {actual_width}x{actual_height}<br>
          <strong>Stream Resolution:</strong> {STREAM_WIDTH}x{STREAM_HEIGHT}<br>
          <strong>Snapshot Resolution:</strong> {actual_width}x{actual_height}<br>
          
        </div>
        <img src="/video" style="width: 90%; max-width: 1200px;">
        <p><a href="/frame" style="color: #4CAF50;">Full Resolution Snapshot (for calibration)</a></p>
      </body>
    </html>
    """

@app.route('/video')
def video():
    """MJPEG stream endpoint (downscaled for performance)."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/frame')
def frame():
    """Full-resolution JPEG snapshot endpoint (on-demand, high quality)."""
    frame_bytes = camera_thread.read_full_frame_jpeg()
    
    if frame_bytes is None:
        return "No frame available", 503
    
    print(f"[INFO] Serving full-res snapshot ({actual_width}x{actual_height}, {len(frame_bytes)} bytes)")
    
    return Response(frame_bytes, mimetype='image/jpeg')

# -------------------------
# MAIN
# -------------------------
if __name__ == '__main__':
    print(f"\n[INFO] Streaming on http://0.0.0.0:{PORT}")
    print(f"[INFO] MJPEG stream (downscaled): http://localhost:{PORT}/video")
    print(f"[INFO] Full-res snapshot: http://localhost:{PORT}/frame")
    print(f"[INFO] Press Ctrl+C to stop\n")
    
    try:
        app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    finally:
        camera_thread.stop()
        cap.release()
        print("[INFO] Camera released")