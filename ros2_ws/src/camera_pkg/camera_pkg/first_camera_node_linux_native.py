#!/usr/bin/env python3

import cv2
import time
import threading
import logging
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
from cv_bridge import CvBridge
from flask import Flask, Response, jsonify

# =========================
# CONFIG
# =========================
CAMERA_DEVICE = "/dev/v4l/by-id/usb-HD_USB_Camera_HD_USB_Camera_2020040501-video-index0"

FULL_WIDTH = 3264
FULL_HEIGHT = 2448

STREAM_WIDTH = 1280
STREAM_HEIGHT = 720

TARGET_FPS = 15          # camera + stream fps
ROS_FPS = 5              # ROS publish rate
JPEG_QUALITY = 70
PORT = 8000

# =========================
# LOGGING
# =========================
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger("camera")

# =========================
# CAMERA OPEN
# =========================
def open_camera():
    cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera at {CAMERA_DEVICE}")
    
    # Set MJPEG format for highest resolution at 15 fps
    cap.set(cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FULL_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FULL_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    time.sleep(0.3)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    log.info(f"Camera opened at {CAMERA_DEVICE} ({w}x{h} @ {fps} fps)")
    return cap

# =========================
# CAPTURE THREAD
# =========================
class CameraCapture:
    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.running = True

        self.full_frame = None
        self.stream_jpeg = None

        self.last_encode = 0.0
        self.encode_interval = 1.0 / TARGET_FPS

        self.fps = 0
        self._frames = 0
        self._last_fps = time.time()

        threading.Thread(target=self._loop, daemon=True).start()
        log.info("Camera capture thread started")

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            now = time.time()

            # Store full resolution frame (for CV / ROS)
            with self.lock:
                self.full_frame = frame

            # Encode MJPEG stream at fixed rate
            if now - self.last_encode >= self.encode_interval:
                small = cv2.resize(
                    frame,
                    (STREAM_WIDTH, STREAM_HEIGHT),
                    interpolation=cv2.INTER_LINEAR
                )
                ok, buf = cv2.imencode(
                    ".jpg",
                    small,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                )
                if ok:
                    with self.lock:
                        self.stream_jpeg = buf.tobytes()
                self.last_encode = now

            # FPS calculation
            self._frames += 1
            if now - self._last_fps >= 1.0:
                self.fps = self._frames
                self._frames = 0
                self._last_fps = now

    def get_stream(self):
        with self.lock:
            return self.stream_jpeg

    def get_full_frame(self):
        with self.lock:
            if self.full_frame is None:
                return None
            return self.full_frame.copy()

# =========================
# ROS2 NODE
# =========================
class CameraNode(Node):
    def __init__(self, camera):
        super().__init__("camera_node")
        self.camera = camera
        self.bridge = CvBridge()

        self.pub_raw = self.create_publisher(Image, "camera/image_raw", 10)
        self.pub_comp = self.create_publisher(
            CompressedImage, "camera/image_raw/compressed", 10
        )
        self.pub_info = self.create_publisher(String, "camera/info", 10)

        self.create_timer(1.0 / ROS_FPS, self.publish)
        self.get_logger().info("ROS2 camera node started")

    def publish(self):
        frame = self.camera.get_full_frame()
        if frame is None:
            return

        stamp = self.get_clock().now().to_msg()

        # Raw image (full resolution)
        img = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        img.header.stamp = stamp
        img.header.frame_id = "camera"
        self.pub_raw.publish(img)

        # Compressed (downscaled)
        small = cv2.resize(frame, (STREAM_WIDTH, STREAM_HEIGHT))
        ok, buf = cv2.imencode(
            ".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        if ok:
            c = CompressedImage()
            c.header = img.header
            c.format = "jpeg"
            c.data = buf.tobytes()
            self.pub_comp.publish(c)

        info = String()
        info.data = f"Capture FPS: {self.camera.fps}"
        self.pub_info.publish(info)

# =========================
# FLASK WEB SERVER
# =========================
app = Flask(__name__)
camera = None

@app.route("/")
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Camera Stream</title>
        <style>
            body {
                font-family: Arial;
                background: #f4f4f4;
                text-align: center;
                padding: 20px;
            }
            img {
                max-width: 95%;
                border: 2px solid #333;
                border-radius: 6px;
            }
        </style>
    </head>
    <body>
        <h1>🎥 Camera Stream</h1>
        <img src="/video">
        <p><a href="/stats">View stats (JSON)</a></p>
    </body>
    </html>
    """

@app.route("/video")
def video():
    def gen():
        last = None
        while True:
            frame = camera.get_stream()
            if frame and frame != last:
                last = frame
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            time.sleep(0.01)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stats")
def stats():
    return jsonify({
        "fps": camera.fps,
        "resolution_full": f"{FULL_WIDTH}x{FULL_HEIGHT}",
        "resolution_stream": f"{STREAM_WIDTH}x{STREAM_HEIGHT}"
    })

# =========================
# MAIN
# =========================
def main():
    global camera

    cap = open_camera()
    camera = CameraCapture(cap)

    rclpy.init()
    node = CameraNode(camera)

    # Flask in separate thread
    threading.Thread(
        target=lambda: app.run(
            host="0.0.0.0",
            port=PORT,
            threaded=True,
            debug=False
        ),
        daemon=True
    ).start()

    log.info(f"Web UI: http://localhost:{PORT}")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        camera.running = False
        cap.release()
        node.destroy_node()
        rclpy.shutdown()
        log.info("Shutdown complete")

if __name__ == "__main__":
    main()