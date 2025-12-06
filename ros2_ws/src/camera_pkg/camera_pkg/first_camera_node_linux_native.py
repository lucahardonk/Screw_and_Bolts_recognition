#!/usr/bin/env python3
import cv2
import time
from flask import Flask, Response, jsonify
from threading import Thread, Lock
import logging

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
from cv_bridge import CvBridge

#debugging
import sys, os, cv2, subprocess

print("=== DEBUG ENV ===", flush=True)
print("sys.executable:", sys.executable, flush=True)
print("cwd:", os.getcwd(), flush=True)
print("cv2.__file__:", cv2.__file__, flush=True)
print("id:", subprocess.getoutput("id"), flush=True)
print("video devices:\n", subprocess.getoutput("ls -l /dev/video* 2>&1 || echo \"no /dev/video*\""), flush=True)
print("===============\n", flush=True)

# -------------------------
# LOGGING SETUP
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# -------------------------
# CONFIG
# -------------------------
CAMERA_NAME = "HD USB Camera"
PORT = 8000

# Requested full resolution
FULL_WIDTH = 3264
FULL_HEIGHT = 2448

STREAM_WIDTH = 1280
STREAM_HEIGHT = 720

TARGET_FPS = 15
STREAM_QUALITY = 70  # JPEG quality for stream (0-100)
SNAPSHOT_QUALITY = 100  # JPEG quality for snapshots

logger.info("Running on: Linux")

# -------------------------
# CAMERA DETECTION
# -------------------------
def find_camera():
    """
    Try to find any working camera from /dev/video* (Linux)
    """
    logger.info("Searching for camera...")

    # Try /dev/video indices 0-9
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                logger.info(f"Linux camera found at index {i}")
                return i, cap
            cap.release()

    raise RuntimeError("No working camera found")

# -------------------------
# CAMERA INIT
# -------------------------
def initialize_camera():
    """Initialize camera with optimal settings"""
    CAM_INDEX, cap = find_camera()
    
    logger.info(f"Opening camera index {CAM_INDEX}")
    
    # Force MJPEG for better performance
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    
    # Try setting high resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FULL_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FULL_HEIGHT)
    
    # FPS and buffer settings
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    time.sleep(0.3)  # Give camera time to adjust
    
    # Get actual settings
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_setting = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)])
    
    logger.info(f"Camera initialized")
    logger.info(f"Resolution = {actual_width}x{actual_height}")
    logger.info(f"FOURCC = {fourcc_str}")
    logger.info(f"FPS = {fps_setting}")
    
    # Configure camera parameters
    configure_camera_parameters(cap)
    
    return CAM_INDEX, cap, actual_width, actual_height

def configure_camera_parameters(cap):
    """Configure camera parameters for optimal image quality"""
    
    logger.info("Configuring Linux camera parameters...")
    
    # CRITICAL FIX: Enable auto exposure (3 = aperture priority mode)
    # 1 = manual, 3 = auto
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    logger.info(f"Auto Exposure: {cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}")
    
    # Enable auto white balance
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    
    # If you want manual exposure, uncomment these:
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Manual mode
    # cap.set(cv2.CAP_PROP_EXPOSURE, 156)  # Exposure time (higher = brighter)
    
    # Brightness (range typically -64 to 64, default 0)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)
    
    # Contrast (range typically 0 to 64, default 32)
    cap.set(cv2.CAP_PROP_CONTRAST, 32)
    
    # Saturation (range typically 0 to 128, default 64)
    cap.set(cv2.CAP_PROP_SATURATION, 64)
    
    # Sharpness (range typically 0 to 6, default 3)
    cap.set(cv2.CAP_PROP_SHARPNESS, 3)
    
    # Gain (range typically 0 to 100)
    cap.set(cv2.CAP_PROP_GAIN, 0)  # Let auto-exposure handle this
    
    # Log actual values
    logger.info(f"Brightness: {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
    logger.info(f"Contrast: {cap.get(cv2.CAP_PROP_CONTRAST)}")
    logger.info(f"Saturation: {cap.get(cv2.CAP_PROP_SATURATION)}")
    logger.info(f"Sharpness: {cap.get(cv2.CAP_PROP_SHARPNESS)}")
    logger.info(f"Gain: {cap.get(cv2.CAP_PROP_GAIN)}")

# -------------------------
# THREADED CAPTURE
# -------------------------
class CameraThread:
    def __init__(self, cap, ros_node=None):
        self.cap = cap
        self.ros_node = ros_node
        self.full_frame = None
        self.stream_jpeg = None
        self.lock = Lock()
        self.running = True
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
        self.thread = Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info("Camera capture thread started")

    def _capture_loop(self):
        """Main capture loop running in separate thread"""
        consecutive_failures = 0
        max_failures = 10
        
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                consecutive_failures += 1
                logger.warning(f"Frame capture failed ({consecutive_failures}/{max_failures})")
                
                if consecutive_failures >= max_failures:
                    logger.error("Too many consecutive failures, stopping capture")
                    self.running = False
                    break
                
                time.sleep(0.1)
                continue
            
            consecutive_failures = 0
            self.frame_count += 1
            
            # Calculate FPS every second
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                self.current_fps = self.frame_count / (current_time - self.last_fps_time)
                self.frame_count = 0
                self.last_fps_time = current_time
            
            # Store full resolution frame
            with self.lock:
                self.full_frame = frame.copy()
            
            # Publish to ROS2 if node is available
            if self.ros_node is not None:
                self.ros_node.publish_frame(frame)
            
            # Create stream frame (lower resolution)
            stream_frame = cv2.resize(frame, (STREAM_WIDTH, STREAM_HEIGHT), 
                                     interpolation=cv2.INTER_LINEAR)
            
            # Encode to JPEG
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, STREAM_QUALITY]
            ret_encode, buffer = cv2.imencode('.jpg', stream_frame, encode_params)
            
            if ret_encode:
                with self.lock:
                    self.stream_jpeg = buffer.tobytes()
            
            # Small delay to prevent CPU overload
            time.sleep(0.001)

    def read_stream(self):
        """Get latest stream JPEG"""
        with self.lock:
            return self.stream_jpeg

    def read_snapshot(self):
        """Get full resolution snapshot as JPEG"""
        with self.lock:
            frame = self.full_frame.copy() if self.full_frame is not None else None
        
        if frame is None:
            return None
        
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, SNAPSHOT_QUALITY]
        ret, buffer = cv2.imencode('.jpg', frame, encode_params)
        
        if ret:
            return buffer.tobytes()
        return None
    
    def get_fps(self):
        """Get current FPS"""
        return self.current_fps
    
    def stop(self):
        """Stop the capture thread"""
        self.running = False
        self.thread.join(timeout=2.0)
        logger.info("Camera capture thread stopped")

# -------------------------
# ROS2 NODE
# -------------------------
class CameraStreamNode(Node):
    def __init__(self):
        super().__init__('camera_stream_node')
        
        # Declare parameters
        self.declare_parameter('camera_name', CAMERA_NAME)
        self.declare_parameter('port', PORT)
        self.declare_parameter('full_width', FULL_WIDTH)
        self.declare_parameter('full_height', FULL_HEIGHT)
        self.declare_parameter('stream_width', STREAM_WIDTH)
        self.declare_parameter('stream_height', STREAM_HEIGHT)
        self.declare_parameter('target_fps', TARGET_FPS)
        
        # Get parameters
        self.camera_name = self.get_parameter('camera_name').value
        self.port = self.get_parameter('port').value
        
        self.get_logger().info(f'Camera Stream Node initialized')
        self.get_logger().info(f'Camera: {self.camera_name}')
        self.get_logger().info(f'Web server port: {self.port}')
        
        # Create publishers
        self.image_pub = self.create_publisher(Image, 'camera/image_raw', 10)
        self.compressed_pub = self.create_publisher(CompressedImage, 'camera/image_raw/compressed', 10)
        self.info_pub = self.create_publisher(String, 'camera/info', 10)
        
        # CV Bridge for ROS2 image conversion
        self.bridge = CvBridge()
        
        # Timer for publishing camera info
        self.create_timer(1.0, self.publish_info)
        
        self.get_logger().info('ROS2 publishers created')
    
    def publish_frame(self, frame):
        """Publish frame to ROS2 topics"""
        try:
            # Publish raw image
            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = 'camera_frame'
            self.image_pub.publish(img_msg)
            
            # Publish compressed image
            compressed_msg = CompressedImage()
            compressed_msg.header = img_msg.header
            compressed_msg.format = 'jpeg'
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, STREAM_QUALITY]
            ret, buffer = cv2.imencode('.jpg', frame, encode_params)
            if ret:
                compressed_msg.data = buffer.tobytes()
                self.compressed_pub.publish(compressed_msg)
        except Exception as e:
            self.get_logger().error(f'Error publishing frame: {e}')
    
    def publish_info(self):
        """Publish camera info periodically"""
        if camera_thread is not None:
            info_msg = String()
            info_msg.data = f"FPS: {camera_thread.get_fps():.1f}, Resolution: {actual_width}x{actual_height}"
            self.info_pub.publish(info_msg)

# -------------------------
# INITIALIZE CAMERA
# -------------------------
CAM_INDEX, cap, actual_width, actual_height = initialize_camera()

# Initialize ROS2
rclpy.init()
ros_node = CameraStreamNode()

# Create camera thread with ROS2 node
camera_thread = CameraThread(cap, ros_node)

# Give camera time to adjust exposure
logger.info("Waiting for camera to adjust exposure...")
time.sleep(2.0)

# -------------------------
# WEB SERVER
# -------------------------
app = Flask(__name__)

@app.route("/")
def index():
    """Main page with video stream"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Camera Stream</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            h1 {{
                color: #333;
            }}
            .info {{
                background: white;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .stream-container {{
                background: white;
                padding: 15px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            img {{
                max-width: 100%;
                height: auto;
                border: 2px solid #ddd;
                border-radius: 4px;
            }}
            .button {{
                display: inline-block;
                padding: 10px 20px;
                margin: 10px 5px;
                background-color: #007bff;
                color: white;
                text-decoration: none;
                border-radius: 4px;
                transition: background-color 0.3s;
            }}
            .button:hover {{
                background-color: #0056b3;
            }}
            #fps {{
                font-weight: bold;
                color: #28a745;
            }}
            .ros-badge {{
                display: inline-block;
                background: #22863a;
                color: white;
                padding: 5px 10px;
                border-radius: 3px;
                font-size: 12px;
                font-weight: bold;
            }}
        </style>
        <script>
            function updateFPS() {{
                fetch('/stats')
                    .then(response => response.json())
                    .then(data => {{
                        document.getElementById('fps').textContent = data.fps.toFixed(1);
                    }})
                    .catch(err => console.error('Error fetching FPS:', err));
            }}
            setInterval(updateFPS, 1000);
            window.onload = updateFPS;
        </script>
    </head>
    <body>
        <h1>🎥 Camera Stream <span class="ros-badge">ROS2</span></h1>
        
        <div class="info">
            <p><strong>Camera Resolution:</strong> {actual_width}x{actual_height}</p>
            <p><strong>Stream Resolution:</strong> {STREAM_WIDTH}x{STREAM_HEIGHT}</p>
            <p><strong>Current FPS:</strong> <span id="fps">--</span></p>
            <p><strong>Platform:</strong> Linux</p>
            <p><strong>ROS2 Topics:</strong></p>
            <ul>
                <li><code>/camera/image_raw</code> - Raw image stream</li>
                <li><code>/camera/image_raw/compressed</code> - Compressed image stream</li>
                <li><code>/camera/info</code> - Camera information</li>
            </ul>
        </div>
        
        <div class="stream-container">
            <h2>Live Stream</h2>
            <img src="/video" alt="Camera Stream">
        </div>
        
        <div style="margin-top: 20px;">
            <a href="/frame" class="button" target="_blank">📸 Capture Full Resolution Snapshot</a>
            <a href="/stats" class="button" target="_blank">📊 View Stats (JSON)</a>
        </div>
    </body>
    </html>
    """

@app.route("/video")
def video():
    """Video stream endpoint"""
    def gen():
        last = None
        while True:
            frame = camera_thread.read_stream()
            if frame and frame != last:
                last = frame
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            time.sleep(0.01)
    
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/frame")
def frame():
    """Full resolution snapshot endpoint"""
    img = camera_thread.read_snapshot()
    if img is None:
        return "No snapshot available yet", 503
    return Response(img, mimetype="image/jpeg")

@app.route("/stats")
def stats():
    """Statistics endpoint (JSON)"""
    return jsonify({
        "fps": camera_thread.get_fps(),
        "resolution": {
            "full": f"{actual_width}x{actual_height}",
            "stream": f"{STREAM_WIDTH}x{STREAM_HEIGHT}"
        },
        "platform": "Linux",
        "camera_index": CAM_INDEX,
        "ros2_topics": {
            "raw": "/camera/image_raw",
            "compressed": "/camera/image_raw/compressed",
            "info": "/camera/info"
        }
    })

# -------------------------
# CLEANUP
# -------------------------
def cleanup():
    """Cleanup resources on exit"""
    logger.info("Cleaning up...")
    camera_thread.stop()
    cap.release()
    ros_node.destroy_node()
    rclpy.shutdown()
    logger.info("Cleanup complete")

import atexit
atexit.register(cleanup)

# -------------------------
# MAIN
# -------------------------
def main():
    """Main function to run both Flask and ROS2"""
    # Start Flask in a separate thread
    flask_thread = Thread(target=lambda: app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True), daemon=True)
    flask_thread.start()
    
    logger.info(f"Server starting on http://0.0.0.0:{PORT}")
    logger.info(f"Access from browser: http://localhost:{PORT}")
    logger.info("ROS2 node spinning...")
    
    try:
        # Spin ROS2 node in main thread
        rclpy.spin(ros_node)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        cleanup()

if __name__ == "__main__":
    main()