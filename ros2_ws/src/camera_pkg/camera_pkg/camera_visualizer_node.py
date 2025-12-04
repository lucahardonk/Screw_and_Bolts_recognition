#!/usr/bin/env python3
"""
Camera Visualizer ROS2 Node
===========================

Subscribes to N sensor_msgs/Image topics (parameter `topics`)
and exposes a web viewer on http://localhost:<port>/ that shows
all streams tiled side-by-side with title, resolution and FPS.

ROS Parameters
--------------
topics : list(str)    mandatory list of image topics
port   : int          HTTP port for the viewer (default 8001)
refresh_hz : float    browser refresh rate (default 10 Hz)


Usage example
-------------
ros2 run camera_pkg camera_visualizer_node --ros-args -p topics:="[/camera/calibrated,/camera/gaussian_blurred]"
"""
import os  
import threading  
import time  
from functools import partial  
from typing import Dict, Optional, Tuple  
import base64  
  
import cv2  
import numpy as np  
import rclpy  
from rclpy.node import Node  
from sensor_msgs.msg import Image  
from cv_bridge import CvBridge  
  
from flask import Flask, make_response, jsonify, send_from_directory  
  
# ------------------------------------------------------------------  
# Flask app (created once; run in background thread)  
# ------------------------------------------------------------------  
app = Flask(__name__)  
_flask_ready = threading.Event()  # used only for clean startup logging  
  
  
# ------------------------------------------------------------------  
# Helper: slug encoding/decoding  
# ------------------------------------------------------------------  
def topic_to_slug(topic: str) -> str:  
    """Convert topic name to URL-safe slug using base64."""  
    return base64.urlsafe_b64encode(topic.encode()).decode().rstrip('=')  
  
  
def slug_to_topic(slug: str) -> str:  
    """Convert slug back to topic name."""  
    # Add back padding if needed  
    padding = 4 - (len(slug) % 4)  
    if padding != 4:  
        slug += '=' * padding  
    return base64.urlsafe_b64decode(slug.encode()).decode()  
  
  
# ------------------------------------------------------------------  
# Helper structures held by the ROS node  
# ------------------------------------------------------------------  
class StreamStats:  
    """Keeps latest frame + simple stats for one topic."""  
    def __init__(self):  
        self.lock = threading.Lock()  
        self.frame: Optional[np.ndarray] = None  
        self.width = self.height = 0  
        self.frames = 0  
        self.start_time = time.time()  
        self.fps = 0.0  
  
    def update(self, cv_img: np.ndarray):  
        with self.lock:  
            self.frame = cv_img  
            self.height, self.width = cv_img.shape[:2]  
            self.frames += 1  
            dt = time.time() - self.start_time  
            if dt > 0.0:  
                self.fps = self.frames / dt  
  
    def get(self) -> Tuple[Optional[np.ndarray], int, int, float]:  
        with self.lock:  
            return (  
                None if self.frame is None else self.frame.copy(),  
                self.width,  
                self.height,  
                self.fps,  
            )  
  
  
# ------------------------------------------------------------------  
# ROS2 Node  
# ------------------------------------------------------------------  
class CameraVisualizerNode(Node):  
    def __init__(self):  
        super().__init__("camera_visualizer_node")  
  
        # Declare + read parameters  
        self.declare_parameter("topics", rclpy.Parameter.Type.STRING_ARRAY)  
        self.declare_parameter("port", 8001)  
        self.declare_parameter("refresh_hz", 10.0)  
  
        topics_param = self.get_parameter("topics").value  
        self.topics = list(topics_param) if topics_param else []  
        self.port: int = self.get_parameter("port").value  
        self.refresh_hz: float = self.get_parameter("refresh_hz").value  
  
        if not self.topics:  
            self.get_logger().error("Parameter 'topics' is empty – nothing to visualize.")  
            raise RuntimeError("No topics given")  
  
        self.get_logger().info(f"Visualizing {len(self.topics)} topics: {self.topics}")  
        self.get_logger().info(f"HTTP port: {self.port}, browser refresh: {self.refresh_hz} Hz")  
  
        # Structures to store frames & stats  
        self.bridge = CvBridge()  
        self.streams: Dict[str, StreamStats] = {t: StreamStats() for t in self.topics}  
  
        # ROS subscriptions  
        for topic in self.topics:  
            self.create_subscription(Image, topic,  
                                     partial(self.image_cb, topic),  
                                     qos_profile=10)  
  
        # Expose viewer parameters to Flask through globals  
        app.config["STREAM_TOPICS"] = self.topics  
        app.config["STREAMS_DICT"] = self.streams  
        app.config["REFRESH_MS"] = int(1000.0 / self.refresh_hz)  
  
        # Start Flask in background thread  
        threading.Thread(target=self._run_flask, daemon=True).start()  
  
    # ------------------------------------------------------------------  
    # ROS image callback  
    # ------------------------------------------------------------------  
    def image_cb(self, topic: str, msg: Image):  
        try:  
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")  
            self.streams[topic].update(cv_img)  
        except Exception as exc:  
            self.get_logger().warning(f"Failed converting image from {topic}: {exc}")  
  
    # ------------------------------------------------------------------  
    # Flask run helper  
    # ------------------------------------------------------------------  
    def _run_flask(self):  
        """Runs Flask; called in a daemon thread."""  
        _flask_ready.set()  
        # Suppress Werkzeug banner unless ROS node is in debug  
        app.run(host="0.0.0.0", port=self.port, debug=False, threaded=True,  
                use_reloader=False)  # reloader=False avoids doubling threads  
  
  
# ------------------------------------------------------------------  
# Flask routes  
# ------------------------------------------------------------------  
@app.route("/")
def index():
    topics = app.config["STREAM_TOPICS"]
    refresh = app.config["REFRESH_MS"]
    # Responsive CSS grid (auto-fit min 400 px tiles)
    tiles = []
    for t in topics:
        slug = topic_to_slug(t)
        tiles.append(f"""
        <div class="tile">
            <h3>{t}</h3>
            <img id="{slug}" src="/img/{slug}" alt="{t}">
            <p id="{slug}_info">loading…</p>
        </div>""")
    tiles_html = "\n".join(tiles)
    
    # Build JavaScript array properly
    slugs_js = "[" + ",".join([f'"{topic_to_slug(t)}"' for t in topics]) + "]"
    topics_js = "[" + ",".join([f'"{t}"' for t in topics]) + "]"
    
    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ROS Camera Visualizer</title>
<style>
  body{{margin:0;background:#111;color:#eee;font-family:sans-serif}}
  h1{{padding:20px;margin:0;font-size:32px}}
  .grid{{display:grid;grid-gap:20px;padding:20px;
        grid-template-columns:repeat(auto-fit,minmax(800px,1fr));}}
  .tile{{background:#222;padding:20px;border-radius:12px;text-align:center}}
  .tile h3{{font-size:24px;margin:0 0 15px 0}}
  img{{max-width:100%;height:auto;border:3px solid #444}}
  p{{font-size:18px;color:#aaa;margin:10px 0 0}}
</style>
</head>
<body>
<h1>ROS Camera Visualizer</h1>
<div class="grid">
{tiles_html}
</div>
<script>
const refresh = {refresh};
const slugs = {slugs_js};
const topics = {topics_js};
function update() {{
  slugs.forEach(s => {{
    const img = document.getElementById(s);
    img.src = `/img/${{s}}?rand=${{Date.now()}}`;
  }});
  fetch("/stats").then(r=>r.json()).then(js=>{{
    slugs.forEach((s,i)=>{{
      const info = document.getElementById(s+"_info");
      const t = topics[i];
      const st = js[t] || {{}};
      if(st.width)
        info.textContent = `${{st.width}}×${{st.height}}  –  ${{st.fps.toFixed(1)}} fps`;
    }});
  }});
}}
setInterval(update, refresh);
update();
</script>
</body>
</html>
"""
  
  
@app.route("/img/<slug>")  
def img_route(slug: str):  
    try:  
        topic = slug_to_topic(slug)  
    except Exception:  
        return make_response("Invalid slug", 800)  
      
    streams: Dict[str, StreamStats] = app.config["STREAMS_DICT"]  
    if topic not in streams:  
        return make_response(f"Topic not found: {topic}", 404)  
  
    frame, _, _, _ = streams[topic].get()  
    if frame is None:  
        return make_response("No frame yet", 503)  
  
    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])  
    if not ok:  
        return make_response("Encoding failed", 500)  
  
    resp = make_response(jpg.tobytes())  
    resp.headers["Content-Type"] = "image/jpeg"  
    resp.headers["Cache-Control"] = "no-store"  
    return resp  
  
  
@app.route("/stats")  
def stats_route():  
    streams: Dict[str, StreamStats] = app.config["STREAMS_DICT"]  
    data = {}  
    for topic, st in streams.items():  
        _, w, h, fps = st.get()  
        data[topic] = {"width": w, "height": h, "fps": fps}  
    return jsonify(data)  
  
  
# ------------------------------------------------------------------  
# MAIN  
# ------------------------------------------------------------------  
def main(args=None):  
    rclpy.init(args=args)  
    node = CameraVisualizerNode()  
  
    # Wait until Flask thread started so we can print the URL nicely  
    _flask_ready.wait()  
    node.get_logger().info(f"Open your browser at http://localhost:{node.port}/")  
  
    try:  
        rclpy.spin(node)  
    except KeyboardInterrupt:  
        pass  
    finally:  
        node.destroy_node()  
        rclpy.shutdown()  
  
  
if __name__ == "__main__":  
    main()