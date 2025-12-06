#!/usr/bin/env python3
"""
ROS2 node that:
  - Subscribes to /camera/image_raw (raw camera feed)
  - Applies fisheye undistortion + fixed center crop
  - Publishes the result as a ROS2 Image on /camera/calibrated
  - Publishes CameraInfo on /camera/calibrated/camera_info
"""

import os
import yaml
import numpy as np
import cv2

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

# ---------------------------------------------------------------
# CONFIG (can be later turned into ROS2 parameters)
# ---------------------------------------------------------------

# FIXED CENTER CROP SIZE (pixels)
CROP_W = 1500   # width
CROP_H = 1500   # height

# Undistortion balance (0.0 = strongest crop / least distortion)
BALANCE = 0.0

# ---------------------------------------------------------------
# GLOBAL STATS (for logging/debug)
# ---------------------------------------------------------------
stats = {
    "frames_received": 0,
    "frames_processed": 0,
    "errors": 0
}

# ---------------------------------------------------------------
# CALIBRATION LOADING (same logic as your original script)
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
    print(f"[calibrated_camera_node] Loading calibration: {calib_file}")

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
print("[calibrated_camera_node] Preparing undistortion maps...")

new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
    K, D, (W, H), np.eye(3), balance=BALANCE
)

map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K, D, np.eye(3), new_K, (W, H), cv2.CV_16SC2
)

print("[calibrated_camera_node] ✓ Undistortion maps ready")

# ---------------------------------------------------------------
# HELPER: UNDISTORT + FIXED CENTER CROP
# ---------------------------------------------------------------
def undistort_and_center_crop(img):
    """Apply undistortion and center crop to the image."""
    global stats

    if img is None:
        return None, None, None, None

    # Ensure size matches calibration
    if img.shape[1] != W or img.shape[0] != H:
        img = cv2.resize(img, (W, H))

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

    return cropped, x1, y1, (x2 - x1, y2 - y1)

# ---------------------------------------------------------------
# ROS2 NODE
# ---------------------------------------------------------------
class CalibratedCameraNode(Node):
    def __init__(self):
        super().__init__("calibrated_camera_node")

        self.get_logger().info("=" * 60)
        self.get_logger().info("Calibrated Center-Crop Image ROS2 Node")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Subscribing to:  /camera/image_raw")
        self.get_logger().info(f"Crop Size:       {CROP_W} × {CROP_H}")
        self.get_logger().info(f"YAML Path:       {YAML_PATH}")
        self.get_logger().info("=" * 60)
        self.get_logger().info("Publishing Image on:       /camera/calibrated")
        self.get_logger().info("Publishing CameraInfo on:  /camera/calibrated/camera_info")
        self.get_logger().info("=" * 60)

        # Publishers
        self.image_pub = self.create_publisher(Image, "/camera/calibrated", 10)
        self.cinfo_pub = self.create_publisher(CameraInfo, "/camera/calibrated/camera_info", 10)

        # Subscriber
        self.image_sub = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            10
        )

        self.bridge = CvBridge()

        # Precompute crop parameters once (to also adjust CameraInfo)
        dummy = np.zeros((H, W, 3), dtype=np.uint8)
        _, x1, y1, (crop_w_eff, crop_h_eff) = undistort_and_center_crop(dummy)

        self.crop_x = x1
        self.crop_y = y1
        self.crop_w = crop_w_eff
        self.crop_h = crop_h_eff

        # Prepare CameraInfo for the rectified, cropped image
        self.camera_info = self._create_camera_info()

    def _create_camera_info(self):
        """
        Build a CameraInfo message for the rectified, cropped image.

        - Uses new_K from fisheye rectification
        - Adjusts principal point for the crop
        - Assumes rectified image (zero distortion)
        """
        cinfo = CameraInfo()
        cinfo.width = self.crop_w
        cinfo.height = self.crop_h

        # Adjust intrinsics for crop
        new_K_local = new_K.copy()
        new_K_local[0, 2] -= self.crop_x  # cx' = cx - x1
        new_K_local[1, 2] -= self.crop_y  # cy' = cy - y1

        # Fill K (3x3) and P (3x4) from this adjusted matrix
        cinfo.k = [
            float(new_K_local[0, 0]), float(new_K_local[0, 1]), float(new_K_local[0, 2]),
            float(new_K_local[1, 0]), float(new_K_local[1, 1]), float(new_K_local[1, 2]),
            float(new_K_local[2, 0]), float(new_K_local[2, 1]), float(new_K_local[2, 2]),
        ]

        cinfo.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]

        # Projection matrix P = [K | 0]
        cinfo.p = [
            float(new_K_local[0, 0]), float(new_K_local[0, 1]), float(new_K_local[0, 2]), 0.0,
            float(new_K_local[1, 0]), float(new_K_local[1, 1]), float(new_K_local[1, 2]), 0.0,
            float(new_K_local[2, 0]), float(new_K_local[2, 1]), float(new_K_local[2, 2]), 0.0,
        ]

        # Since this image is already rectified, we set zero distortion
        cinfo.distortion_model = "plumb_bob"
        cinfo.d = [0.0, 0.0, 0.0, 0.0, 0.0]

        return cinfo

    def image_callback(self, msg):
        """Callback for incoming raw camera images."""
        global stats
        
        stats["frames_received"] += 1

        try:
            # Convert ROS Image to OpenCV format
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            stats["errors"] += 1
            return

        # Apply undistortion and crop
        calibrated, _, _, _ = undistort_and_center_crop(img)
        if calibrated is None:
            self.get_logger().warn("Failed to undistort/crop frame")
            stats["errors"] += 1
            return

        # Convert back to ROS Image
        try:
            out_msg = self.bridge.cv2_to_imgmsg(calibrated, encoding="bgr8")
            out_msg.header.stamp = msg.header.stamp  # Preserve original timestamp
            out_msg.header.frame_id = "camera_calibrated"

            # Set timestamp on CameraInfo and publish
            cinfo = self.camera_info
            cinfo.header.stamp = out_msg.header.stamp
            cinfo.header.frame_id = out_msg.header.frame_id

            self.image_pub.publish(out_msg)
            self.cinfo_pub.publish(cinfo)

            self.get_logger().debug(
                f"Published calibrated frame. "
                f"received={stats['frames_received']}, processed={stats['frames_processed']}, "
                f"errors={stats['errors']}"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to publish calibrated image: {e}")
            stats["errors"] += 1

# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = CalibratedCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()