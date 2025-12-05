#!/usr/bin/env python3
"""
Background Removal ROS2 Node
============================

Subscribes to a camera image topic, captures the first received frame as a
static background, saves it to background.jpg (JPEG quality 90), and for all
subsequent frames publishes the absolute difference between the current frame
and that background.

ROS Parameters
--------------
input_image_topic : str   – Input image topic (default /camera/calibrated)
output_image_topic : str  – Output image topic (default /camera/background_removed)

Usage
-----
ros2 run camera_pkg background_removal_node

With custom topics:
ros2 run camera_pkg background_removal_node --ros-args  -p input_image_topic:=/camera/calibrated -p output_image_topic:=/camera/background_removed
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os

class BackgroundRemovalNode(Node):
    def __init__(self):
        super().__init__("background_removal_node")

        # Declare parameters
        self.declare_parameter("input_image_topic", "/camera/calibrated")
        self.declare_parameter("output_image_topic", "/camera/background_removed")

        # Read parameters
        input_image_topic = self.get_parameter("input_image_topic").value
        output_image_topic = self.get_parameter("output_image_topic").value

        # CV Bridge and background storage
        self.bridge = CvBridge()
        self.background = None  # Will hold the first frame
        
        # --- Determine path to ros2_ws/src/camera_pkg/background.jpg ---
        # Start from this file's location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Walk up directory tree to find 'src/camera_pkg'
        candidate = None
        cur = script_dir
        for _ in range(10):  # walk up safely
            src_dir = os.path.join(cur, 'src')
            pkg_dir = os.path.join(src_dir, 'camera_pkg')
            if os.path.isdir(pkg_dir):
                candidate = pkg_dir
                break
            parent = os.path.dirname(cur)
            if parent == cur:  # reached root
                break
            cur = parent
        
        # Fallback: use ROS_WORKSPACE environment variable if set
        if candidate is None:
            ws = os.environ.get('ROS_WORKSPACE')
            if ws and os.path.isdir(ws):
                pkg_dir = os.path.join(ws, 'src', 'camera_pkg')
                if os.path.isdir(pkg_dir):
                    candidate = pkg_dir
        
        # Final fallback: assume ~/ros2_ws/src/camera_pkg
        if candidate is None:
            home_ws = os.path.join(os.path.expanduser('~'), 'ros2_ws', 'src', 'camera_pkg')
            candidate = home_ws
        
        self.background_path = os.path.join(candidate, 'background.jpg')
        self.get_logger().info(f"Background image path: {self.background_path}")
        self.get_logger().info("Will capture new background on first frame")

        # Subscriber
        self.image_sub = self.create_subscription(
            Image, input_image_topic, self.image_callback, 10
        )

        # Publisher
        self.image_pub = self.create_publisher(Image, output_image_topic, 10)

        self.get_logger().info("Background Removal node started")
        self.get_logger().info(f"Subscribing to: {input_image_topic}")
        self.get_logger().info(f"Publishing to: {output_image_topic}")

    def image_callback(self, msg: Image):
        """Capture background on first frame, then publish frame - background."""
        try:
            # Convert ROS Image to OpenCV (BGR8)
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            if self.background is None:
                # First frame: store as background and save to disk
                self.background = cv_img.copy()
                try:
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(self.background_path), exist_ok=True)
                    
                    cv2.imwrite(
                        self.background_path,
                        self.background,
                        [cv2.IMWRITE_JPEG_QUALITY, 90],
                    )
                    self.get_logger().info(f"Saved background image to: {self.background_path} (quality=90)")
                except Exception as e:
                    self.get_logger().error(f"Failed to save background.jpg: {e}")

                # Publish the first frame as-is
                out_img = cv_img
            else:
                # Ensure the current frame matches background size
                if cv_img.shape != self.background.shape:
                    self.get_logger().error(
                        f"Current frame shape {cv_img.shape} != background shape {self.background.shape}"
                    )
                    return

                # Absolute difference between current frame and background
                out_img = cv2.absdiff(cv_img, self.background)

            # Convert back to ROS Image and publish
            out_msg = self.bridge.cv2_to_imgmsg(out_img, encoding="bgr8")
            out_msg.header = msg.header
            self.image_pub.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = BackgroundRemovalNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()