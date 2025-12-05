#!/usr/bin/env python3
"""
Canny Edge Detection ROS2 Node
===============================

Subscribes to /camera/calibrated and applies Canny edge detection.
Publishes the edge-detected image to /camera/canny.

ROS Parameters
--------------
low_threshold : int    – Canny low threshold (default 50)
high_threshold : int   – Canny high threshold (default 150)
aperture_size : int    – Sobel kernel size, must be 3, 5, or 7 (default 3)
l2_gradient : bool     – Use L2 norm for gradient (default False)

Usage
-----
ros2 run camera_pkg canny_edge_node

# With custom thresholds:
ros2 run camera_pkg canny_edge_node --ros-args -p low_threshold:=20 -p high_threshold:=100 -p aperture_size:=3 -p l2_gradient:=false
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np


class CannyEdgeNode(Node):
    def __init__(self):
        super().__init__("canny_edge_node")

        # Declare parameters
        self.declare_parameter("low_threshold", 50)
        self.declare_parameter("high_threshold", 150)
        self.declare_parameter("aperture_size", 3)
        self.declare_parameter("l2_gradient", False)

        # Read parameters
        self.low_threshold = self.get_parameter("low_threshold").value
        self.high_threshold = self.get_parameter("high_threshold").value
        self.aperture_size = self.get_parameter("aperture_size").value
        self.l2_gradient = self.get_parameter("l2_gradient").value

        # Validate aperture_size
        if self.aperture_size not in [3, 5, 7]:
            self.get_logger().error(f"aperture_size must be 3, 5, or 7. Got {self.aperture_size}")
            raise ValueError("Invalid aperture_size")

        self.get_logger().info(
            f"Canny parameters: low={self.low_threshold}, high={self.high_threshold}, "
            f"aperture={self.aperture_size}, L2={self.l2_gradient}"
        )

        # CV Bridge
        self.bridge = CvBridge()

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, "/camera/calibrated", self.image_callback, 10
        )
        self.cinfo_sub = self.create_subscription(
            CameraInfo, "/camera/calibrated/camera_info", self.cinfo_callback, 10
        )

        # Publishers
        self.image_pub = self.create_publisher(Image, "/camera/canny", 10)
        self.cinfo_pub = self.create_publisher(CameraInfo, "/camera/canny/camera_info", 10)

        # Store latest CameraInfo
        self.latest_cinfo = None

        self.get_logger().info("Canny Edge Detection node started")
        self.get_logger().info("Subscribing to: /camera/calibrated")
        self.get_logger().info("Publishing to: /camera/canny")

    def cinfo_callback(self, msg: CameraInfo):
        """Store the latest CameraInfo to republish."""
        self.latest_cinfo = msg

    def image_callback(self, msg: Image):
        """Apply Canny edge detection and publish."""
        try:
            # Convert ROS Image to OpenCV
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            # Convert to grayscale (Canny requires single channel)
            #gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

           

            # Apply Canny edge detection
            edges = cv2.Canny(
                cv_img,
                self.low_threshold,
                self.high_threshold,
                apertureSize=self.aperture_size,
                L2gradient=self.l2_gradient,
            )

            # Convert back to BGR for visualization (edges will be white on black)
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            # Convert back to ROS Image
            out_msg = self.bridge.cv2_to_imgmsg(edges_bgr, encoding="bgr8")
            out_msg.header = msg.header

            # Publish
            self.image_pub.publish(out_msg)

            # Republish CameraInfo (geometry unchanged)
            if self.latest_cinfo is not None:
                cinfo = self.latest_cinfo
                cinfo.header = msg.header
                self.cinfo_pub.publish(cinfo)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = CannyEdgeNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()