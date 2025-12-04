#!/usr/bin/env python3
"""
Gaussian Blur Filter ROS2 Node

This node subscribes to calibrated camera frames and applies Gaussian blur.

Input:  /camera/calibrated (sensor_msgs/Image)
Output: /camera/gaussian_blurred (sensor_msgs/Image)
        /camera/gaussian_blurred/camera_info (sensor_msgs/CameraInfo)

PARAMETERS:
- gaussian_kernel_size: Size of Gaussian kernel (must be odd: 3, 5, 7, 9, 11, etc.)
                        Larger = more blur. Typical range: 3-15
- gaussian_sigma: Standard deviation for Gaussian kernel
                  0 = auto-calculate from kernel size
                  Larger = more blur. Typical range: 0.5-5.0
"""

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

# ---------------------------------------------------------------
# CONFIG (can be later turned into ROS2 parameters)
# ---------------------------------------------------------------
GAUSSIAN_KERNEL_SIZE = 15  # Must be odd (3, 5, 7, 9, 11, etc.)
GAUSSIAN_SIGMA = 0         # 0 = auto-calculate

# ---------------------------------------------------------------
# GLOBAL STATS (for logging/debug)
# ---------------------------------------------------------------
stats = {
    "frames_received": 0,
    "frames_processed": 0,
    "errors": 0
}

# ---------------------------------------------------------------
# ROS2 NODE
# ---------------------------------------------------------------
class GaussianBlurNode(Node):
    def __init__(self):
        super().__init__("gaussian_blur_node")

        self.get_logger().info("=" * 60)
        self.get_logger().info("Gaussian Blur Filter ROS2 Node")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Kernel Size:     {GAUSSIAN_KERNEL_SIZE}")
        self.get_logger().info(f"Sigma:           {GAUSSIAN_SIGMA}")
        self.get_logger().info("=" * 60)
        self.get_logger().info("Subscribing to:  /camera/calibrated")
        self.get_logger().info("                 /camera/calibrated/camera_info")
        self.get_logger().info("Publishing to:   /camera/gaussian_blurred")
        self.get_logger().info("                 /camera/gaussian_blurred/camera_info")
        self.get_logger().info("=" * 60)

        # Ensure kernel size is odd
        self.kernel_size = GAUSSIAN_KERNEL_SIZE
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
            self.get_logger().warn(
                f"Kernel size must be odd. Adjusted to {self.kernel_size}"
            )

        self.sigma = GAUSSIAN_SIGMA

        # CV Bridge for converting between ROS Image and OpenCV
        self.bridge = CvBridge()

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            "/camera/calibrated",
            self.image_callback,
            10
        )

        self.cinfo_sub = self.create_subscription(
            CameraInfo,
            "/camera/calibrated/camera_info",
            self.cinfo_callback,
            10
        )

        # Publishers
        self.image_pub = self.create_publisher(
            Image,
            "/camera/gaussian_blurred",
            10
        )

        self.cinfo_pub = self.create_publisher(
            CameraInfo,
            "/camera/gaussian_blurred/camera_info",
            10
        )

        # Store latest camera info to republish with blurred image
        self.latest_cinfo = None

        self.get_logger().info("Gaussian Blur Node ready")

    def cinfo_callback(self, msg):
        """Store the latest CameraInfo to republish with blurred images."""
        self.latest_cinfo = msg

    def image_callback(self, msg):
        """
        Callback for incoming calibrated images.
        Applies Gaussian blur and republishes.
        """
        global stats

        stats["frames_received"] += 1

        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(
                cv_image,
                (self.kernel_size, self.kernel_size),
                self.sigma
            )

            stats["frames_processed"] += 1

            # Convert back to ROS Image
            blurred_msg = self.bridge.cv2_to_imgmsg(blurred, encoding="bgr8")
            blurred_msg.header = msg.header  # Keep original timestamp and frame_id

            # Publish blurred image
            self.image_pub.publish(blurred_msg)

            # Publish camera info if available
            if self.latest_cinfo is not None:
                cinfo = self.latest_cinfo
                cinfo.header = msg.header  # Match timestamp and frame_id
                self.cinfo_pub.publish(cinfo)

            self.get_logger().debug(
                f"Processed frame. "
                f"received={stats['frames_received']}, "
                f"processed={stats['frames_processed']}, "
                f"errors={stats['errors']}"
            )

        except Exception as e:
            stats["errors"] += 1
            self.get_logger().error(f"Error processing frame: {e}")

# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = GaussianBlurNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()