#!/usr/bin/env python3
"""
Grayscale Conversion ROS2 Node

This node subscribes to camera frames and converts them to grayscale.

ROS2 PARAMETERS:
- input_image_topic: Input image topic (default: /camera/calibrated)
- output_image_topic: Output grayscale image topic (default: /camera/grayed)

EXAMPLE ROS2 RUN COMMAND:
ros2 run camera_pkg grey_scaled_node --ros-args -p input_image_topic:=/camera/calibrated -p output_image_topic:=/camera/grayed
"""

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# ---------------------------------------------------------------
# ROS2 NODE
# ---------------------------------------------------------------
class GrayscaleNode(Node):
    def __init__(self):
        super().__init__("grey_scaled_node")

        # Declare ROS2 parameters with default values and descriptions
        self.declare_parameter(
            "input_image_topic",
            "/camera/calibrated",
            ParameterDescriptor(description="Input image topic")
        )
        self.declare_parameter(
            "output_image_topic",
            "/camera/grayed",
            ParameterDescriptor(description="Output grayscale image topic")
        )

        # Get parameter values
        input_image_topic = self.get_parameter("input_image_topic").value
        output_image_topic = self.get_parameter("output_image_topic").value

        # Log configuration
        self.get_logger().info("=" * 60)
        self.get_logger().info("Grayscale Conversion ROS2 Node")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Subscribing to:  {input_image_topic}")
        self.get_logger().info(f"Publishing to:   {output_image_topic}")
        self.get_logger().info("=" * 60)

        # CV Bridge for converting between ROS Image and OpenCV
        self.bridge = CvBridge()

        # Subscriber
        self.image_sub = self.create_subscription(
            Image,
            input_image_topic,
            self.image_callback,
            10
        )

        # Publisher
        self.image_pub = self.create_publisher(
            Image,
            output_image_topic,
            10
        )

        # Stats tracking
        self.frames_received = 0
        self.frames_processed = 0
        self.errors = 0

        self.get_logger().info("Grayscale Node ready")

    def image_callback(self, msg):
        """
        Callback for incoming images.
        Converts to grayscale and republishes.
        """
        self.frames_received += 1

        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            # Convert to grayscale
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            self.frames_processed += 1

            # Convert back to ROS Image (mono8 encoding for grayscale)
            gray_msg = self.bridge.cv2_to_imgmsg(gray_image, encoding="mono8")
            gray_msg.header = msg.header  # Keep original timestamp and frame_id

            # Publish grayscale image
            self.image_pub.publish(gray_msg)

            self.get_logger().debug(
                f"Processed frame. "
                f"received={self.frames_received}, "
                f"processed={self.frames_processed}, "
                f"errors={self.errors}"
            )

        except Exception as e:
            self.errors += 1
            self.get_logger().error(f"Error processing frame: {e}")

# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = GrayscaleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


