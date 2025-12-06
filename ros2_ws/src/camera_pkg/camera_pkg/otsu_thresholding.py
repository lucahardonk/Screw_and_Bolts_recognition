#!/usr/bin/env python3
"""
Otsu Thresholding ROS2 Node

This node subscribes to camera frames, converts them to grayscale
(if not already), applies Otsu thresholding, and publishes the
resulting binary image.

ROS2 PARAMETERS:
- input_image_topic: Input image topic (default: /camera/calibrated)
- output_image_topic: Output binary image topic (default: /camera/otsu)

EXAMPLE RUN:
ros2 run camera_pkg otsu_node --ros-args -p input_image_topic:=/camera/background_grayed -p output_image_topic:=/camera/otsu
"""

import cv2

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor

from sensor_msgs.msg import Image
from cv_bridge import CvBridge


# ---------------------------------------------------------------
# ROS2 NODE
# ---------------------------------------------------------------
class OtsuNode(Node):
    def __init__(self):
        super().__init__("otsu_node")

        # -------------------------------------------------------
        # Parameters
        # -------------------------------------------------------
        self.declare_parameter(
            "input_image_topic",
            "/camera/calibrated",
            ParameterDescriptor(description="Input image topic")
        )
        self.declare_parameter(
            "output_image_topic",
            "/camera/otsu",
            ParameterDescriptor(description="Output Otsu-binary image topic")
        )

        input_topic = self.get_parameter("input_image_topic").value
        output_topic = self.get_parameter("output_image_topic").value

        # -------------------------------------------------------
        # Logging configuration
        # -------------------------------------------------------
        self.get_logger().info("=" * 60)
        self.get_logger().info("Otsu Thresholding ROS2 Node")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Subscribing to:  {input_topic}")
        self.get_logger().info(f"Publishing to:   {output_topic}")
        self.get_logger().info("=" * 60)

        # Bridge to convert between ROS Image messages and OpenCV images
        self.bridge = CvBridge()

        # -------------------------------------------------------
        # Subscriber
        # -------------------------------------------------------
        self.image_sub = self.create_subscription(
            Image,
            input_topic,
            self.image_callback,
            10
        )

        # -------------------------------------------------------
        # Publisher
        # -------------------------------------------------------
        self.image_pub = self.create_publisher(
            Image,
            output_topic,
            10
        )

        self.get_logger().info("Otsu Node ready")

    def image_callback(self, msg: Image):
        """Receive image → convert to grayscale (if needed) → Otsu → publish."""
        try:
            # ---------------------------------------------------
            # Convert from ROS Image → OpenCV image
            # `passthrough` tries to respect the original encoding
            # so we can handle both color and already-grayscale images.
            # ---------------------------------------------------
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            # ---------------------------------------------------
            # Step 1: Convert to grayscale if necessary
            #
            # Cases:
            # - Already single-channel (shape: H x W)          → use as-is
            # - Single-channel with explicit 3D shape (H x W x 1) → squeeze to 2D
            # - 3-channel/4-channel color image                → convert to gray
            # ---------------------------------------------------
            if len(cv_image.shape) == 2:
                # Already grayscale (e.g. mono8)
                gray = cv_image
            elif len(cv_image.shape) == 3 and cv_image.shape[2] == 1:
                # Single-channel with trailing dimension
                gray = cv_image[:, :, 0]
            else:
                # Assume a color image (BGR or RGB).
                # OpenCV defaults generally assume BGR; if your camera
                # publishes RGB, the result will still be reasonable,
                # but you can adapt this based on `msg.encoding` if needed.
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # ---------------------------------------------------
            # Step 2: Apply Otsu's Thresholding
            #
            # With a threshold value of 0 and flag THRESH_OTSU,
            # OpenCV automatically computes the optimal threshold.
            # `thresh_image` is a binary image with values 0 or 255.
            # ---------------------------------------------------
            _, thresh_image = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # ---------------------------------------------------
            # Convert back to ROS Image (mono8) and preserve header
            # ---------------------------------------------------
            out_msg = self.bridge.cv2_to_imgmsg(thresh_image, encoding="mono8")
            out_msg.header = msg.header  # keep original timestamp, frame_id, etc.

            # ---------------------------------------------------
            # Publish result
            # ---------------------------------------------------
            self.image_pub.publish(out_msg)

        except Exception as e:
            # Catch any runtime errors (encoding issues, bad data, etc.)
            self.get_logger().error(f"Error processing frame: {e}")


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = OtsuNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()