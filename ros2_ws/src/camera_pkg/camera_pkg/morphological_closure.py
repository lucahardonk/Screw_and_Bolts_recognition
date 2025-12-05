#!/usr/bin/env python3
"""
Morphological Closing ROS2 Node

This node subscribes to a binary image (e.g., from Otsu thresholding),
applies morphological closing to remove small holes and noise,
and publishes the cleaned binary mask.

Morphological closing = dilation followed by erosion.
It helps fill small gaps and smooth object boundaries.

ROS2 PARAMETERS:
- input_image_topic: Input binary image topic (default: /camera/otsu)
- output_image_topic: Output cleaned mask topic (default: /camera/closure)
- kernel_size: Size of the morphological kernel (default: 5)
- iterations: Number of closing iterations (default: 1)

EXAMPLE RUN:
ros2 run camera_pkg morphological_closure_node --ros-args -p input_image_topic:=/camera/otsu  -p output_image_topic:=/camera/closure -p kernel_size:=5 -p iterations:=2
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
class MorphologicalClosingNode(Node):
    def __init__(self):
        super().__init__("morphological_closing_node")

        # -------------------------------------------------------
        # Parameters
        # -------------------------------------------------------
        self.declare_parameter(
            "input_image_topic",
            "/camera/otsu",
            ParameterDescriptor(description="Input binary image topic")
        )
        self.declare_parameter(
            "output_image_topic",
            "/camera/closure",
            ParameterDescriptor(description="Output cleaned mask topic")
        )
        self.declare_parameter(
            "kernel_size",
            5,
            ParameterDescriptor(description="Size of morphological kernel (odd number)")
        )
        self.declare_parameter(
            "iterations",
            1,
            ParameterDescriptor(description="Number of closing iterations")
        )

        input_topic = self.get_parameter("input_image_topic").value
        output_topic = self.get_parameter("output_image_topic").value
        self.kernel_size = self.get_parameter("kernel_size").value
        self.iterations = self.get_parameter("iterations").value

        # Ensure kernel size is odd and positive
        if self.kernel_size < 1:
            self.kernel_size = 3
            self.get_logger().warn("kernel_size must be >= 1, setting to 3")
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
            self.get_logger().warn(f"kernel_size must be odd, adjusted to {self.kernel_size}")

        # -------------------------------------------------------
        # Create morphological kernel (rectangular structuring element)
        # You can also use cv2.MORPH_ELLIPSE or cv2.MORPH_CROSS
        # -------------------------------------------------------
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.kernel_size, self.kernel_size)
        )

        # -------------------------------------------------------
        # Logging configuration
        # -------------------------------------------------------
        self.get_logger().info("=" * 60)
        self.get_logger().info("Morphological Closing ROS2 Node")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Subscribing to:  {input_topic}")
        self.get_logger().info(f"Publishing to:   {output_topic}")
        self.get_logger().info(f"Kernel size:     {self.kernel_size}x{self.kernel_size}")
        self.get_logger().info(f"Iterations:      {self.iterations}")
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

        self.get_logger().info("Morphological Closing Node ready")

    def image_callback(self, msg: Image):
        """Receive binary image → apply morphological closing → publish cleaned mask."""
        try:
            # ---------------------------------------------------
            # Convert from ROS Image → OpenCV image
            # `passthrough` preserves the original encoding
            # (typically mono8 for binary images)
            # ---------------------------------------------------
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            # ---------------------------------------------------
            # Ensure we have a single-channel image
            # If it's 3D with 1 channel, squeeze to 2D
            # ---------------------------------------------------
            if len(cv_image.shape) == 3 and cv_image.shape[2] == 1:
                cv_image = cv_image[:, :, 0]

            # ---------------------------------------------------
            # Apply Morphological Closing
            #
            # Closing = Dilation followed by Erosion
            # - Fills small holes inside foreground objects
            # - Smooths object boundaries
            # - Connects nearby components
            #
            # cv2.morphologyEx performs the operation in one call
            # ---------------------------------------------------
            closed_image = cv2.morphologyEx(
                cv_image,
                cv2.MORPH_CLOSE,
                self.kernel,
                iterations=self.iterations
            )

            # ---------------------------------------------------
            # Convert back to ROS Image (mono8) and preserve header
            # ---------------------------------------------------
            out_msg = self.bridge.cv2_to_imgmsg(closed_image, encoding="mono8")
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
    node = MorphologicalClosingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()