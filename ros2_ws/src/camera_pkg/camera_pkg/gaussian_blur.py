#!/usr/bin/env python3
"""
Gaussian Blur Filter ROS2 Node

This node subscribes to calibrated camera frames and applies Gaussian blur.

ROS2 PARAMETERS:
- input_image_topic: Input image topic (default: /camera/calibrated)
- output_image_topic: Output blurred image topic (default: /camera/gaussian_blurred)
- gaussian_kernel_size: Size of Gaussian kernel (must be odd: 3, 5, 7, 9, 11, etc.)
                        Larger = more blur. Typical range: 3-15 (default: 15)
- gaussian_sigma: Standard deviation for Gaussian kernel
                  0 = auto-calculate from kernel size
                  Larger = more blur. Typical range: 0.5-5.0 (default: 0.0)
- queue_size: Queue size for publishers and subscribers (default: 10)

EXAMPLE ROS2 RUN COMMAND:
ros2 run <package_name> gaussian_blur_node \
  --ros-args \
  -p input_image_topic:=/camera/calibrated \
  -p output_image_topic:=/camera/gaussian_blurred \
  -p gaussian_kernel_size:=15 \
  -p gaussian_sigma:=0.0 \
  -p queue_size:=10
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
class GaussianBlurNode(Node):
    def __init__(self):
        super().__init__("gaussian_blur_node")

        # Declare ROS2 parameters with default values and descriptions
        self.declare_parameter(
            "input_image_topic",
            "/camera/calibrated",
            ParameterDescriptor(description="Input image topic")
        )
        self.declare_parameter(
            "output_image_topic",
            "/camera/gaussian_blurred",
            ParameterDescriptor(description="Output blurred image topic")
        )
        self.declare_parameter(
            "gaussian_kernel_size",
            15,
            ParameterDescriptor(description="Gaussian kernel size (must be odd)")
        )
        self.declare_parameter(
            "gaussian_sigma",
            0.0,
            ParameterDescriptor(description="Gaussian sigma (0 = auto-calculate)")
        )
        self.declare_parameter(
            "queue_size",
            10,
            ParameterDescriptor(description="Queue size for publishers and subscribers")
        )

        # Get parameter values
        input_image_topic = self.get_parameter("input_image_topic").value
        output_image_topic = self.get_parameter("output_image_topic").value
        self.kernel_size = self.get_parameter("gaussian_kernel_size").value
        self.sigma = self.get_parameter("gaussian_sigma").value
        queue_size = self.get_parameter("queue_size").value

        # Ensure kernel size is odd
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
            self.get_logger().warn(
                f"Kernel size must be odd. Adjusted to {self.kernel_size}"
            )

        # Log configuration
        self.get_logger().info("=" * 60)
        self.get_logger().info("Gaussian Blur Filter ROS2 Node")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Kernel Size:     {self.kernel_size}")
        self.get_logger().info(f"Sigma:           {self.sigma}")
        self.get_logger().info(f"Queue Size:      {queue_size}")
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
            queue_size
        )

        # Publisher
        self.image_pub = self.create_publisher(
            Image,
            output_image_topic,
            queue_size
        )

        # Stats tracking
        self.frames_received = 0
        self.frames_processed = 0
        self.errors = 0

        self.get_logger().info("Gaussian Blur Node ready")

    def image_callback(self, msg):
        """
        Callback for incoming calibrated images.
        Applies Gaussian blur and republishes.
        """
        self.frames_received += 1

        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(
                cv_image,
                (self.kernel_size, self.kernel_size),
                self.sigma
            )

            self.frames_processed += 1

            # Convert back to ROS Image
            blurred_msg = self.bridge.cv2_to_imgmsg(blurred, encoding="bgr8")
            blurred_msg.header = msg.header  # Keep original timestamp and frame_id

            # Publish blurred image
            self.image_pub.publish(blurred_msg)

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


"""
FULL ROS2 RUN COMMAND WITH ALL PARAMETERS:

EXAMPLE WITH CUSTOM VALUES:

ros2 run <package_name> gaussian_blur_node \
  --ros-args \
  -p input_image_topic:=/my_camera/image_raw \
  -p output_image_topic:=/my_camera/blurred \
  -p gaussian_kernel_size:=9 \
  -p gaussian_sigma:=2.0 \
  -p queue_size:=5
"""