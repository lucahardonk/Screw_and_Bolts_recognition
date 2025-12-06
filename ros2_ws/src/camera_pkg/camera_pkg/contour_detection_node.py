#!/usr/bin/env python3
"""
Contour Detection ROS2 Node

This node subscribes to a binary/cleaned mask image, detects contours
using OpenCV, and publishes TWO outputs:
1. A visualization image with contours drawn (/camera/contour)
2. The original binary mask (/camera/contour/binary_mask)

Contour detection finds the boundaries of objects in binary images.
This is useful for object detection, shape analysis, and segmentation.

ROS2 PARAMETERS:
- input_image_topic: Input binary mask topic (default: /camera/closure)
- output_image_topic: Output contour visualization topic (default: /camera/contour)
- output_binary_topic: Output binary mask topic (default: /camera/contour/binary_mask)
- contour_mode: Contour retrieval mode (default: "external")
  Options: "external", "list", "tree", "ccomp"
- contour_method: Contour approximation method (default: "simple")
  Options: "none", "simple", "tc89_l1", "tc89_kcos"
- contour_color_r: Red channel for contour color (default: 0)
- contour_color_g: Green channel for contour color (default: 255)
- contour_color_b: Blue channel for contour color (default: 0)
- contour_thickness: Thickness of contour lines, -1 for filled (default: 2)
- min_contour_area: Minimum contour area to draw (default: 100)

EXAMPLE RUN:
ros2 run camera_pkg contour_detection_node --ros-args \
    -p input_image_topic:=/camera/closure \
    -p output_image_topic:=/camera/contour \
    -p output_binary_topic:=/camera/contour/binary_mask \
    -p contour_color_g:=255 \
    -p contour_thickness:=2 \
    -p min_contour_area:=100.0
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
class ContourDetectionNode(Node):
    def __init__(self):
        super().__init__("contour_detection_node")

        # -------------------------------------------------------
        # Parameters
        # -------------------------------------------------------
        self.declare_parameter(
            "input_image_topic",
            "/camera/closure",
            ParameterDescriptor(description="Input binary mask topic")
        )
        self.declare_parameter(
            "output_image_topic",
            "/camera/contour",
            ParameterDescriptor(description="Output contour visualization topic")
        )
        self.declare_parameter(
            "output_binary_topic",
            "/camera/contour/binary_mask",
            ParameterDescriptor(description="Output binary mask topic (passthrough)")
        )
        self.declare_parameter(
            "contour_mode",
            "external",
            ParameterDescriptor(description="Contour retrieval mode: external, list, tree, ccomp")
        )
        self.declare_parameter(
            "contour_method",
            "simple",
            ParameterDescriptor(description="Contour approximation: none, simple, tc89_l1, tc89_kcos")
        )
        self.declare_parameter(
            "contour_color_r",
            0,
            ParameterDescriptor(description="Red channel for contour color (0-255)")
        )
        self.declare_parameter(
            "contour_color_g",
            255,
            ParameterDescriptor(description="Green channel for contour color (0-255)")
        )
        self.declare_parameter(
            "contour_color_b",
            0,
            ParameterDescriptor(description="Blue channel for contour color (0-255)")
        )
        self.declare_parameter(
            "contour_thickness",
            2,
            ParameterDescriptor(description="Thickness of contour lines (-1 for filled)")
        )
        self.declare_parameter(
            "min_contour_area",
            100.0,
            ParameterDescriptor(description="Minimum contour area to draw (pixels)")
        )

        input_topic = self.get_parameter("input_image_topic").value
        output_topic = self.get_parameter("output_image_topic").value
        output_binary_topic = self.get_parameter("output_binary_topic").value
        contour_mode_str = self.get_parameter("contour_mode").value
        contour_method_str = self.get_parameter("contour_method").value
        
        # Contour drawing parameters
        self.contour_color = (
            self.get_parameter("contour_color_b").value,  # OpenCV uses BGR
            self.get_parameter("contour_color_g").value,
            self.get_parameter("contour_color_r").value
        )
        self.contour_thickness = self.get_parameter("contour_thickness").value
        self.min_contour_area = self.get_parameter("min_contour_area").value

        # -------------------------------------------------------
        # Map string parameters to OpenCV constants
        # -------------------------------------------------------
        # Contour retrieval modes
        mode_map = {
            "external": cv2.RETR_EXTERNAL,      # Only external contours
            "list": cv2.RETR_LIST,              # All contours, no hierarchy
            "tree": cv2.RETR_TREE,              # Full hierarchy tree
            "ccomp": cv2.RETR_CCOMP             # Two-level hierarchy
        }
        self.contour_mode = mode_map.get(contour_mode_str.lower(), cv2.RETR_EXTERNAL)
        
        # Contour approximation methods
        method_map = {
            "none": cv2.CHAIN_APPROX_NONE,      # Store all contour points
            "simple": cv2.CHAIN_APPROX_SIMPLE,  # Compress horizontal/vertical/diagonal segments
            "tc89_l1": cv2.CHAIN_APPROX_TC89_L1,
            "tc89_kcos": cv2.CHAIN_APPROX_TC89_KCOS
        }
        self.contour_method = method_map.get(contour_method_str.lower(), cv2.CHAIN_APPROX_SIMPLE)

        # -------------------------------------------------------
        # Logging configuration
        # -------------------------------------------------------
        self.get_logger().info("=" * 60)
        self.get_logger().info("Contour Detection ROS2 Node")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Subscribing to:      {input_topic}")
        self.get_logger().info(f"Publishing to:       {output_topic} (visualization)")
        self.get_logger().info(f"Publishing to:       {output_binary_topic} (binary mask)")
        self.get_logger().info(f"Contour mode:        {contour_mode_str} ({self.contour_mode})")
        self.get_logger().info(f"Contour method:      {contour_method_str} ({self.contour_method})")
        self.get_logger().info(f"Contour color (BGR): {self.contour_color}")
        self.get_logger().info(f"Contour thickness:   {self.contour_thickness}")
        self.get_logger().info(f"Min contour area:    {self.min_contour_area} px²")
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
        # Publishers (TWO outputs)
        # -------------------------------------------------------
        # Publisher 1: Visualization with contours drawn
        self.image_pub = self.create_publisher(
            Image,
            output_topic,
            10
        )
        
        # Publisher 2: Binary mask (passthrough of input after filtering)
        self.binary_pub = self.create_publisher(
            Image,
            output_binary_topic,
            10
        )

        self.get_logger().info("Contour Detection Node ready")

    def image_callback(self, msg: Image):
        """Receive binary mask → detect contours → publish visualization AND binary mask."""
        try:
            # ---------------------------------------------------
            # Convert from ROS Image → OpenCV image
            # `passthrough` preserves the original encoding
            # ---------------------------------------------------
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            # ---------------------------------------------------
            # Ensure we have a single-channel binary image
            # ---------------------------------------------------
            if len(cv_image.shape) == 3:
                # If it's a 3-channel image, convert to grayscale
                if cv_image.shape[2] == 3:
                    binary_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                else:
                    binary_image = cv_image[:, :, 0]
            else:
                binary_image = cv_image

            # ---------------------------------------------------
            # Find Contours
            #
            # cv2.findContours returns:
            # - contours: list of contours (each is an array of points)
            # - hierarchy: hierarchical representation of contours
            #
            # Note: In OpenCV 4.x, findContours returns (contours, hierarchy)
            #       In OpenCV 3.x, it returns (image, contours, hierarchy)
            # ---------------------------------------------------
            contours, hierarchy = cv2.findContours(
                binary_image,
                self.contour_mode,
                self.contour_method
            )

            # ---------------------------------------------------
            # Create output image (color) for visualization
            # Convert binary image to BGR so we can draw colored contours
            # ---------------------------------------------------
            output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

            # ---------------------------------------------------
            # Filter and draw contours
            #
            # Filter by minimum area to remove noise/small contours
            # ---------------------------------------------------
            filtered_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= self.min_contour_area:
                    filtered_contours.append(contour)

            # Draw all filtered contours on the visualization image
            # -1 means draw all contours in the list
            cv2.drawContours(
                output_image,
                filtered_contours,
                -1,  # Draw all contours
                self.contour_color,
                self.contour_thickness
            )

            # ---------------------------------------------------
            # Log contour statistics
            # ---------------------------------------------------
            self.get_logger().info(
                f"Found {len(contours)} contours, "
                f"drew {len(filtered_contours)} (area >= {self.min_contour_area})",
                throttle_duration_sec=1.0  # Log once per second to avoid spam
            )

            # ---------------------------------------------------
            # Publish 1: Visualization with contours (bgr8)
            # ---------------------------------------------------
            viz_msg = self.bridge.cv2_to_imgmsg(output_image, encoding="bgr8")
            viz_msg.header = msg.header  # keep original timestamp, frame_id, etc.
            self.image_pub.publish(viz_msg)

            # ---------------------------------------------------
            # Publish 2: Binary mask (mono8)
            # This is the original binary image, unchanged
            # ---------------------------------------------------
            binary_msg = self.bridge.cv2_to_imgmsg(binary_image, encoding="mono8")
            binary_msg.header = msg.header  # keep original timestamp, frame_id, etc.
            self.binary_pub.publish(binary_msg)

        except Exception as e:
            # Catch any runtime errors (encoding issues, bad data, etc.)
            self.get_logger().error(f"Error processing frame: {e}")


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = ContourDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()