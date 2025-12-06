#!/usr/bin/env python3
"""
Minimum Area Rectangle Detection ROS2 Node

This node subscribes to a BINARY MASK image (from contour detection node),
detects contours, computes the minimum area rectangle for each contour,
and publishes an annotated image showing:
- Bounding boxes (rotated rectangles)
- Center coordinates
- Width and height
- Orientation angle

The minimum area rectangle is the smallest rotated rectangle that
completely encloses a contour, useful for object pose estimation.

ROS2 PARAMETERS:
- input_image_topic: Input BINARY MASK topic (default: /camera/contour/binary_mask)
- output_image_topic: Output annotated image topic (default: /camera/objects)
- min_contour_area: Minimum contour area to process (default: 100.0)
- box_color_r: Red channel for box color (default: 0)
- box_color_g: Green channel for box color (default: 255)
- box_color_b: Blue channel for box color (default: 0)
- box_thickness: Thickness of box lines (default: 2)
- text_color_r: Red channel for text color (default: 255)
- text_color_g: Green channel for text color (default: 255)
- text_color_b: Blue channel for text color (default: 0)
- text_scale: Font scale for text (default: 0.5)
- text_thickness: Thickness of text (default: 1)
- show_center_dot: Draw a dot at the center (default: True)
- center_dot_radius: Radius of center dot (default: 5)

EXAMPLE RUN:
ros2 run camera_pkg min_rect_area_node --ros-args \
    -p input_image_topic:=/camera/contour/binary_mask \
    -p output_image_topic:=/camera/objects \
    -p min_contour_area:=500.0 \
    -p box_color_g:=255 \
    -p text_scale:=0.6
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
class MinRectAreaNode(Node):
    def __init__(self):
        super().__init__("min_rect_area_node")

        # -------------------------------------------------------
        # Parameters - Topics
        # -------------------------------------------------------
        self.declare_parameter(
            "input_image_topic",
            "/camera/contour/binary_mask",
            ParameterDescriptor(description="Input BINARY MASK topic from contour detection node")
        )
        self.declare_parameter(
            "output_image_topic",
            "/camera/objects",
            ParameterDescriptor(description="Output annotated image topic")
        )

        # -------------------------------------------------------
        # Parameters - Contour filtering
        # -------------------------------------------------------
        self.declare_parameter(
            "min_contour_area",
            100.0,
            ParameterDescriptor(description="Minimum contour area to process (pixels²)")
        )

        # -------------------------------------------------------
        # Parameters - Box visualization
        # -------------------------------------------------------
        self.declare_parameter(
            "box_color_r",
            0,
            ParameterDescriptor(description="Red channel for box color (0-255)")
        )
        self.declare_parameter(
            "box_color_g",
            255,
            ParameterDescriptor(description="Green channel for box color (0-255)")
        )
        self.declare_parameter(
            "box_color_b",
            0,
            ParameterDescriptor(description="Blue channel for box color (0-255)")
        )
        self.declare_parameter(
            "box_thickness",
            2,
            ParameterDescriptor(description="Thickness of box lines")
        )

        # -------------------------------------------------------
        # Parameters - Text visualization
        # -------------------------------------------------------
        self.declare_parameter(
            "text_color_r",
            255,
            ParameterDescriptor(description="Red channel for text color (0-255)")
        )
        self.declare_parameter(
            "text_color_g",
            255,
            ParameterDescriptor(description="Green channel for text color (0-255)")
        )
        self.declare_parameter(
            "text_color_b",
            0,
            ParameterDescriptor(description="Blue channel for text color (0-255)")
        )
        self.declare_parameter(
            "text_scale",
            0.5,
            ParameterDescriptor(description="Font scale for text")
        )
        self.declare_parameter(
            "text_thickness",
            1,
            ParameterDescriptor(description="Thickness of text")
        )

        # -------------------------------------------------------
        # Parameters - Center dot visualization
        # -------------------------------------------------------
        self.declare_parameter(
            "show_center_dot",
            True,
            ParameterDescriptor(description="Draw a dot at the center of each object")
        )
        self.declare_parameter(
            "center_dot_radius",
            5,
            ParameterDescriptor(description="Radius of center dot (pixels)")
        )

        # -------------------------------------------------------
        # Get parameter values
        # -------------------------------------------------------
        input_topic = self.get_parameter("input_image_topic").value
        output_topic = self.get_parameter("output_image_topic").value
        
        self.min_contour_area = self.get_parameter("min_contour_area").value
        
        # Box color (BGR format for OpenCV)
        self.box_color = (
            self.get_parameter("box_color_b").value,
            self.get_parameter("box_color_g").value,
            self.get_parameter("box_color_r").value
        )
        self.box_thickness = self.get_parameter("box_thickness").value
        
        # Text color (BGR format for OpenCV)
        self.text_color = (
            self.get_parameter("text_color_b").value,
            self.get_parameter("text_color_g").value,
            self.get_parameter("text_color_r").value
        )
        self.text_scale = self.get_parameter("text_scale").value
        self.text_thickness = self.get_parameter("text_thickness").value
        
        # Center dot
        self.show_center_dot = self.get_parameter("show_center_dot").value
        self.center_dot_radius = self.get_parameter("center_dot_radius").value

        # -------------------------------------------------------
        # Logging configuration
        # -------------------------------------------------------
        self.get_logger().info("=" * 60)
        self.get_logger().info("Minimum Area Rectangle Detection ROS2 Node")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Subscribing to:      {input_topic}")
        self.get_logger().info(f"Publishing to:       {output_topic}")
        self.get_logger().info(f"Min contour area:    {self.min_contour_area} px²")
        self.get_logger().info(f"Box color (BGR):     {self.box_color}")
        self.get_logger().info(f"Box thickness:       {self.box_thickness}")
        self.get_logger().info(f"Text color (BGR):    {self.text_color}")
        self.get_logger().info(f"Text scale:          {self.text_scale}")
        self.get_logger().info(f"Show center dot:     {self.show_center_dot}")
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

        self.get_logger().info("Minimum Area Rectangle Node ready")

    def image_callback(self, msg: Image):
        """
        Receive BINARY MASK → detect contours → compute min area rectangles → 
        annotate with boxes and info → publish.
        """
        try:
            # ---------------------------------------------------
            # Convert from ROS Image → OpenCV image
            # Expect mono8 encoding (binary mask)
            # ---------------------------------------------------
            binary_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")

            # ---------------------------------------------------
            # Ensure we have a single-channel image
            # (Should already be single-channel from mono8, but just in case)
            # ---------------------------------------------------
            if len(binary_image.shape) == 3:
                binary_image = binary_image[:, :, 0]

            # ---------------------------------------------------
            # Create color output image for visualization
            # Convert binary mask to BGR so we can draw colored annotations
            # ---------------------------------------------------
            output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

            # ---------------------------------------------------
            # Find Contours
            # Use RETR_EXTERNAL to get only outer contours
            # Use CHAIN_APPROX_SIMPLE to compress contour points
            #
            # NOTE: This is the ONLY place we detect contours.
            # We're using the binary mask from the contour_detection_node,
            # so the contours should match what was visualized there.
            # ---------------------------------------------------
            contours, _ = cv2.findContours(
                binary_image,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            # ---------------------------------------------------
            # Process each contour
            # ---------------------------------------------------
            object_count = 0
            
            for contour in contours:
                # Calculate contour area
                area = cv2.contourArea(contour)
                
                # Filter by minimum area
                if area < self.min_contour_area:
                    continue
                
                object_count += 1

                # ---------------------------------------------------
                # Compute Minimum Area Rectangle
                #
                # cv2.minAreaRect returns:
                # - center: (x, y) center coordinates
                # - size: (width, height) dimensions
                # - angle: rotation angle in degrees
                #
                # Note: The angle is between -90 and 0 degrees
                # ---------------------------------------------------
                rect = cv2.minAreaRect(contour)
                center, size, angle = rect
                
                # Extract values
                center_x, center_y = center
                width, height = size

                # ---------------------------------------------------
                # Get box corner points
                # cv2.boxPoints converts the rotated rectangle to 4 corner points
                # ---------------------------------------------------
                box_points = cv2.boxPoints(rect)
                box_points = np.int0(box_points)  # Convert to integer coordinates

                # ---------------------------------------------------
                # Draw the rotated rectangle
                # ---------------------------------------------------
                cv2.drawContours(
                    output_image,
                    [box_points],
                    0,  # Draw the first (and only) contour in the list
                    self.box_color,
                    self.box_thickness
                )

                # ---------------------------------------------------
                # Draw center dot (optional)
                # ---------------------------------------------------
                if self.show_center_dot:
                    cv2.circle(
                        output_image,
                        (int(center_x), int(center_y)),
                        self.center_dot_radius,
                        self.box_color,
                        -1  # Filled circle
                    )

                # ---------------------------------------------------
                # Prepare annotation text
                # ---------------------------------------------------
                # Line 1: Object ID
                text_id = f"Object #{object_count}"
                
                # Line 2: Center coordinates
                text_center = f"Center: ({center_x:.1f}, {center_y:.1f})"
                
                # Line 3: Width and Height
                text_size = f"Size: {width:.1f} x {height:.1f}"
                
                # Line 4: Orientation angle
                text_angle = f"Angle: {angle:.1f} deg"

                # ---------------------------------------------------
                # Position text near the object
                # Place text above the center, offset to avoid overlap
                # ---------------------------------------------------
                text_x = int(center_x) + 10
                text_y_start = int(center_y) - 60
                line_spacing = 20

                # Draw each line of text
                cv2.putText(
                    output_image,
                    text_id,
                    (text_x, text_y_start),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.text_scale,
                    self.text_color,
                    self.text_thickness,
                    cv2.LINE_AA
                )
                
                cv2.putText(
                    output_image,
                    text_center,
                    (text_x, text_y_start + line_spacing),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.text_scale,
                    self.text_color,
                    self.text_thickness,
                    cv2.LINE_AA
                )
                
                cv2.putText(
                    output_image,
                    text_size,
                    (text_x, text_y_start + 2 * line_spacing),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.text_scale,
                    self.text_color,
                    self.text_thickness,
                    cv2.LINE_AA
                )
                
                cv2.putText(
                    output_image,
                    text_angle,
                    (text_x, text_y_start + 3 * line_spacing),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.text_scale,
                    self.text_color,
                    self.text_thickness,
                    cv2.LINE_AA
                )

            # ---------------------------------------------------
            # Log detection statistics
            # ---------------------------------------------------
            self.get_logger().info(
                f"Detected {object_count} objects (area >= {self.min_contour_area})",
                throttle_duration_sec=1.0  # Log once per second
            )

            # ---------------------------------------------------
            # Convert back to ROS Image (bgr8) and preserve header
            # ---------------------------------------------------
            out_msg = self.bridge.cv2_to_imgmsg(output_image, encoding="bgr8")
            out_msg.header = msg.header  # keep original timestamp, frame_id, etc.

            # ---------------------------------------------------
            # Publish result
            # ---------------------------------------------------
            self.image_pub.publish(out_msg)

        except Exception as e:
            # Catch any runtime errors
            self.get_logger().error(f"Error processing frame: {e}")


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = MinRectAreaNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()