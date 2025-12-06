#!/usr/bin/env python3
"""
Minimum Area Rectangle + Principal Axis Detection ROS2 Node

This node subscribes to a BINARY IMAGE (from morphological closure),
detects contours, computes for each object:

- OpenCV minimum area rectangle
- Principal axis from image moments (orientation)
- Principal-axis–aligned bounding box (via PCA)

It publishes:

1) An annotated IMAGE on `output_image_topic` showing:
   - Green: OpenCV min-area rotated rectangle
   - Red: principal axis line (orientation)
   - Blue: principal-axis–aligned bounding box
   - Center coordinates and principal-axis angle as text

2) A JSON STRING on `output_info_topic` with, per object:
   - object_id
   - center_px: [cx, cy] in pixels
   - moment_vector:
       - vx, vy: unit vector along principal axis
       - angle_deg: principal-axis angle in degrees
   - bbox_aligned: 4 corners of the principal-axis–aligned bounding box
                   as [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]

ROS2 PARAMETERS:
- input_image_topic:    Input binary image topic
                        (default: /camera/closure)
- output_image_topic:   Output annotated image topic
                        (default: /camera/objects)
- output_info_topic:    Output object information topic (JSON String)
                        (default: /camera/object_information)
- min_contour_area:     Minimum contour area to process (pixels²)
                        (default: 100.0)
- box_color_r:          Red channel for box/text color (0–255)
                        (default: 0)
- box_color_g:          Green channel for box/text color (0–255)
                        (default: 255)
- box_color_b:          Blue channel for box/text color (0–255)
                        (default: 0)
- box_thickness:        Thickness of box lines (default: 2)
- text_scale:           Font scale for text (default: 0.5)
- show_center_dot:      Draw a dot at the object center (default: True)
- center_dot_radius:    Radius of center dot in pixels (default: 5)

EXAMPLE RUNS:

# Minimal: just change input and min area
ros2 run camera_pkg min_rect_area_node --ros-args \
    -p input_image_topic:=/camera/closure \
    -p min_contour_area:=500.0

# Full parameter example
ros2 run camera_pkg min_rect_area_node --ros-args \
    -p input_image_topic:=/camera/closure \
    -p output_image_topic:=/camera/objects \
    -p output_info_topic:=/camera/object_information \
    -p min_contour_area:=500.0 \
    -p box_color_r:=0 \
    -p box_color_g:=255 \
    -p box_color_b:=0 \
    -p box_thickness:=3 \
    -p text_scale:=0.6 \
    -p show_center_dot:=true \
    -p center_dot_radius:=7
"""

import cv2
import numpy as np
import json

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from std_msgs.msg import String  
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class MinRectAreaNode(Node):
    def __init__(self):
        super().__init__("min_rect_area_node")

        # Parameters
        self.declare_parameter(
            "input_image_topic",
            "/camera/closure",
            ParameterDescriptor(description="Input binary image topic")
        )
        self.declare_parameter(
            "output_image_topic",
            "/camera/objects",
            ParameterDescriptor(description="Output annotated image topic")
        )
        self.declare_parameter(
            "min_contour_area",
            100.0,
            ParameterDescriptor(description="Minimum contour area (pixels²)")
        )
        self.declare_parameter(
            "output_info_topic",
            "/camera/object_information",
            ParameterDescriptor(description="Output object information topic (JSON)")
        )
        self.declare_parameter("box_color_r", 0)
        self.declare_parameter("box_color_g", 255)
        self.declare_parameter("box_color_b", 0)
        self.declare_parameter("box_thickness", 2)
        self.declare_parameter("text_scale", 0.5)
        self.declare_parameter("show_center_dot", True)
        self.declare_parameter("center_dot_radius", 5)

        # Get parameter values
        input_topic = self.get_parameter("input_image_topic").value
        output_topic = self.get_parameter("output_image_topic").value
        output_info_topic = self.get_parameter("output_info_topic").value
        self.min_contour_area = self.get_parameter("min_contour_area").value
        
        self.box_color = (
            self.get_parameter("box_color_b").value,
            self.get_parameter("box_color_g").value,
            self.get_parameter("box_color_r").value
        )
        self.box_thickness = self.get_parameter("box_thickness").value
        self.text_scale = self.get_parameter("text_scale").value
        self.show_center_dot = self.get_parameter("show_center_dot").value
        self.center_dot_radius = self.get_parameter("center_dot_radius").value

        # Logging
        self.get_logger().info("=" * 60)
        self.get_logger().info("Minimum Area Rectangle Detection Node")
        self.get_logger().info(f"Input topic:         {input_topic}")
        self.get_logger().info(f"Output topic:        {output_topic}")
        self.get_logger().info(f"Min contour area:    {self.min_contour_area} px²")
        self.get_logger().info("=" * 60)

        self.bridge = CvBridge()

        # Subscriber and Publisher
        self.image_sub = self.create_subscription(
            Image, input_topic, self.image_callback, 10
        )
        self.image_pub = self.create_publisher(Image, output_topic, 10)

        self.info_pub = self.create_publisher(String,output_info_topic,10
        )

        self.get_logger().info(f"Input topic:         {input_topic}")
        self.get_logger().info(f"Output image topic:  {output_topic}")
        self.get_logger().info(f"Output info topic:   {output_info_topic}")

    def image_callback(self, msg: Image):
        try:
            # ---------------------------------------------------
            # 1. Convert ROS Image -> OpenCV (binary image)
            # ---------------------------------------------------
            binary_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")

            # Create a 3‑channel BGR image so we can draw colored boxes and text.
            output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

            # Collect info for all detected objects to publish as JSON
            objects_info = []

            # ---------------------------------------------------
            # 2. Find contours
            # ---------------------------------------------------
            contours, _ = cv2.findContours(
                binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            object_count = 0

            # ---------------------------------------------------
            # 3. Process each contour
            # ---------------------------------------------------
            for contour in contours:
                # 3.1 Filter by area
                area = cv2.contourArea(contour)
                if area < self.min_contour_area:
                    continue

                object_count += 1

                # ---------------------------------------------------
                # 3.2 Minimum area rectangle (for bounding box)
                # ---------------------------------------------------
                rect = cv2.minAreaRect(contour)
                (center_x, center_y), (width, height), angle_raw = rect

                box_points = cv2.boxPoints(rect)
                box_points = np.int0(box_points)
                cv2.drawContours(
                    output_image,
                    [box_points],
                    0,
                    self.box_color,
                    self.box_thickness
                )

                # ---------------------------------------------------
                # 3.3 Normalize minAreaRect angle to long side
                # ---------------------------------------------------
                angle_box_deg = angle_raw
                if width < height:
                    angle_box_deg = angle_raw + 90.0

                # ---------------------------------------------------
                # 3.4 Orientation from image moments (principal axis)
                # ---------------------------------------------------
                M = cv2.moments(contour)

                # defaults in case moments fail
                angle_moments_deg = 0.0
                center_x_draw = center_x
                center_y_draw = center_y

                if M["m00"] != 0:
                    cx_m = M["m10"] / M["m00"]
                    cy_m = M["m01"] / M["m00"]

                    mu20 = M["mu20"] / M["m00"]
                    mu02 = M["mu02"] / M["m00"]
                    mu11 = M["mu11"] / M["m00"]

                    theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
                    angle_moments_deg = np.degrees(theta)

                    long_side = max(width, height)
                    axis_length = (long_side / 2.0) + 10

                    x2 = int(cx_m + axis_length * np.cos(theta))
                    y2 = int(cy_m + axis_length * np.sin(theta))
                    x1 = int(cx_m - axis_length * np.cos(theta))
                    y1 = int(cy_m - axis_length * np.sin(theta))

                    cv2.line(
                        output_image,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),  # red
                        2
                    )

                    center_x_draw = cx_m
                    center_y_draw = cy_m

                # ---------------------------------------------------
                # 3.5 Draw center dot (optional)
                # ---------------------------------------------------
                if self.show_center_dot:
                    cv2.circle(
                        output_image,
                        (int(center_x_draw), int(center_y_draw)),
                        self.center_dot_radius,
                        self.box_color,
                        -1
                    )

                # ---------------------------------------------------
                # 3.6 Principal axis bounding box (PCA)
                # ---------------------------------------------------
                pts = contour.reshape(-1, 2).astype(np.float32)

                mean, eigenvectors, eigenvalues = cv2.PCACompute2(pts, mean=None)

                v0 = eigenvectors[0]   # principal axis
                v1 = eigenvectors[1]   # perpendicular axis

                v0 = v0 / np.linalg.norm(v0)
                v1 = v1 / np.linalg.norm(v1)

                pts_centered = pts - mean
                proj_v0 = pts_centered @ v0
                proj_v1 = pts_centered @ v1

                min_v0, max_v0 = np.min(proj_v0), np.max(proj_v0)
                min_v1, max_v1 = np.min(proj_v1), np.max(proj_v1)

                corners_local = np.array([
                    [min_v0, min_v1],
                    [max_v0, min_v1],
                    [max_v0, max_v1],
                    [min_v0, max_v1]
                ], dtype=np.float32)

                corners_img = []
                for u, v in corners_local:
                    p = mean[0] + u * v0 + v * v1
                    corners_img.append(p)

                corners_img = np.array(corners_img, dtype=np.int32).reshape(-1, 1, 2)

                cv2.polylines(
                    output_image,
                    [corners_img],
                    isClosed=True,
                    color=(255, 0, 0),  # blue
                    thickness=self.box_thickness
                )

                # ---------------------------------------------------
                # 3.7 Text annotations
                # ---------------------------------------------------
                text_x = int(center_x_draw) + 35
                text_y = int(center_y_draw) - 80
                line_spacing = 20

                texts = [
                    f"Object #{object_count}",
                    f"Center: ({center_x_draw:.1f}, {center_y_draw:.1f})",
                    f"Size (px): {width:.1f} x {height:.1f}",
                    f"Moments angle: {angle_moments_deg:.1f} deg",
                ]

                for i, text in enumerate(texts):
                    cv2.putText(
                        output_image,
                        text,
                        (text_x, text_y + i * line_spacing),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.text_scale,
                        self.box_color,
                        1,
                        cv2.LINE_AA
                    )

                # ---------------------------------------------------
                # 3.8 Collect information for JSON publishing
                # ---------------------------------------------------
                moment_vector = {
                    "vx": float(v0[0]),
                    "vy": float(v0[1]),
                    "angle_deg": float(angle_moments_deg),
                }

                bbox_aligned = []
                for p in corners_img.reshape(-1, 2):
                    bbox_aligned.append([int(p[0]), int(p[1])])

                obj_info = {
                    "object_id": int(object_count),
                    "center_px": [float(center_x_draw), float(center_y_draw)],
                    "moment_vector": moment_vector,
                    "bbox_aligned": bbox_aligned,
                }

                objects_info.append(obj_info)

            # ---------------------------------------------------
            # 4. Log and publish image
            # ---------------------------------------------------
            self.get_logger().info(
                f"Detected {object_count} objects",
                throttle_duration_sec=1.0
            )

            out_msg = self.bridge.cv2_to_imgmsg(output_image, encoding="bgr8")
            out_msg.header = msg.header
            self.image_pub.publish(out_msg)

            # ---------------------------------------------------
            # 5. Compile JSON with all relevant information
            # ---------------------------------------------------
            info_dict = {
                "header": {
                    "stamp_sec": int(msg.header.stamp.sec),
                    "stamp_nanosec": int(msg.header.stamp.nanosec),
                    "frame_id": msg.header.frame_id,
                },
                "objects": objects_info
            }

            info_msg = String()
            info_msg.data = json.dumps(info_dict)
            self.info_pub.publish(info_msg)

        except Exception as e:
            self.get_logger().error(f"Error processing frame: {e}")


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