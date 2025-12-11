#!/usr/bin/env python3
"""
Physical Features Extraction ROS2 Node

This node subscribes to:
  - /camera/closure (sensor_msgs/Image): binary image for overlay
  - /camera/object_information (std_msgs/String): JSON with detected objects

It synchronizes these two topics (using ApproximateTimeSynchronizer with
allow_headerless=True), parses the JSON, and draws:
  - Center dot for each object
  - Principal-axis–aligned bounding box (from bbox_aligned)

Publishes an annotated image on /camera/physical_features.

ROS2 PARAMETERS:
- input_image_topic:    Input binary image topic
                        (default: /camera/closure)
- input_info_topic:     Input object information topic (JSON String)
                        (default: /camera/object_information)
- output_image_topic:   Output annotated image topic
                        (default: /camera/physical_features)
- pixel_to_mm_ratio:    Conversion ratio from pixels to millimeters
                        (default: 1.0, meaning 1 px = 1 mm; adjust as needed)
- center_dot_radius:    Radius of center dot in pixels (default: 5)
- bbox_color_r:         Red channel for bbox color (0-255) (default: 255)
- bbox_color_g:         Green channel for bbox color (0-255) (default: 0)
- bbox_color_b:         Blue channel for bbox color (0-255) (default: 0)
- bbox_thickness:       Thickness of bbox lines (default: 2)

EXAMPLE RUN:

ros2 run camera_pkg physical_features_extraction_node --ros-args \
    -p input_image_topic:=/camera/closure \
    -p input_info_topic:=/camera/object_information \
    -p output_image_topic:=/camera/physical_features \
    -p pixel_to_mm_ratio:=0.1 \
    -p center_dot_radius:=5 \
    -p bbox_color_r:=255 \
    -p bbox_color_g:=0 \
    -p bbox_color_b:=0 \
    -p bbox_thickness:=2
"""

import json
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

# For message synchronization
from message_filters import Subscriber, ApproximateTimeSynchronizer


class PhysicalFeaturesNode(Node):
    def __init__(self):
        super().__init__("physical_features_extraction")

        # ---------------------------------------------------
        # Parameters
        # ---------------------------------------------------
        self.declare_parameter(
            "input_image_topic",
            "/camera/closure",
            ParameterDescriptor(description="Input binary image topic")
        )
        self.declare_parameter(
            "input_info_topic",
            "/camera/object_information",
            ParameterDescriptor(description="Input object information topic (JSON)")
        )
        self.declare_parameter(
            "output_image_topic",
            "/camera/physical_features",
            ParameterDescriptor(description="Output annotated image topic")
        )
        self.declare_parameter(
            "pixel_to_mm_ratio",
            0.1,
            ParameterDescriptor(description="Pixel to mm conversion ratio (px/mm)")
        )
        self.declare_parameter("center_dot_radius", 5)
        self.declare_parameter("bbox_color_r", 255)
        self.declare_parameter("bbox_color_g", 0)
        self.declare_parameter("bbox_color_b", 0)
        self.declare_parameter("bbox_thickness", 2)

        # Get parameter values
        input_image_topic = self.get_parameter("input_image_topic").value
        input_info_topic = self.get_parameter("input_info_topic").value
        output_image_topic = self.get_parameter("output_image_topic").value
        self.pixel_to_mm_ratio = self.get_parameter("pixel_to_mm_ratio").value
        self.center_dot_radius = self.get_parameter("center_dot_radius").value

        self.bbox_color = (
            self.get_parameter("bbox_color_b").value,
            self.get_parameter("bbox_color_g").value,
            self.get_parameter("bbox_color_r").value
        )
        self.bbox_thickness = self.get_parameter("bbox_thickness").value

        # Logging
        self.get_logger().info("=" * 60)
        self.get_logger().info("Physical Features Extraction Node")
        self.get_logger().info(f"Input image topic:   {input_image_topic}")
        self.get_logger().info(f"Input info topic:    {input_info_topic}")
        self.get_logger().info(f"Output image topic:  {output_image_topic}")
        self.get_logger().info(f"Pixel to mm ratio:   {self.pixel_to_mm_ratio}")
        self.get_logger().info("=" * 60)

        self.bridge = CvBridge()

        # ---------------------------------------------------
        # Synchronized subscribers using message_filters
        # ---------------------------------------------------
        # We use ApproximateTimeSynchronizer to match messages
        # from the two topics based on their timestamps.
        self.image_sub = Subscriber(self, Image, input_image_topic)
        self.info_sub = Subscriber(self, String, input_info_topic)

        # Synchronizer: queue_size=10, slop=0.1s (100ms tolerance)
        # allow_headerless=True is required because std_msgs/String has no header.
        self.sync = ApproximateTimeSynchronizer(
            [self.image_sub, self.info_sub],
            queue_size=10,
            slop=0.1,
            allow_headerless=True
        )
        self.sync.registerCallback(self.synchronized_callback)

        # Publisher for annotated image
        self.image_pub = self.create_publisher(Image, output_image_topic, 10)

        self.get_logger().info("Node ready. Waiting for synchronized messages...")

    def synchronized_callback(self, image_msg: Image, info_msg: String):
        """
        Called when image and info messages are synchronized.

        Args:
            image_msg: Binary image from /camera/closure
            info_msg: JSON string from /camera/object_information
        """
        try:
            # ---------------------------------------------------
            # 1. Convert binary image to BGR for visualization
            # ---------------------------------------------------
            binary_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="mono8")
            output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

            # ---------------------------------------------------
            # 2. Parse JSON from info_msg
            # ---------------------------------------------------
            info_data = json.loads(info_msg.data)
            objects = info_data.get("objects", [])

            self.get_logger().info(
                f"Processing {len(objects)} objects",
                throttle_duration_sec=1.0
            )

            # ---------------------------------------------------
            # 3. Process each object
            # ---------------------------------------------------
            for obj in objects:
                # Basic fields from JSON
                object_id = obj.get("object_id", 0)
                name = obj.get("name", f"Object #{object_id}")
                center_px = obj.get("center_px", [0.0, 0.0])
                bbox_aligned = obj.get("bbox_aligned", [])
                moment_vector = obj.get("moment_vector", {})

                cx, cy = int(center_px[0]), int(center_px[1])

                # Draw center dot
                cv2.circle(
                    output_image,
                    (cx, cy),
                    self.center_dot_radius,
                    self.bbox_color,
                    -1  # filled
                )

                # Draw principal-axis–aligned bounding box
                if len(bbox_aligned) == 4:
                    bbox_np = np.array(bbox_aligned, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(
                        output_image,
                        [bbox_np],
                        isClosed=True,
                        color=self.bbox_color,
                        thickness=self.bbox_thickness
                    )

                # ---------------------------------------------------
                # 3.1 Compute size in px and convert to mm
                # ---------------------------------------------------
                width_px = height_px = None

                if len(bbox_aligned) == 4:
                    pts = np.array(bbox_aligned, dtype=np.float32)

                    # Edge vectors
                    v_edge0 = pts[1] - pts[0]
                    v_edge1 = pts[2] - pts[1]

                    # Lengths in pixels
                    len0 = np.linalg.norm(v_edge0)
                    len1 = np.linalg.norm(v_edge1)

                    width_px = max(len0, len1)
                    height_px = min(len0, len1)

                # Convert to mm
                size_text = ""
                if width_px is not None and height_px is not None:
                    width_mm = width_px * self.pixel_to_mm_ratio
                    height_mm = height_px * self.pixel_to_mm_ratio
                    size_text = f"{width_mm:.2f} x {height_mm:.2f} mm"

                # ---------------------------------------------------
                # 3.2 Prepare text list
                # ---------------------------------------------------
                texts = [name]
                if size_text:
                    texts.append(f"Size: {size_text}")

                # ---------------------------------------------------
                # 4. Extract PCA info from JSON (already computed upstream)
                # ---------------------------------------------------
                if len(bbox_aligned) == 4 and moment_vector:
                    # Principal axis direction from moment_vector
                    vx = moment_vector.get("vx", 1.0)
                    vy = moment_vector.get("vy", 0.0)
                    v0 = np.array([vx, vy], dtype=np.float32)
                    v0 = v0 / (np.linalg.norm(v0) + 1e-9)  # normalize

                    # Perpendicular direction
                    v1 = np.array([-v0[1], v0[0]], dtype=np.float32)

                    # Mean (center from JSON)
                    mean = np.array([[cx, cy]], dtype=np.float32)

                    # Project bbox corners onto v0 and v1 to get extents
                    pts = np.array(bbox_aligned, dtype=np.float32)
                    proj_v0 = np.dot(pts - mean[0], v0)
                    proj_v1 = np.dot(pts - mean[0], v1)

                    min_v0 = np.min(proj_v0)
                    max_v0 = np.max(proj_v0)
                    min_v1 = np.min(proj_v1)
                    max_v1 = np.max(proj_v1)

                    # ---------------------------------------------------
                    # 5. Scan perpendicular slices for head-body separation
                    # ---------------------------------------------------
                    # Use more slices for better resolution (2-3 per pixel)
                    length_px = max_v0 - min_v0
                    num_slices = int(length_px * 2)  # 2 slices per pixel
                    if num_slices < 10:
                        num_slices = 10

                    self.get_logger().info(
                        f"Object {object_id}: Scanning {num_slices} slices along {length_px:.1f}px",
                        throttle_duration_sec=2.0
                    )

                    slice_positions = np.linspace(min_v0, max_v0, num_slices)
                    slice_widths = []

                    # ---------------------------------------------------
                    # 5.1 Record all slice widths
                    # ---------------------------------------------------
                    for i, s in enumerate(slice_positions):
                        # Position along principal axis
                        center_slice = mean[0] + s * v0

                        # Endpoints perpendicular
                        p1 = center_slice + min_v1 * v1
                        p2 = center_slice + max_v1 * v1

                        p1_int = (int(p1[0]), int(p1[1]))
                        p2_int = (int(p2[0]), int(p2[1]))

                        # Create line mask
                        line_mask = np.zeros_like(binary_image)
                        cv2.line(line_mask, p1_int, p2_int, 255, 1)

                        # Count white pixels along this line
                        overlap = cv2.bitwise_and(binary_image, line_mask)
                        width_px_slice = np.count_nonzero(overlap)

                        slice_widths.append(width_px_slice)

                    slice_widths = np.array(slice_widths, dtype=np.float32)

                    # ---------------------------------------------------
                    # 5.2 Detect head-body separation with improved algorithm
                    # ---------------------------------------------------
                    n = len(slice_widths)
                    center_idx = n // 2

                    # Apply strong smoothing to remove noise
                    from scipy.ndimage import uniform_filter1d
                    window_size = max(5, n // 20)  # Larger smoothing window
                    slice_widths_smooth = uniform_filter1d(
                        slice_widths,
                        size=window_size,
                        mode='nearest'
                    )

                    # Compute gradient
                    gradient = np.abs(np.diff(slice_widths_smooth))

                    # Find regions with sustained width difference (not just spikes)
                    # Strategy: Look for the transition where width changes significantly
                    # and stays different for multiple slices
                    
                    # Compute mean width in first 1/3 and last 1/3
                    third = n // 3
                    if third < 5:
                        third = 5
                    
                    head_region_width = np.mean(slice_widths_smooth[:third])
                    body_region_width = np.mean(slice_widths_smooth[-third:])
                    
                    self.get_logger().info(
                        f"Object {object_id}: Head avg width={head_region_width:.1f}px, "
                        f"Body avg width={body_region_width:.1f}px",
                        throttle_duration_sec=2.0
                    )

                    # Find the slice closest to the midpoint between head and body widths
                    target_width = (head_region_width + body_region_width) / 2.0
                    
                    # Search in the middle 60% of the object (avoid edges)
                    search_start = int(n * 0.2)
                    search_end = int(n * 0.8)
                    
                    # Find where smoothed width crosses the target
                    separation_idx = center_idx  # fallback
                    min_diff = float('inf')
                    
                    for i in range(search_start, search_end):
                        diff = abs(slice_widths_smooth[i] - target_width)
                        if diff < min_diff:
                            min_diff = diff
                            separation_idx = i

                    self.get_logger().info(
                        f"Object {object_id}: Separation at slice {separation_idx}/{n} "
                        f"(width={slice_widths_smooth[separation_idx]:.1f}px)",
                        throttle_duration_sec=2.0
                    )

                    separation_pos = slice_positions[separation_idx]

                    # Separation line endpoints
                    separation_center = mean[0] + separation_pos * v0
                    sep_p1 = separation_center + min_v1 * v1
                    sep_p2 = separation_center + max_v1 * v1

                    sep_p1_int = (int(sep_p1[0]), int(sep_p1[1]))
                    sep_p2_int = (int(sep_p2[0]), int(sep_p2[1]))

                    # Draw separation line in RED
                    cv2.line(
                        output_image,
                        sep_p1_int,
                        sep_p2_int,
                        (0, 0, 255),  # red
                        2
                    )

                    # ---------------------------------------------------
                    # 5.3 Compute body diameter and length
                    # ---------------------------------------------------
                    # Body is the region AFTER the separation line
                    body_slice_widths = slice_widths_smooth[separation_idx + 1:]
                    if len(body_slice_widths) > 0:
                        body_diameter_px = np.mean(body_slice_widths)
                        body_diameter_mm = body_diameter_px * self.pixel_to_mm_ratio
                    else:
                        body_diameter_mm = 0.0

                    body_length_px = max_v0 - separation_pos
                    body_length_mm = body_length_px * self.pixel_to_mm_ratio

                    # ---------------------------------------------------
                    # 5.3.1 Calculate pick-up point (center of body region)
                    # ---------------------------------------------------
                    # Pick-up point is at the center of the body (midpoint between separation line and tail end)
                    pick_up_pos = (separation_pos + max_v0) / 2.0
                    pick_up_world = mean[0] + pick_up_pos * v0
                    pick_up_x = int(pick_up_world[0])
                    pick_up_y = int(pick_up_world[1])

                    # Draw pick-up point as GREEN dot
                    cv2.circle(
                        output_image,
                        (pick_up_x, pick_up_y),
                        5,  # slightly larger radius
                        (0, 255, 0),  # green
                        -1  # filled
                    )

                    # Optional: Draw a small cross for better visibility
                    cross_size = 8
                    cv2.line(
                        output_image,
                        (pick_up_x - cross_size, pick_up_y),
                        (pick_up_x + cross_size, pick_up_y),
                        (0, 255, 0),
                        2
                    )
                    cv2.line(
                        output_image,
                        (pick_up_x, pick_up_y - cross_size),
                        (pick_up_x, pick_up_y + cross_size),
                        (0, 255, 0),
                        2
                    )

                    # ---------------------------------------------------
                    # 5.4 Add body info to text
                    # ---------------------------------------------------
                    texts.append(f"Body dia: {body_diameter_mm:.2f} mm")
                    texts.append(f"Body len: {body_length_mm:.2f} mm")
                    texts.append(f"Slices: {n}, Sep: {separation_idx}")

                # ---------------------------------------------------
                # 6. Draw all text annotations
                # ---------------------------------------------------
                text_x = cx + 10
                text_y = cy - 10
                line_spacing = 20

                for i, text in enumerate(texts):
                    cv2.putText(
                        output_image,
                        text,
                        (text_x, text_y + i * line_spacing),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        self.bbox_color,
                        1,
                        cv2.LINE_AA
                    )

            # ---------------------------------------------------
            # 7. Publish annotated image
            # ---------------------------------------------------
            out_msg = self.bridge.cv2_to_imgmsg(output_image, encoding="bgr8")
            out_msg.header = image_msg.header
            self.image_pub.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f"Error in synchronized callback: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = PhysicalFeaturesNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()