#!/usr/bin/env python3
"""
Physical Features Extraction ROS2 Node (OPTIMIZED FOR SLOW STREAMS)

This node subscribes to:
  - /camera/closure (sensor_msgs/Image): binary image for overlay
  - /camera/object_information (std_msgs/String): JSON with detected objects

It synchronizes these two topics (using ApproximateTimeSynchronizer with
allow_headerless=True), parses the JSON, and draws:
  - Center dot for each object
  - Principal-axis–aligned bounding box (from bbox_aligned)
  - Head-body separation line
  - Pick-up point (green dot in body center)
  - Screw type classification (WOOD/METAL) based on jagginess

Publishes:
  - /camera/physical_features (sensor_msgs/Image): annotated image
  - /camera/object_physical_features (std_msgs/String): JSON with all computed features

OPTIMIZATIONS:
  - Processes ONLY centermost object by default (massive speedup)
  - Frame skipping for slow camera streams
  - Capped number of slices (max 100 instead of 150)
  - ROI extraction per object (avoids full-image operations)
  - Rotation-based slice scanning (column sums instead of line masks)
  - Reduced logging overhead
  - Lightweight 1D smoothing (NumPy convolution instead of scipy)
  - Smaller ROI padding and reduced queue size

ROS2 PARAMETERS:
- input_image_topic:    Input binary image topic
                        (default: /camera/closure)
- input_info_topic:     Input object information topic (JSON String)
                        (default: /camera/object_information)
- output_image_topic:   Output annotated image topic
                        (default: /camera/physical_features)
- output_json_topic:    Output JSON features topic
                        (default: /camera/object_physical_features)
- pixel_to_mm_ratio:    Conversion ratio from pixels to millimeters
                        (default: 0.1)
- center_dot_radius:    Radius of center dot in pixels (default: 5)
- bbox_color_r:         Red channel for bbox color (0-255) (default: 255)
- bbox_color_g:         Green channel for bbox color (0-255) (default: 0)
- bbox_color_b:         Blue channel for bbox color (0-255) (default: 0)
- bbox_thickness:       Thickness of bbox lines (default: 2)
- jagginess_threshold:  Normalized jagginess threshold for wood detection
                        (default: 0.05)
- max_slices:           Maximum number of slices per object (default: 100)
- min_slices:           Minimum number of slices per object (default: 40)
- roi_padding:          Padding around object ROI in pixels (default: 15)
- frame_skip:           Process every Nth frame (1=all, 2=half, 3=third) (default: 1)
- debug_logging:        Enable detailed per-object logging (default: False)
- process_closest_only: Process only the object closest to camera center (default: True)

EXAMPLE RUN:

ros2 run camera_pkg physical_features_extraction_node --ros-args \
    -p input_image_topic:=/camera/closure \
    -p input_info_topic:=/camera/object_information \
    -p output_image_topic:=/camera/physical_features \
    -p output_json_topic:=/camera/object_physical_features \
    -p pixel_to_mm_ratio:=0.1 \
    -p max_slices:=100 \
    -p min_slices:=40 \
    -p roi_padding:=15 \
    -p frame_skip:=2 \
    -p debug_logging:=false \
    -p process_closest_only:=true
"""

import json
import numpy as np
import numpy.linalg as LA
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
            "output_json_topic",
            "/camera/object_physical_features",
            ParameterDescriptor(description="Output JSON features topic")
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
        self.declare_parameter(
            "jagginess_threshold",
            0.05,
            ParameterDescriptor(description="Normalized jagginess threshold for wood detection")
        )
        self.declare_parameter(
            "max_slices",
            100,
            ParameterDescriptor(description="Maximum number of slices per object (reduced for speed)")
        )
        self.declare_parameter(
            "min_slices",
            40,
            ParameterDescriptor(description="Minimum number of slices per object (reduced for speed)")
        )
        self.declare_parameter(
            "roi_padding",
            15,
            ParameterDescriptor(description="Padding around object ROI in pixels (reduced for speed)")
        )
        self.declare_parameter(
            "frame_skip",
            1,
            ParameterDescriptor(description="Process every Nth frame (1=all, 2=every other, 3=every third)")
        )
        self.declare_parameter(
            "debug_logging",
            False,
            ParameterDescriptor(description="Enable detailed per-object logging")
        )
        self.declare_parameter(
            "process_closest_only",
            True,
            ParameterDescriptor(description="Process only the object closest to camera center")
        )

        # Get parameter values
        input_image_topic = self.get_parameter("input_image_topic").value
        input_info_topic = self.get_parameter("input_info_topic").value
        output_image_topic = self.get_parameter("output_image_topic").value
        output_json_topic = self.get_parameter("output_json_topic").value
        self.pixel_to_mm_ratio = self.get_parameter("pixel_to_mm_ratio").value
        self.center_dot_radius = self.get_parameter("center_dot_radius").value
        self.jagginess_threshold = self.get_parameter("jagginess_threshold").value
        self.max_slices = self.get_parameter("max_slices").value
        self.min_slices = self.get_parameter("min_slices").value
        self.roi_padding = self.get_parameter("roi_padding").value
        self.frame_skip = self.get_parameter("frame_skip").value
        self.debug_logging = self.get_parameter("debug_logging").value
        self.process_closest_only = self.get_parameter("process_closest_only").value

        self.bbox_color = (
            self.get_parameter("bbox_color_b").value,
            self.get_parameter("bbox_color_g").value,
            self.get_parameter("bbox_color_r").value
        )
        self.bbox_thickness = self.get_parameter("bbox_thickness").value

        # Frame counter for skipping
        self.frame_counter = 0

        # Logging
        self.get_logger().info("=" * 70)
        self.get_logger().info("     PHYSICAL FEATURES EXTRACTION NODE (OPTIMIZED)")
        self.get_logger().info("=" * 70)
        self.get_logger().info(f"  Input image topic:   {input_image_topic}")
        self.get_logger().info(f"  Input info topic:    {input_info_topic}")
        self.get_logger().info(f"  Output image topic:  {output_image_topic}")
        self.get_logger().info(f"  Output JSON topic:   {output_json_topic}")
        self.get_logger().info(f"  Pixel to mm ratio:   {self.pixel_to_mm_ratio}")
        self.get_logger().info(f"  Max slices:          {self.max_slices}")
        self.get_logger().info(f"  Min slices:          {self.min_slices}")
        self.get_logger().info(f"  ROI padding:         {self.roi_padding}px")
        self.get_logger().info(f"  Frame skip:          {self.frame_skip} (process every {self.frame_skip} frame(s))")
        self.get_logger().info(f"  Debug logging:       {self.debug_logging}")
        self.get_logger().info(f"  Process closest only: {self.process_closest_only}")
        self.get_logger().info("-" * 70)
        self.get_logger().info("  CLASSIFICATION THRESHOLD:")
        self.get_logger().info(f"    Jagginess threshold: {self.jagginess_threshold}")
        self.get_logger().info("=" * 70)

        self.bridge = CvBridge()

        # ---------------------------------------------------
        # Synchronized subscribers using message_filters
        # ---------------------------------------------------
        self.image_sub = Subscriber(self, Image, input_image_topic)
        self.info_sub = Subscriber(self, String, input_info_topic)

        # Optimized for slow streams: reduced queue, increased slop
        self.sync = ApproximateTimeSynchronizer(
            [self.image_sub, self.info_sub],
            queue_size=5,  # Reduced from 10
            slop=0.2,      # Increased from 0.1 for better sync
            allow_headerless=True
        )
        self.sync.registerCallback(self.synchronized_callback)

        # ---------------------------------------------------
        # Publishers
        # ---------------------------------------------------
        self.image_pub = self.create_publisher(Image, output_image_topic, 10)
        self.json_pub = self.create_publisher(String, output_json_topic, 10)

        self.get_logger().info("✓ Node ready. Waiting for synchronized messages...\n")

    # =======================================================
    # OPTIMIZED: Lightweight 1D smoothing (replaces scipy)
    # =======================================================
    @staticmethod
    def smooth_1d(data, window_size):
        """
        Simple moving average using NumPy convolution.
        Much faster than scipy for small arrays.
        """
        if window_size < 3:
            return data
        kernel = np.ones(window_size) / window_size
        # mode='same' keeps output same length as input
        smoothed = np.convolve(data, kernel, mode='same')

        # Fix edges
        half_window = window_size // 2
        smoothed[:half_window] = data[:half_window]
        smoothed[-half_window:] = data[-half_window:]

        return smoothed

    # =======================================================
    # OPTIMIZED: Extract and rotate ROI for fast slice scanning
    # =======================================================
    def extract_rotated_roi(self, binary_image, bbox_aligned, mean, v0):
        """
        Extract a rotated ROI around the object aligned with principal axis.

        Returns:
            rotated_roi: Binary ROI with object aligned horizontally
            roi_info: Dict with transformation info for mapping back to original coords
        """
        # Get bounding box of the aligned bbox
        pts = np.array(bbox_aligned, dtype=np.float32)
        x_coords = pts[:, 0]
        y_coords = pts[:, 1]

        min_x = int(np.floor(np.min(x_coords))) - self.roi_padding
        max_x = int(np.ceil(np.max(x_coords))) + self.roi_padding
        min_y = int(np.floor(np.min(y_coords))) - self.roi_padding
        max_y = int(np.ceil(np.max(y_coords))) + self.roi_padding

        # Clamp to image bounds
        h, w = binary_image.shape[:2]
        min_x = max(0, min_x)
        max_x = min(w, max_x)
        min_y = max(0, min_y)
        max_y = min(h, max_y)

        roi_width = max_x - min_x
        roi_height = max_y - min_y

        if roi_width <= 0 or roi_height <= 0:
            return None, None

        # Extract ROI
        roi = binary_image[min_y:max_y, min_x:max_x]

        # Compute rotation angle to align v0 with x-axis
        angle_rad = np.arctan2(v0[1], v0[0])
        angle_deg = np.degrees(angle_rad)

        # Center of ROI in ROI coordinates
        roi_center = (roi_width / 2.0, roi_height / 2.0)

        # Rotation matrix
        rot_matrix = cv2.getRotationMatrix2D(roi_center, -angle_deg, 1.0)

        # Rotate ROI
        rotated_roi = cv2.warpAffine(
            roi,
            rot_matrix,
            (roi_width, roi_height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        # Store info for reverse mapping
        roi_info = {
            'offset': np.array([min_x, min_y], dtype=np.float32),
            'roi_center': np.array(roi_center, dtype=np.float32),
            'rot_matrix': rot_matrix,
            'angle_deg': angle_deg,
            'roi_width': roi_width,
            'roi_height': roi_height
        }

        return rotated_roi, roi_info

    # =======================================================
    # OPTIMIZED: Fast slice width computation using column sums
    # =======================================================
    def compute_slice_widths_fast(self, rotated_roi):
        """
        Compute width profile by summing columns in rotated ROI.
        This is MUCH faster than drawing line masks.

        Returns:
            slice_widths: 1D array of widths (one per column)
        """
        # Count non-zero pixels in each column
        slice_widths = np.count_nonzero(rotated_roi, axis=0).astype(np.float32)
        return slice_widths

    # =======================================================
    # OPTIMIZED: Fast jagginess estimation using edge tracking
    # =======================================================
    def estimate_jagginess_fast(self, rotated_roi, body_start_col, body_end_col):
        """
        Estimate jagginess by tracking top and bottom edges in the body region.

        Returns:
            jag_overall, jag_top, jag_bottom
        """
        if body_end_col <= body_start_col or body_end_col - body_start_col < 5:
            return 0.0, 0.0, 0.0

        # Extract body region columns
        body_region = rotated_roi[:, body_start_col:body_end_col + 1]

        top_edges = []
        bottom_edges = []
        valid_cols = []

        # For each column, find top and bottom edge
        for col_idx in range(body_region.shape[1]):
            col = body_region[:, col_idx]
            nonzero_rows = np.where(col > 0)[0]

            if len(nonzero_rows) > 0:
                top_edges.append(nonzero_rows[0])
                bottom_edges.append(nonzero_rows[-1])
                valid_cols.append(col_idx)

        if len(valid_cols) < 5:
            return 0.0, 0.0, 0.0

        valid_cols = np.array(valid_cols, dtype=np.float32)
        top_edges = np.array(top_edges, dtype=np.float32)
        bottom_edges = np.array(bottom_edges, dtype=np.float32)

        def compute_edge_jagginess(x, y):
            """Fit line and compute std of residuals"""
            if len(y) < 5:
                return 0.0
            A = np.vstack([x, np.ones_like(x)]).T
            coeffs = LA.lstsq(A, y, rcond=None)[0]
            y_fit = coeffs[0] * x + coeffs[1]
            residuals = y - y_fit

            # Smooth residuals
            smooth_window = max(3, len(residuals) // 15)
            residuals_smooth = self.smooth_1d(residuals, smooth_window)

            return float(np.std(residuals_smooth))

        jag_top = compute_edge_jagginess(valid_cols, top_edges)
        jag_bottom = compute_edge_jagginess(valid_cols, bottom_edges)
        jag_overall = max(jag_top, jag_bottom)

        return jag_overall, jag_top, jag_bottom

    # =======================================================
    # OPTIMIZED: Classify screw type
    # =======================================================
    def classify_screw_by_jagginess(self, jag_value, body_diameter_px):
        """
        Decide WOOD vs METAL from jagginess on the side contour.
        """
        if body_diameter_px <= 0:
            return "UNKNOWN", (128, 128, 128), 0.0

        jag_norm = jag_value / (body_diameter_px + 1e-9)
        is_wood = jag_norm > self.jagginess_threshold

        label = "WOOD" if is_wood else "METAL"
        color = (0, 255, 255) if is_wood else (255, 0, 255)  # yellow for wood, magenta for metal

        if self.debug_logging:
            self.get_logger().info(
                f"    Jagginess: raw={jag_value:.3f}px, "
                f"dia={body_diameter_px:.1f}px, norm={jag_norm:.4f} → {label}"
            )

        return label, color, jag_norm

    # =======================================================
    # OPTIMIZED: Map point from rotated ROI back to original image
    # =======================================================
    def map_roi_to_original(self, col, row, roi_info):
        """
        Map a point from rotated ROI coordinates back to original image coordinates.
        """
        # Point in ROI coordinates (before rotation)
        roi_center = roi_info['roi_center']
        angle_deg = roi_info['angle_deg']

        # Reverse rotation
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Translate to center, rotate, translate back
        dx = col - roi_center[0]
        dy = row - roi_center[1]

        x_rot = dx * cos_a - dy * sin_a + roi_center[0]
        y_rot = dx * sin_a + dy * cos_a + roi_center[1]

        # Add ROI offset
        x_orig = x_rot + roi_info['offset'][0]
        y_orig = y_rot + roi_info['offset'][1]

        return int(x_orig), int(y_orig)

    # =======================================================
    # Main synchronized callback
    # =======================================================
    def synchronized_callback(self, image_msg: Image, info_msg: String):
        """
        Called when image and info messages are synchronized.
        OPTIMIZED: Frame skipping and centermost object processing.
        """

        # ---------------------------------------------------
        # OPTIMIZATION: Frame skipping for slow streams
        # ---------------------------------------------------
        self.frame_counter += 1
        if self.frame_counter % self.frame_skip != 0:
            return

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

            # ---------------------------------------------------
            # 2.1 Prepare output JSON structure
            # ---------------------------------------------------
            output_json = {
                "timestamp": image_msg.header.stamp.sec + image_msg.header.stamp.nanosec * 1e-9,
                "frame_id": image_msg.header.frame_id,
                "pixel_to_mm_ratio": self.pixel_to_mm_ratio,
                "objects": []
            }

            if len(objects) > 0 and self.debug_logging:
                self.get_logger().info(f"\n{'='*70}\n  PROCESSING {len(objects)} OBJECT(S)\n{'='*70}")

            # ---------------------------------------------------
            # 2.2 OPTIMIZATION: Filter to closest object if enabled
            # ---------------------------------------------------
            if self.process_closest_only and len(objects) > 0:
                # Get image center
                h, w = binary_image.shape[:2]
                img_center = np.array([w / 2.0, h / 2.0])

                # Find object closest to image center
                min_dist = float('inf')
                closest_obj = None

                for obj in objects:
                    center_px = obj.get("center_px", [0.0, 0.0])
                    obj_center = np.array(center_px)
                    dist = np.linalg.norm(obj_center - img_center)

                    if dist < min_dist:
                        min_dist = dist
                        closest_obj = obj

                # Process only the closest object
                objects = [closest_obj] if closest_obj is not None else []

                if self.debug_logging:
                    self.get_logger().info(
                        f"  ✓ Filtered to closest object (distance: {min_dist:.1f}px from center)"
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

                # Initialize object features dictionary
                obj_features = {
                    "object_id": object_id,
                    "name": name,
                    "center_px": center_px,
                    "bbox_aligned": bbox_aligned,
                    "moment_vector": moment_vector
                }

                if self.debug_logging:
                    self.get_logger().info(f"\n  ┌─ OBJECT {object_id}: {name} ─────────────────────────")

                # Draw center dot
                cv2.circle(output_image, (cx, cy), self.center_dot_radius, self.bbox_color, -1)

                # Draw principal-axis–aligned bounding box
                if len(bbox_aligned) == 4:
                    bbox_np = np.array(bbox_aligned, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(output_image, [bbox_np], isClosed=True, color=(0, 255, 0), thickness=self.bbox_thickness)

                # ---------------------------------------------------
                # 3.1 Compute size in px and convert to mm
                # ---------------------------------------------------
                width_px = height_px = None

                if len(bbox_aligned) == 4:
                    pts = np.array(bbox_aligned, dtype=np.float32)
                    v_edge0 = pts[1] - pts[0]
                    v_edge1 = pts[2] - pts[1]
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

                    obj_features["bbox_width_px"] = float(width_px)
                    obj_features["bbox_height_px"] = float(height_px)
                    obj_features["bbox_width_mm"] = float(width_mm)
                    obj_features["bbox_height_mm"] = float(height_mm)

                # ---------------------------------------------------
                # 3.2 Prepare text list
                # ---------------------------------------------------
                texts = [name]
                if size_text:
                    texts.append(f"Size: {size_text}")

                # ---------------------------------------------------
                # 4. OPTIMIZED: Extract rotated ROI and process
                # ---------------------------------------------------
                if len(bbox_aligned) == 4 and moment_vector:
                    # Principal axis direction
                    vx = moment_vector.get("vx", 1.0)
                    vy = moment_vector.get("vy", 0.0)
                    v0 = np.array([vx, vy], dtype=np.float32)
                    v0 = v0 / (np.linalg.norm(v0) + 1e-9)

                    mean = np.array([cx, cy], dtype=np.float32)

                    # ---------------------------------------------------
                    # 4.1 Extract and rotate ROI (BIG OPTIMIZATION)
                    # ---------------------------------------------------
                    rotated_roi, roi_info = self.extract_rotated_roi(binary_image, bbox_aligned, mean, v0)

                    if rotated_roi is None:
                        if self.debug_logging:
                            self.get_logger().warn(f"  │ Failed to extract ROI for object {object_id}")
                        output_json["objects"].append(obj_features)
                        continue

                    # ---------------------------------------------------
                    # 4.2 Fast slice width computation (column sums)
                    # ---------------------------------------------------
                    slice_widths = self.compute_slice_widths_fast(rotated_roi)

                    # Adaptive slice count based on object length
                    num_cols = len(slice_widths)
                    num_slices = int(np.clip(num_cols, self.min_slices, self.max_slices))

                    # Downsample if needed
                    if num_cols > num_slices:
                        indices = np.linspace(0, num_cols - 1, num_slices).astype(int)
                        slice_widths = slice_widths[indices]

                    n = len(slice_widths)

                    if self.debug_logging:
                        self.get_logger().info(f"  │ Using {n} slices (ROI: {rotated_roi.shape[1]}x{rotated_roi.shape[0]})")

                    # ---------------------------------------------------
                    # 4.3 Smooth and detect head-body separation
                    # ---------------------------------------------------
                    window_size = max(5, n // 20)
                    slice_widths_smooth = self.smooth_1d(slice_widths, window_size)

                    # Compute mean width in first 1/3 and last 1/3
                    third = max(5, n // 3)
                    head_region_width = np.mean(slice_widths_smooth[:third])
                    body_region_width = np.mean(slice_widths_smooth[-third:])

                    if self.debug_logging:
                        self.get_logger().info(
                            f"  │ Head avg: {head_region_width:.1f}px, Body avg: {body_region_width:.1f}px"
                        )

                    # Find separation point
                    target_width = (head_region_width + body_region_width) / 2.0
                    search_start = int(n * 0.2)
                    search_end = int(n * 0.8)

                    separation_idx = n // 2
                    min_diff = float('inf')

                    for i in range(search_start, search_end):
                        diff = abs(slice_widths_smooth[i] - target_width)
                        if diff < min_diff:
                            min_diff = diff
                            separation_idx = i

                    if self.debug_logging:
                        self.get_logger().info(
                            f"  │ Separation at slice {separation_idx}/{n} "
                            f"(width={slice_widths_smooth[separation_idx]:.1f}px)"
                        )

                    # ---------------------------------------------------
                    # 4.4 Map separation line back to original image
                    # ---------------------------------------------------
                    # Find actual column index in rotated ROI
                    if num_cols > num_slices:
                        sep_col = indices[separation_idx]
                    else:
                        sep_col = separation_idx

                    # Get top and bottom points of separation line
                    sep_col_data = rotated_roi[:, sep_col]
                    nonzero_rows = np.where(sep_col_data > 0)[0]

                    if len(nonzero_rows) > 0:
                        sep_top_row = nonzero_rows[0]
                        sep_bottom_row = nonzero_rows[-1]

                        # Map back to original coordinates
                        sep_p1 = self.map_roi_to_original(sep_col, sep_top_row, roi_info)
                        sep_p2 = self.map_roi_to_original(sep_col, sep_bottom_row, roi_info)

                        # Draw separation line
                        cv2.line(output_image, sep_p1, sep_p2, (255, 255, 0), 2)  # cyan

                        obj_features["separation_line"] = {
                            "p1": [float(sep_p1[0]), float(sep_p1[1])],
                            "p2": [float(sep_p2[0]), float(sep_p2[1])],
                            "slice_index": int(separation_idx),
                            "total_slices": int(n)
                        }

                    # ---------------------------------------------------
                    # 4.5 Determine body region
                    # ---------------------------------------------------
                    before = slice_widths_smooth[:separation_idx + 1]
                    after = slice_widths_smooth[separation_idx + 1:]

                    if len(before) == 0 or len(after) == 0:
                        body_slice_widths = slice_widths_smooth
                        body_start_idx = 0
                        body_end_idx = n - 1
                    else:
                        mean_before = float(np.mean(before))
                        mean_after = float(np.mean(after))

                        if mean_before < mean_after:
                            body_slice_widths = before
                            body_start_idx = 0
                            body_end_idx = separation_idx
                        else:
                            body_slice_widths = after
                            body_start_idx = separation_idx + 1
                            body_end_idx = n - 1

                    # Body measurements
                    body_diameter_px = float(np.mean(body_slice_widths)) if len(body_slice_widths) > 0 else 0.0
                    body_diameter_mm = body_diameter_px * self.pixel_to_mm_ratio
                    body_length_px = float(body_end_idx - body_start_idx)
                    body_length_mm = body_length_px * self.pixel_to_mm_ratio

                    obj_features["body_diameter_px"] = float(body_diameter_px)
                    obj_features["body_diameter_mm"] = float(body_diameter_mm)
                    obj_features["body_length_px"] = float(body_length_px)
                    obj_features["body_length_mm"] = float(body_length_mm)

                    # ---------------------------------------------------
                    # 4.6 OPTIMIZED: Fast jagginess estimation
                    # ---------------------------------------------------
                    # Map slice indices to actual columns
                    if num_cols > num_slices:
                        body_start_col = indices[body_start_idx]
                        body_end_col = indices[body_end_idx]
                    else:
                        body_start_col = body_start_idx
                        body_end_col = body_end_idx

                    jag_overall, jag_top, jag_bottom = self.estimate_jagginess_fast(
                        rotated_roi, body_start_col, body_end_col
                    )

                    if self.debug_logging:
                        self.get_logger().info(
                            f"  │ Jagginess: overall={jag_overall:.3f}px, "
                            f"top={jag_top:.3f}px, bottom={jag_bottom:.3f}px"
                        )

                    obj_features["jagginess"] = {
                        "overall_px": float(jag_overall),
                        "top_px": float(jag_top),
                        "bottom_px": float(jag_bottom)
                    }

                    # ---------------------------------------------------
                    # 4.7 Classify screw type
                    # ---------------------------------------------------
                    screw_type, type_color, jag_norm = self.classify_screw_by_jagginess(
                        jag_value=jag_overall,
                        body_diameter_px=body_diameter_px
                    )

                    obj_features["classification"] = {
                        "type": screw_type,
                        "jagginess_normalized": float(jag_norm),
                        "jagginess_threshold": float(self.jagginess_threshold)
                    }

                    # ---------------------------------------------------
                    # 4.8 Pick-up point (center of body region)
                    # ---------------------------------------------------
                    body_center_idx = (body_start_idx + body_end_idx) // 2
                    if num_cols > num_slices:
                        body_center_col = indices[body_center_idx]
                    else:
                        body_center_col = body_center_idx

                    # Find center row in body center column
                    body_center_col_data = rotated_roi[:, body_center_col]
                    nonzero_rows = np.where(body_center_col_data > 0)[0]

                    if len(nonzero_rows) > 0:
                        body_center_row = (nonzero_rows[0] + nonzero_rows[-1]) // 2
                        pick_up_x, pick_up_y = self.map_roi_to_original(body_center_col, body_center_row, roi_info)

                        obj_features["pickup_point_px"] = [float(pick_up_x), float(pick_up_y)]
                        obj_features["pickup_point_mm"] = [
                            float(pick_up_x * self.pixel_to_mm_ratio),
                            float(pick_up_y * self.pixel_to_mm_ratio)
                        ]

                        # Draw pick-up point
                        cv2.circle(output_image, (pick_up_x, pick_up_y), 5, (0, 255, 0), -1)
                        cross_size = 8
                        cv2.line(output_image, (pick_up_x - cross_size, pick_up_y),
                                (pick_up_x + cross_size, pick_up_y), (0, 255, 0), 2)
                        cv2.line(output_image, (pick_up_x, pick_up_y - cross_size),
                                (pick_up_x, pick_up_y + cross_size), (0, 255, 0), 2)

                    # ---------------------------------------------------
                    # 4.9 Draw body region highlight
                    # ---------------------------------------------------
                    # Get corners of body region in rotated ROI
                    if num_cols > num_slices:
                        body_start_col_actual = indices[body_start_idx]
                        body_end_col_actual = indices[body_end_idx]
                    else:
                        body_start_col_actual = body_start_idx
                        body_end_col_actual = body_end_idx

                    # Find top and bottom edges at start and end
                    start_col_data = rotated_roi[:, body_start_col_actual]
                    end_col_data = rotated_roi[:, body_end_col_actual]

                    start_rows = np.where(start_col_data > 0)[0]
                    end_rows = np.where(end_col_data > 0)[0]

                    if len(start_rows) > 0 and len(end_rows) > 0:
                        # Map corners back to original
                        p1 = self.map_roi_to_original(body_start_col_actual, start_rows[0], roi_info)
                        p2 = self.map_roi_to_original(body_start_col_actual, start_rows[-1], roi_info)
                        p3 = self.map_roi_to_original(body_end_col_actual, end_rows[-1], roi_info)
                        p4 = self.map_roi_to_original(body_end_col_actual, end_rows[0], roi_info)

                        body_poly = np.array([p1, p2, p3, p4], dtype=np.int32)

                        obj_features["body_region_polygon"] = [
                            [float(p[0]), float(p[1])] for p in body_poly
                        ]

                        # Draw semi-transparent overlay
                        overlay = output_image.copy()
                        cv2.fillPoly(overlay, [body_poly], type_color)
                        cv2.addWeighted(overlay, 0.3, output_image, 0.7, 0, output_image)

                    # ---------------------------------------------------
                    # 4.10 Add text annotations
                    # ---------------------------------------------------
                    texts.append(f"Body dia: {body_diameter_mm:.2f} mm")
                    texts.append(f"Body len: {body_length_mm:.2f} mm")
                    texts.append(f"Type: {screw_type} (jag={jag_norm:.4f})")

                    # Draw screw type label
                    cv2.putText(
                        output_image,
                        screw_type,
                        (cx + 15, cy + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        type_color,
                        2,
                        cv2.LINE_AA
                    )

                    if self.debug_logging:
                        symbol = "🪵" if screw_type == "WOOD" else "🔩"
                        self.get_logger().info(f"  └─ ✓ {symbol} {screw_type}\n")

                # ---------------------------------------------------
                # 5. Draw all text annotations
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

                # Add processed object to output JSON
                output_json["objects"].append(obj_features)

            # ---------------------------------------------------
            # 6. Publish annotated image
            # ---------------------------------------------------
            out_msg = self.bridge.cv2_to_imgmsg(output_image, encoding="bgr8")
            out_msg.header = image_msg.header
            self.image_pub.publish(out_msg)

            # ---------------------------------------------------
            # 7. Publish JSON features
            # ---------------------------------------------------
            json_msg = String()
            json_msg.data = json.dumps(output_json, indent=2)
            self.json_pub.publish(json_msg)

            if self.debug_logging:
                self.get_logger().info(
                    f"✓ Published features for {len(output_json['objects'])} object(s)"
                )

        except Exception as e:
            self.get_logger().error(f"❌ Error in synchronized callback: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())


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