#!/usr/bin/env python3
"""
Physical Features Extraction ROS2 Node (OPTIMIZED VERSION)

High-performance version with efficient line sampling and reduced memory allocations.
All original functionalities preserved including jagginess-based classification.
"""

import json
import numpy as np
import numpy.linalg as LA
import cv2
from scipy.ndimage import uniform_filter1d

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge


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
            0.040,
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
        self.declare_parameter(
            "max_slices",
            300,
            ParameterDescriptor(description="Maximum number of slices per object (performance cap)")
        )

        # Get parameter values
        input_image_topic = self.get_parameter("input_image_topic").value
        input_info_topic = self.get_parameter("input_info_topic").value
        output_image_topic = self.get_parameter("output_image_topic").value
        output_json_topic = self.get_parameter("output_json_topic").value
        self.pixel_to_mm_ratio = self.get_parameter("pixel_to_mm_ratio").value
        self.center_dot_radius = self.get_parameter("center_dot_radius").value
        self.jagginess_threshold = self.get_parameter("jagginess_threshold").value
        self.frame_skip = self.get_parameter("frame_skip").value
        self.debug_logging = self.get_parameter("debug_logging").value
        self.process_closest_only = self.get_parameter("process_closest_only").value
        self.max_slices = self.get_parameter("max_slices").value

        self.bbox_color = (
            self.get_parameter("bbox_color_b").value,
            self.get_parameter("bbox_color_g").value,
            self.get_parameter("bbox_color_r").value
        )
        self.bbox_thickness = self.get_parameter("bbox_thickness").value

        # Frame counter for skipping
        self.frame_counter = 0

        # Store latest data from each topic
        self.latest_image = None
        self.latest_info = None
        self.latest_image_header = None
        
        # Flag to prevent double processing
        self.processing = False

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
        self.get_logger().info(f"  Frame skip:          {self.frame_skip} (process every {self.frame_skip} frame(s))")
        self.get_logger().info(f"  Debug logging:       {self.debug_logging}")
        self.get_logger().info(f"  Process closest only: {self.process_closest_only}")
        self.get_logger().info("-" * 70)
        self.get_logger().info("  CLASSIFICATION THRESHOLD:")
        self.get_logger().info(f"    Jagginess threshold: {self.jagginess_threshold}")
        self.get_logger().info("=" * 70)

        self.bridge = CvBridge()

        # ---------------------------------------------------
        # Independent subscribers (no synchronization)
        # ---------------------------------------------------
        self.image_sub = self.create_subscription(
            Image,
            input_image_topic,
            self.image_callback,
            10
        )
        
        self.info_sub = self.create_subscription(
            String,
            input_info_topic,
            self.info_callback,
            10
        )

        # ---------------------------------------------------
        # Publishers
        # ---------------------------------------------------
        self.image_pub = self.create_publisher(Image, output_image_topic, 10)
        self.json_pub = self.create_publisher(String, output_json_topic, 10)

        self.get_logger().info("✓ Node ready. Processing topics independently...\n")

    # =======================================================
    # OPTIMIZED: Efficient line sampling using Bresenham
    # =======================================================
    def sample_line_efficient(self, binary_image, p1, p2):
        """
        Sample pixels along a line efficiently without creating full-frame masks.
        Uses Bresenham's line algorithm.
        
        Returns:
            tuple: (points_array, white_count) where points_array is Nx2 array of [x,y]
        """
        x1, y1 = int(round(p1[0])), int(round(p1[1]))
        x2, y2 = int(round(p2[0])), int(round(p2[1]))
        
        h, w = binary_image.shape[:2]
        
        # Bresenham's line algorithm
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        points = []
        white_count = 0
        
        x, y = x1, y1
        
        while True:
            # Check bounds and sample
            if 0 <= x < w and 0 <= y < h:
                if binary_image[y, x] > 0:
                    points.append([x, y])
                    white_count += 1
            
            if x == x2 and y == y2:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return np.array(points, dtype=np.float32) if points else np.empty((0, 2), dtype=np.float32), white_count

    # =======================================================
    # Image callback - stores latest image
    # =======================================================
    def image_callback(self, msg: Image):
        """Store the latest image and trigger processing."""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
            self.latest_image_header = msg.header
            
            # Only trigger processing from image callback to avoid double processing
            if self.latest_info is not None and not self.processing:
                self.process_and_publish()
                
        except Exception as e:
            self.get_logger().error(f"❌ Error in image callback: {e}")

    # =======================================================
    # Info callback - stores latest object info
    # =======================================================
    def info_callback(self, msg: String):
        """Store the latest object information."""
        try:
            self.latest_info = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"❌ Error in info callback: {e}")

    # =======================================================
    # OPTIMIZED: Estimate side jagginess with efficient sampling
    # =======================================================
    def estimate_side_jagginess(
        self,
        binary_image,
        mean,
        v0, v1,
        slice_positions,
        min_v1, max_v1,
        body_start_idx,
        body_end_idx,
        slice_data  # Reuse already computed slice data
    ):
        """
        Estimate 'jagginess' of the shaft side contour using pre-computed slice data.
        """
        s_samples = []
        t_side_min = []
        t_side_max = []

        for i in range(body_start_idx, body_end_idx + 1):
            if i >= len(slice_data):
                continue
                
            points = slice_data[i]['points']
            
            if len(points) == 0:
                continue

            rel = points - mean
            s_vals = rel @ v0
            t_vals = rel @ v1

            t_min = float(np.min(t_vals))
            t_max = float(np.max(t_vals))
            s_mid = float(np.mean(s_vals))

            s_samples.append(s_mid)
            t_side_min.append(t_min)
            t_side_max.append(t_max)

        if len(s_samples) < 5:
            return 0.0, 0.0, 0.0

        s_samples = np.asarray(s_samples, dtype=np.float32)
        t_side_min = np.asarray(t_side_min, dtype=np.float32)
        t_side_max = np.asarray(t_side_max, dtype=np.float32)

        def jag_for_side(s, t):
            if len(t) < 5:
                return 0.0
            A = np.vstack([s, np.ones_like(s)]).T
            a, b = LA.lstsq(A, t, rcond=None)[0]
            t_fit = a * s + b
            residuals = t - t_fit

            # Smooth the residuals
            smooth_window = max(3, len(residuals) // 15)
            residuals_smooth = uniform_filter1d(residuals, size=smooth_window, mode='nearest')

            return float(np.std(residuals_smooth))

        jag_min = jag_for_side(s_samples, t_side_min)
        jag_max = jag_for_side(s_samples, t_side_max)
        jag_overall = max(jag_min, jag_max)

        return jag_overall, jag_min, jag_max

    # =======================================================
    # Classify screw type based on jagginess
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
        color = (0, 255, 255) if is_wood else (255, 0, 255)

        if self.debug_logging:
            self.get_logger().info(
                f"    Jagginess: raw={jag_value:.3f}px, "
                f"dia={body_diameter_px:.1f}px, norm={jag_norm:.4f} → {label}"
            )

        return label, color, jag_norm

    # =======================================================
    # OPTIMIZED: Main processing function
    # =======================================================
    def process_and_publish(self):
        """
        Process the latest image and info data with optimized algorithms.
        """
        # Prevent re-entry
        if self.processing:
            return
            
        self.processing = True

        try:
            # Frame skipping
            self.frame_counter += 1
            if self.frame_counter % self.frame_skip != 0:
                return

            binary_image = self.latest_image.copy()
            output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

            objects = self.latest_info.get("objects", [])

            # Prepare output JSON
            output_json = {
                "timestamp": self.latest_image_header.stamp.sec + self.latest_image_header.stamp.nanosec * 1e-9,
                "frame_id": self.latest_image_header.frame_id,
                "pixel_to_mm_ratio": self.pixel_to_mm_ratio,
                "objects": []
            }

            if len(objects) > 0 and self.debug_logging:
                self.get_logger().info(f"\n{'='*70}\n  PROCESSING {len(objects)} OBJECT(S)\n{'='*70}")

            # Filter to closest object if enabled
            if self.process_closest_only and len(objects) > 0:
                h, w = binary_image.shape[:2]
                img_center = np.array([w / 2.0, h / 2.0])

                min_dist = float('inf')
                closest_obj = None

                for obj in objects:
                    center_px = obj.get("center_px", [0.0, 0.0])
                    obj_center = np.array(center_px)
                    dist = np.linalg.norm(obj_center - img_center)

                    if dist < min_dist:
                        min_dist = dist
                        closest_obj = obj

                objects = [closest_obj] if closest_obj is not None else []

                if self.debug_logging:
                    self.get_logger().info(
                        f"  ✓ Filtered to closest object (distance: {min_dist:.1f}px from center)"
                    )

            # Process each object
            for obj in objects:
                object_id = obj.get("object_id", 0)
                name = obj.get("name", f"Object #{object_id}")
                center_px = obj.get("center_px", [0.0, 0.0])
                bbox_aligned = obj.get("bbox_aligned", [])
                moment_vector = obj.get("moment_vector", {})

                cx, cy = int(center_px[0]), int(center_px[1])

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

                # Draw bounding box
                if len(bbox_aligned) == 4:
                    bbox_np = np.array(bbox_aligned, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(output_image, [bbox_np], isClosed=True, color=(0, 255, 0), thickness=self.bbox_thickness)

                # Compute size
                width_px = height_px = None
                if len(bbox_aligned) == 4:
                    pts = np.array(bbox_aligned, dtype=np.float32)
                    v_edge0 = pts[1] - pts[0]
                    v_edge1 = pts[2] - pts[1]
                    len0 = np.linalg.norm(v_edge0)
                    len1 = np.linalg.norm(v_edge1)
                    width_px = max(len0, len1)
                    height_px = min(len0, len1)

                size_text = ""
                if width_px is not None and height_px is not None:
                    width_mm = width_px * self.pixel_to_mm_ratio
                    height_mm = height_px * self.pixel_to_mm_ratio
                    size_text = f"{width_mm:.2f} x {height_mm:.2f} mm"

                    obj_features["bbox_width_px"] = float(width_px)
                    obj_features["bbox_height_px"] = float(height_px)
                    obj_features["bbox_width_mm"] = float(width_mm)
                    obj_features["bbox_height_mm"] = float(height_mm)

                texts = [name]
                if size_text:
                    texts.append(f"Size: {size_text}")

                # Extract PCA info
                if len(bbox_aligned) == 4 and moment_vector:
                    vx = moment_vector.get("vx", 1.0)
                    vy = moment_vector.get("vy", 0.0)
                    v0 = np.array([vx, vy], dtype=np.float32)
                    v0 = v0 / (np.linalg.norm(v0) + 1e-9)

                    v1 = np.array([-v0[1], v0[0]], dtype=np.float32)
                    mean = np.array([cx, cy], dtype=np.float32)

                    pts = np.array(bbox_aligned, dtype=np.float32)
                    proj_v0 = np.dot(pts - mean, v0)
                    proj_v1 = np.dot(pts - mean, v1)

                    min_v0 = np.min(proj_v0)
                    max_v0 = np.max(proj_v0)
                    min_v1 = np.min(proj_v1)
                    max_v1 = np.max(proj_v1)

                    # OPTIMIZED: Scan slices with cap
                    length_px = max_v0 - min_v0
                    num_slices = min(int(length_px * 2), self.max_slices)
                    if num_slices < 10:
                        num_slices = 10

                    if self.debug_logging:
                        self.get_logger().info(f"  │ Scanning {num_slices} slices along {length_px:.1f}px")

                    slice_positions = np.linspace(min_v0, max_v0, num_slices)
                    slice_widths = []
                    slice_data = []  # Store for reuse in jagginess

                    # OPTIMIZED: Single pass slice scanning
                    for s in slice_positions:
                        center_slice = mean + s * v0
                        p1 = center_slice + min_v1 * v1
                        p2 = center_slice + max_v1 * v1

                        # Use efficient line sampling
                        points, white_count = self.sample_line_efficient(binary_image, p1, p2)
                        
                        slice_widths.append(white_count)
                        slice_data.append({'points': points, 'width': white_count})

                    slice_widths = np.array(slice_widths, dtype=np.float32)

                    # Detect head-body separation
                    n = len(slice_widths)
                    window_size = max(5, n // 20)
                    slice_widths_smooth = uniform_filter1d(slice_widths, size=window_size, mode='nearest')

                    third = max(5, n // 3)
                    head_region_width = np.mean(slice_widths_smooth[:third])
                    body_region_width = np.mean(slice_widths_smooth[-third:])

                    if self.debug_logging:
                        self.get_logger().info(
                            f"  │ Head avg: {head_region_width:.1f}px, Body avg: {body_region_width:.1f}px"
                        )

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

                    separation_pos = slice_positions[separation_idx]
                    separation_center = mean + separation_pos * v0
                    sep_p1 = separation_center + min_v1 * v1
                    sep_p2 = separation_center + max_v1 * v1

                    sep_p1_int = (int(sep_p1[0]), int(sep_p1[1]))
                    sep_p2_int = (int(sep_p2[0]), int(sep_p2[1]))

                    cv2.line(output_image, sep_p1_int, sep_p2_int, (255, 255, 0), 2)

                    obj_features["separation_line"] = {
                        "p1": [float(sep_p1[0]), float(sep_p1[1])],
                        "p2": [float(sep_p2[0]), float(sep_p2[1])],
                        "center_point": [float(separation_center[0]), float(separation_center[1])],
                        "slice_index": int(separation_idx),
                        "total_slices": int(n)
                    }

                    # Determine body region
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

                    body_diameter_px = float(np.mean(body_slice_widths)) if len(body_slice_widths) > 0 else 0.0
                    body_diameter_mm = body_diameter_px * self.pixel_to_mm_ratio

                    body_start_pos = slice_positions[body_start_idx]
                    body_end_pos = slice_positions[body_end_idx]
                    body_length_px = abs(body_end_pos - body_start_pos)
                    body_length_mm = body_length_px * self.pixel_to_mm_ratio

                    obj_features["body_diameter_px"] = float(body_diameter_px)
                    obj_features["body_diameter_mm"] = float(body_diameter_mm)
                    obj_features["body_length_px"] = float(body_length_px)
                    obj_features["body_length_mm"] = float(body_length_mm)

                    # OPTIMIZED: Estimate jagginess using pre-computed slice data
                    jag_overall, jag_min, jag_max = self.estimate_side_jagginess(
                        binary_image=binary_image,
                        mean=mean,
                        v0=v0,
                        v1=v1,
                        slice_positions=slice_positions,
                        min_v1=min_v1,
                        max_v1=max_v1,
                        body_start_idx=body_start_idx,
                        body_end_idx=body_end_idx,
                        slice_data=slice_data
                    )

                    if self.debug_logging:
                        self.get_logger().info(
                            f"  │ Jagginess: overall={jag_overall:.3f}px, "
                            f"side_min={jag_min:.3f}px, side_max={jag_max:.3f}px"
                        )

                    obj_features["jagginess"] = {
                        "overall_px": float(jag_overall),
                        "side_min_px": float(jag_min),
                        "side_max_px": float(jag_max)
                    }

                    # Classify screw type
                    screw_type, type_color, jag_norm = self.classify_screw_by_jagginess(
                        jag_value=jag_overall,
                        body_diameter_px=body_diameter_px
                    )

                    obj_features["classification"] = {
                        "type": screw_type,
                        "jagginess_normalized": float(jag_norm),
                        "jagginess_threshold": float(self.jagginess_threshold)
                    }

                    # Pick-up point
                    body_center_pos = 0.5 * (body_start_pos + body_end_pos)
                    pick_up_world = mean + body_center_pos * v0
                    pick_up_x = int(pick_up_world[0])
                    pick_up_y = int(pick_up_world[1])

                    # Calculate image center
                    h, w = binary_image.shape[:2]
                    img_center_x = w / 2.0
                    img_center_y = h / 2.0

                    # Calculate pickup point relative to camera center
                    pick_up_relative_x = pick_up_world[0] - img_center_x
                    pick_up_relative_y = pick_up_world[1] - img_center_y

                    # Store both absolute and relative coordinates
                    obj_features["pickup_point_px"] = [float(pick_up_x), float(pick_up_y)]
                    obj_features["pickup_point_mm"] = [
                        float(pick_up_x * self.pixel_to_mm_ratio),
                        float(pick_up_y * self.pixel_to_mm_ratio)
                    ]
                    obj_features["pickup_point_relative_to_camera_px"] = [
                        float(pick_up_relative_x), 
                        float(pick_up_relative_y)
                    ]
                    obj_features["pickup_point_relative_to_camera_mm"] = [
                        float(pick_up_relative_x * self.pixel_to_mm_ratio),
                        float(pick_up_relative_y * self.pixel_to_mm_ratio)
                    ]

                    # Draw pick-up point
                    cv2.circle(output_image, (pick_up_x, pick_up_y), 5, (0, 255, 0), -1)
                    cross_size = 8
                    cv2.line(output_image, (pick_up_x - cross_size, pick_up_y),
                            (pick_up_x + cross_size, pick_up_y), (0, 255, 0), 2)
                    cv2.line(output_image, (pick_up_x, pick_up_y - cross_size),
                            (pick_up_x, pick_up_y + cross_size), (0, 255, 0), 2)

                    # Draw body region
                    body_start_center = mean + body_start_pos * v0
                    body_end_center = mean + body_end_pos * v0

                    body_poly = np.array([
                        body_start_center + min_v1 * v1,
                        body_start_center + max_v1 * v1,
                        body_end_center + max_v1 * v1,
                        body_end_center + min_v1 * v1,
                    ], dtype=np.int32)

                    obj_features["body_region_polygon"] = [
                        [float(p[0]), float(p[1])] for p in body_poly
                    ]

                    overlay = output_image.copy()
                    cv2.fillPoly(overlay, [body_poly], type_color)
                    cv2.addWeighted(overlay, 0.3, output_image, 0.7, 0, output_image)

                    texts.append(f"Body dia: {body_diameter_mm:.2f} mm")
                    texts.append(f"Body len: {body_length_mm:.2f} mm")
                    texts.append(f"Type: {screw_type} (jag={jag_norm:.4f})")

                    cv2.putText(output_image, screw_type, (cx + 15, cy + 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, type_color, 2, cv2.LINE_AA)

                    if self.debug_logging:
                        symbol = "🪵" if screw_type == "WOOD" else "🔩"
                        self.get_logger().info(f"  └─ ✓ {symbol} {screw_type}\n")

                # Draw text annotations
                text_x = cx + 10
                text_y = cy - 10
                line_spacing = 20

                for i, text in enumerate(texts):
                    cv2.putText(output_image, text, (text_x, text_y + i * line_spacing),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.bbox_color, 1, cv2.LINE_AA)

                output_json["objects"].append(obj_features)

            # Publish results
            out_msg = self.bridge.cv2_to_imgmsg(output_image, encoding="bgr8")
            out_msg.header = self.latest_image_header
            self.image_pub.publish(out_msg)

            # OPTIMIZED: Compact JSON (no indentation)
            json_msg = String()
            json_msg.data = json.dumps(output_json)
            self.json_pub.publish(json_msg)

            if self.debug_logging:
                self.get_logger().info(
                    f"✓ Published features for {len(output_json['objects'])} object(s)"
                )

        except Exception as e:
            self.get_logger().error(f"❌ Error in processing: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
        finally:
            self.processing = False


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