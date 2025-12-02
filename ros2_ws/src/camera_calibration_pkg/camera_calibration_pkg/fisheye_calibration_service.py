#!/usr/bin/env python3

import os
import glob
import yaml
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


class FisheyeCalibrationService(Node):
    def __init__(self):
        super().__init__('fisheye_calibration_service')

        # Detect workspace root by walking up from __file__
        current = os.path.dirname(os.path.abspath(__file__))
        workspace_root = None
        
        # Walk up until we find 'ros2_ws' directory
        while current != '/':
            if os.path.basename(current) == 'ros2_ws':
                workspace_root = current
                break
            current = os.path.dirname(current)
        
        if workspace_root is None:
            self.get_logger().error("Could not find 'ros2_ws' in path! Using fallback.")
            # Fallback: assume we're 2 levels deep from package root
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.base_dir = os.path.dirname(script_dir)
        else:
            # Use source tree: ros2_ws/src/camera_calibration_pkg
            self.base_dir = os.path.join(workspace_root, 'src', 'camera_calibration_pkg')
        
        self.get_logger().info(f"Detected base_dir: {self.base_dir}")

        # Parameters
        self.declare_parameter('images_dir', os.path.join(self.base_dir, 'images'))
        self.declare_parameter('output_file', os.path.join(self.base_dir, 'fisheye_camera.yaml'))
        self.declare_parameter('pattern_cols', 8)  # Changed from 9 to 8 - try both!
        self.declare_parameter('pattern_rows', 5)  # Changed from 6 to 5 - try both!
        self.declare_parameter('square_size', 0.025)
        self.declare_parameter('visualize', False)
        self.declare_parameter('save_postprocessed', True)

        self.images_dir = self.get_parameter('images_dir').value
        self.output_file = self.get_parameter('output_file').value
        self.pattern_cols = self.get_parameter('pattern_cols').value
        self.pattern_rows = self.get_parameter('pattern_rows').value
        self.square_size = self.get_parameter('square_size').value
        self.visualize = self.get_parameter('visualize').value
        self.save_postprocessed = self.get_parameter('save_postprocessed').value

        # Create the service
        self.srv = self.create_service(
            Trigger,
            'run_fisheye_calibration',
            self.run_calibration_callback
        )

        self.get_logger().info("=" * 60)
        self.get_logger().info("FisheyeCalibrationService ready!")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Images directory: {self.images_dir}")
        self.get_logger().info(f"Output file: {self.output_file}")
        self.get_logger().info(f"Chessboard pattern: {self.pattern_cols}x{self.pattern_rows} inner corners")
        self.get_logger().info(f"Square size: {self.square_size} meters")
        self.get_logger().info(f"Visualize: {self.visualize}")
        self.get_logger().info(f"Save postprocessed: {self.save_postprocessed}")
        self.get_logger().info("=" * 60)
        self.get_logger().info("Call service with:")
        self.get_logger().info("  ros2 service call /run_fisheye_calibration std_srvs/srv/Trigger \"{}\"")
        self.get_logger().info("=" * 60)

    def run_calibration_callback(self, request, response):
        """Service callback - runs the calibration"""
        try:
            success, message = self.run_calibration()
            response.success = success
            response.message = message
        except Exception as e:
            error_msg = f"Exception during calibration: {type(e).__name__}: {e}"
            self.get_logger().error(error_msg)
            response.success = False
            response.message = error_msg
        return response

    def run_calibration(self):
        """Main calibration logic"""
        self.get_logger().info("\n" + "=" * 60)
        self.get_logger().info("Starting fisheye calibration...")
        self.get_logger().info("=" * 60)

        pattern_size = (self.pattern_cols, self.pattern_rows)

        # Prepare 3D object points (real-world coordinates)
        objp = np.zeros((1, pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objp *= self.square_size  # scale to real-world units

        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane

        # Find all images
        images = sorted(glob.glob(os.path.join(self.images_dir, '*.jpg')))
        images += sorted(glob.glob(os.path.join(self.images_dir, '*.png')))
        images += sorted(glob.glob(os.path.join(self.images_dir, '*.jpeg')))

        if len(images) == 0:
            msg = f"ERROR: No images found in {self.images_dir}"
            self.get_logger().error(msg)
            return False, msg

        self.get_logger().info(f"Found {len(images)} images in {self.images_dir}")

        img_shape = None
        successful_images = []

        for idx, fname in enumerate(images):
            # Skip already postprocessed images
            if '_postprocessed' in os.path.basename(fname):
                continue
                
            self.get_logger().info(f"Processing [{idx+1}/{len(images)}]: {os.path.basename(fname)}")
            
            img = cv2.imread(fname)
            if img is None:
                self.get_logger().warning(f"  ✗ Could not read image")
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # IMPROVED PREPROCESSING for fisheye/screen images
            # 1. Histogram equalization for better contrast
            gray_eq = cv2.equalizeHist(gray)
            
            # 2. Slight blur to reduce noise
            gray_processed = cv2.GaussianBlur(gray_eq, (5, 5), 0)

            if img_shape is None:
                img_shape = gray.shape[::-1]  # (width, height)
                self.get_logger().info(f"Image size: {img_shape[0]}x{img_shape[1]}")
            else:
                if img_shape != gray.shape[::-1]:
                    self.get_logger().warning(f"  ✗ Size mismatch, skipping")
                    continue

            # Try multiple detection strategies
            ret = False
            corners = None
            
            # Strategy 1: Standard detection with preprocessing
            ret, corners = cv2.findChessboardCorners(
                gray_processed,
                pattern_size,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH
                      + cv2.CALIB_CB_NORMALIZE_IMAGE
                      + cv2.CALIB_CB_FILTER_QUADS
            )
            
            # Strategy 2: If failed, try without FAST_CHECK on original gray
            if not ret:
                self.get_logger().info(f"  Trying alternative detection method...")
                ret, corners = cv2.findChessboardCorners(
                    gray,
                    pattern_size,
                    flags=cv2.CALIB_CB_ADAPTIVE_THRESH
                          + cv2.CALIB_CB_NORMALIZE_IMAGE
                )
            
            # Strategy 3: Try with different preprocessing
            if not ret:
                self.get_logger().info(f"  Trying with CLAHE preprocessing...")
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray_clahe = clahe.apply(gray)
                ret, corners = cv2.findChessboardCorners(
                    gray_clahe,
                    pattern_size,
                    flags=cv2.CALIB_CB_ADAPTIVE_THRESH
                          + cv2.CALIB_CB_NORMALIZE_IMAGE
                )

            if not ret:
                self.get_logger().warning(f"  ✗ Chessboard not found (tried multiple methods)")
                
                # Save postprocessed image showing it failed
                if self.save_postprocessed:
                    base_name = os.path.splitext(os.path.basename(fname))[0]
                    postprocessed_path = os.path.join(self.images_dir, f"{base_name}_postprocessed.jpg")
                    
                    # Draw "NOT DETECTED" on the image
                    vis = img.copy()
                    cv2.putText(vis, "CHESSBOARD NOT DETECTED", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(vis, f"Looking for: {self.pattern_cols}x{self.pattern_rows}", (50, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imwrite(postprocessed_path, vis)
                    self.get_logger().info(f"  Saved failed detection: {base_name}_postprocessed.jpg")
                
                continue

            # Refine corner locations (use original gray for subpixel refinement)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            corners_subpix = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints.append(corners_subpix)
            successful_images.append(fname)

            self.get_logger().info(f"  ✓ Chessboard detected!")

            # Save postprocessed image with corners drawn
            if self.save_postprocessed:
                vis = img.copy()
                cv2.drawChessboardCorners(vis, pattern_size, corners_subpix, True)
                
                # Add text showing it was successful
                cv2.putText(vis, f"DETECTED: {self.pattern_cols}x{self.pattern_rows}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Extract base filename without extension
                base_name = os.path.splitext(os.path.basename(fname))[0]
                postprocessed_path = os.path.join(self.images_dir, f"{base_name}_postprocessed.jpg")
                
                cv2.imwrite(postprocessed_path, vis)
                self.get_logger().info(f"  Saved postprocessed: {base_name}_postprocessed.jpg")

            # Show visualization if enabled
            if self.visualize:
                vis = img.copy()
                cv2.drawChessboardCorners(vis, pattern_size, corners_subpix, True)
                cv2.imshow('Detected Corners', vis)
                cv2.waitKey(500)

        if self.visualize:
            cv2.destroyAllWindows()

        if len(objpoints) < 5:
            msg = f"ERROR: Not enough valid images with detected chessboard (have {len(objpoints)}, need at least 5)"
            self.get_logger().error(msg)
            self.get_logger().error(f"TIP: Make sure your chessboard has {self.pattern_cols}x{self.pattern_rows} INNER corners")
            self.get_logger().error(f"TIP: Try adjusting pattern_cols and pattern_rows parameters")
            return False, msg

        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Successfully detected chessboard in {len(objpoints)}/{len([f for f in images if '_postprocessed' not in f])} images")
        self.get_logger().info("=" * 60)
        self.get_logger().info("Running fisheye calibration...")

        N_OK = len(objpoints)
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3)) for _ in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3)) for _ in range(N_OK)]

        # Fisheye calibration flags
        flags = (
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
            cv2.fisheye.CALIB_CHECK_COND +
            cv2.fisheye.CALIB_FIX_SKEW
        )
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

        try:
            rms, _, _, _, _ = cv2.fisheye.calibrate(
                objpoints,
                imgpoints,
                img_shape,
                K,
                D,
                rvecs,
                tvecs,
                flags,
                criteria
            )
        except cv2.error as e:
            msg = f"ERROR: Fisheye calibration failed: {e}"
            self.get_logger().error(msg)
            return False, msg

        self.get_logger().info("=" * 60)
        self.get_logger().info("Calibration complete!")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"RMS reprojection error: {rms:.6f} pixels")
        self.get_logger().info(f"\nCamera Matrix (K):\n{K}")
        self.get_logger().info(f"\nDistortion Coefficients (D):\n{D.ravel()}")

        # Prepare output data
        data = {
            'image_width': int(img_shape[0]),
            'image_height': int(img_shape[1]),
            'camera_name': 'fisheye_camera',
            'camera_matrix': {
                'rows': 3,
                'cols': 3,
                'data': K.flatten().tolist()
            },
            'distortion_model': 'fisheye',
            'distortion_coefficients': {
                'rows': 1,
                'cols': 4,
                'data': D.flatten().tolist()
            },
            'rectification_matrix': {
                'rows': 3,
                'cols': 3,
                'data': np.eye(3).flatten().tolist()
            },
            'projection_matrix': {
                'rows': 3,
                'cols': 4,
                'data': np.hstack([K, np.zeros((3, 1))]).flatten().tolist()
            },
            'rms_error': float(rms),
            'calibration_info': {
                'pattern_cols': int(self.pattern_cols),
                'pattern_rows': int(self.pattern_rows),
                'square_size': float(self.square_size),
                'num_images_used': len(objpoints),
                'num_images_total': len([f for f in images if '_postprocessed' not in f])
            }
        }

        # Save to YAML file
        with open(self.output_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

        success_msg = f"✓ Calibration successful!\n  Output saved to: {self.output_file}\n  RMS error: {rms:.6f} pixels"
        self.get_logger().info("=" * 60)
        self.get_logger().info(success_msg)
        self.get_logger().info("=" * 60)
        
        return True, success_msg


def main(args=None):
    rclpy.init(args=args)
    node = FisheyeCalibrationService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()