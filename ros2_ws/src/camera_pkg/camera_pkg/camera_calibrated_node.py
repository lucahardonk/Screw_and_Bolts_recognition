#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import requests
import yaml

class CameraCalibratedNode(Node):
    def __init__(self):
        super().__init__('camera_calibrated_node')
        
        # Parameters
        self.declare_parameter('stream_url', 'http://192.168.1.161:8000/video')
        self.declare_parameter('calibration_file', 'camera_calibration.yaml')
        self.declare_parameter('publish_rate', 15.0)
        
        self.stream_url = self.get_parameter('stream_url').value
        self.calibration_file = self.get_parameter('calibration_file').value
        publish_rate = self.get_parameter('publish_rate').value
        
        # Publishers
        self.image_pub = self.create_publisher(Image, 'camera/image_rect', 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, 'camera/camera_info', 10)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Load calibration
        self.load_calibration()
        
        # Timer for publishing
        self.timer = self.create_timer(1.0 / publish_rate, self.timer_callback)
        
        # Stream state
        self.stream = None
        self.bytes_data = bytes()
        
        self.get_logger().info('Camera Calibrated Node Started')
        self.get_logger().info(f'Publishing undistorted images to: camera/image_rect')
    
    def load_calibration(self):
        """Load camera calibration from YAML file."""
        try:
            with open(self.calibration_file, 'r') as f:
                calib = yaml.safe_load(f)
            
            self.K = np.array(calib['camera_matrix'])
            self.D = np.array(calib['distortion_coefficients'])
            self.img_width = calib['image_width']
            self.img_height = calib['image_height']
            
            self.get_logger().info(f'Loaded calibration from {self.calibration_file}')
            self.get_logger().info(f'Image size: {self.img_width}x{self.img_height}')
            
            # Compute undistortion maps for fisheye
            self.new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                self.K, self.D, 
                (self.img_width, self.img_height),
                np.eye(3),
                balance=0.0  # 0=preserve all pixels, 1=crop to valid area
            )
            
            self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
                self.K, self.D, np.eye(3), self.new_K,
                (self.img_width, self.img_height), cv2.CV_16SC2
            )
            
            self.get_logger().info('Undistortion maps computed')
            
        except Exception as e:
            self.get_logger().error(f'Failed to load calibration: {e}')
            raise
    
    def get_frame_from_stream(self):
        """Fetch a single frame from MJPEG stream."""
        try:
            if self.stream is None:
                self.stream = requests.get(self.stream_url, stream=True, timeout=5)
            
            for chunk in self.stream.iter_content(chunk_size=4096):
                self.bytes_data += chunk
                a = self.bytes_data.find(b'\xff\xd8')  # JPEG start
                b = self.bytes_data.find(b'\xff\xd9')  # JPEG end
                
                if a != -1 and b != -1:
                    jpg = self.bytes_data[a:b+2]
                    self.bytes_data = self.bytes_data[b+2:]
                    
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    return frame
                    
        except Exception as e:
            self.get_logger().warn(f'Stream error: {e}')
            self.stream = None
            self.bytes_data = bytes()
            return None
    
    def timer_callback(self):
        """Capture, undistort, and publish frame."""
        frame = self.get_frame_from_stream()
        
        if frame is None:
            return
        
        # Undistort fisheye image
        undistorted = cv2.remap(frame, self.map1, self.map2, 
                                interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
        
        # Publish image
        msg = self.bridge.cv2_to_imgmsg(undistorted, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_optical_frame'
        self.image_pub.publish(msg)
        
        # Publish camera info
        camera_info_msg = self.create_camera_info_msg()
        self.camera_info_pub.publish(camera_info_msg)
    
    def create_camera_info_msg(self):
        """Create CameraInfo message."""
        msg = CameraInfo()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_optical_frame'
        msg.height = self.img_height
        msg.width = self.img_width
        msg.distortion_model = 'plumb_bob'  # After undistortion
        
        # Rectified camera has no distortion
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        msg.k = self.new_K.flatten().tolist()
        msg.r = np.eye(3).flatten().tolist()
        msg.p = np.hstack([self.new_K, [[0], [0], [0]]]).flatten().tolist()
        
        return msg

def main(args=None):
    rclpy.init(args=args)
    node = CameraCalibratedNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
