#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import requests
import yaml
import base64
from flask import Flask, render_template_string, jsonify, request
from threading import Thread
import time

class CameraCalibrator(Node):
    def __init__(self):
        super().__init__('camera_calibrator')
        
        # Parameters
        self.declare_parameter('stream_url', 'http://10.255.255.254:8000/frame')
        self.declare_parameter('checkerboard_rows', 6)
        self.declare_parameter('checkerboard_cols', 9)
        self.declare_parameter('square_size', 0.025)
        self.declare_parameter('num_images', 20)
        self.declare_parameter('output_file', 'camera_calibration.yaml')
        self.declare_parameter('web_port', 5000)
        
        self.stream_url = self.get_parameter('stream_url').value
        self.checkerboard_size = (
            self.get_parameter('checkerboard_cols').value,
            self.get_parameter('checkerboard_rows').value
        )
        self.square_size = self.get_parameter('square_size').value
        self.num_images = self.get_parameter('num_images').value
        self.output_file = self.get_parameter('output_file').value
        self.web_port = self.get_parameter('web_port').value
        
        # Calibration data
        self.objpoints = []
        self.imgpoints = []
        self.images_captured = 0
        self.calibration_complete = False
        self.calibration_result = None
        self.img_shape = None
        
        # Current frame data
        self.current_frame = None
        self.pattern_detected = False
        self.status_message = "Waiting for checkerboard..."
        
        # Prepare object points
        self.objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 0:self.checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= self.square_size
        
        self.get_logger().info('Camera Calibrator Node Started')
        self.get_logger().info(f'Stream URL: {self.stream_url}')
        self.get_logger().info(f'Checkerboard: {self.checkerboard_size[0]}x{self.checkerboard_size[1]}')
        self.get_logger().info(f'Web interface: http://localhost:{self.web_port}')
        
        # Start Flask in separate thread
        self.app = Flask(__name__)
        self.setup_routes()
        flask_thread = Thread(target=self.run_flask, daemon=True)
        flask_thread.start()
        
        # Start frame update loop
        self.update_timer = self.create_timer(0.1, self.update_frame)
    
    def setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            return render_template_string(HTML_TEMPLATE, 
                                         checkerboard_size=f"{self.checkerboard_size[0]}x{self.checkerboard_size[1]}",
                                         target_images=self.num_images)
        
        @self.app.route('/api/frame')
        def get_frame():
            """Get current frame with checkerboard overlay."""
            if self.current_frame is not None:
                _, buffer = cv2.imencode('.jpg', self.current_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                return jsonify({
                    'frame': frame_base64,
                    'pattern_detected': self.pattern_detected,
                    'images_captured': self.images_captured,
                    'target_images': self.num_images,
                    'status': self.status_message,
                    'calibration_complete': self.calibration_complete,
                    'calibration_result': self.calibration_result
                })
            return jsonify({'error': 'No frame available'}), 503
        
        @self.app.route('/api/capture', methods=['POST'])
        def capture():
            """Capture current frame for calibration."""
            if self.pattern_detected and not self.calibration_complete:
                # Get fresh frame
                frame = self.get_frame_from_stream()
                if frame is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    ret, corners = cv2.findChessboardCorners(
                        gray, self.checkerboard_size,
                        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
                    )
                    
                    if ret:
                        corners_refined = cv2.cornerSubPix(
                            gray, corners, (11, 11), (-1, -1),
                            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                        )
                        
                        self.objpoints.append(self.objp)
                        self.imgpoints.append(corners_refined)
                        self.images_captured += 1
                        
                        self.get_logger().info(f'Image {self.images_captured}/{self.num_images} captured!')
                        
                        if self.images_captured >= self.num_images:
                            self.perform_calibration()
                        
                        return jsonify({'success': True, 'captured': self.images_captured})
            
            return jsonify({'success': False, 'message': 'No pattern detected'}), 400
        
        @self.app.route('/api/calibrate', methods=['POST'])
        def calibrate():
            """Finish calibration early."""
            if self.images_captured >= 10:
                self.perform_calibration()
                return jsonify({'success': True})
            return jsonify({'success': False, 'message': 'Need at least 10 images'}), 400
    
    def run_flask(self):
        """Run Flask app."""
        self.app.run(host='0.0.0.0', port=self.web_port, debug=False, threaded=True)
    
    def get_frame_from_stream(self):
        """Fetch a single frame from HTTP endpoint."""
        try:
            response = requests.get(self.stream_url, timeout=2)
            if response.status_code == 200:
                frame = cv2.imdecode(
                    np.frombuffer(response.content, dtype=np.uint8), 
                    cv2.IMREAD_COLOR
                )
                return frame
        except Exception as e:
            self.get_logger().error(f'Failed to get frame: {e}')
        return None
    
    def update_frame(self):
        """Update current frame with checkerboard detection."""
        if self.calibration_complete:
            return
        
        frame = self.get_frame_from_stream()
        
        if frame is None:
            self.status_message = "Cannot connect to camera stream"
            return
        
        if self.img_shape is None:
            self.img_shape = frame.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(
            gray, 
            self.checkerboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        # Draw corners
        display_frame = frame.copy()
        self.pattern_detected = ret
        
        if ret:
            cv2.drawChessboardCorners(display_frame, self.checkerboard_size, corners, ret)
            self.status_message = "Pattern detected! Click CAPTURE"
        else:
            self.status_message = "Move checkerboard into view"
        
        # Add text overlay
        cv2.putText(display_frame, f'Captured: {self.images_captured}/{self.num_images}', 
                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        self.current_frame = display_frame
    
    def perform_calibration(self):
        """Perform fisheye camera calibration."""
        self.get_logger().info('Starting calibration...')
        self.status_message = "Calibrating... please wait"
        
        calibration_flags = (
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
            cv2.fisheye.CALIB_CHECK_COND +
            cv2.fisheye.CALIB_FIX_SKEW
        )
        
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        
        objpoints_fisheye = [self.objp.reshape(1, -1, 3) for _ in range(len(self.objpoints))]
        imgpoints_fisheye = [pts.reshape(1, -1, 2) for pts in self.imgpoints]
        
        try:
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                objpoints_fisheye,
                imgpoints_fisheye,
                self.img_shape[::-1],
                K,
                D,
                flags=calibration_flags,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )
            
            self.get_logger().info(f'Calibration successful! RMS error: {rms:.4f}')
            
            calibration_data = {
                'camera_model': 'fisheye',
                'image_width': int(self.img_shape[1]),
                'image_height': int(self.img_shape[0]),
                'camera_matrix': K.tolist(),
                'distortion_coefficients': D.tolist(),
                'rms_error': float(rms)
            }
            
            with open(self.output_file, 'w') as f:
                yaml.dump(calibration_data, f)
            
            self.calibration_complete = True
            self.calibration_result = {
                'success': True,
                'rms_error': float(rms),
                'output_file': self.output_file
            }
            self.status_message = f"Calibration complete! RMS: {rms:.4f}"
            
            self.get_logger().info(f'Calibration saved to {self.output_file}')
            
        except Exception as e:
            self.get_logger().error(f'Calibration failed: {e}')
            self.calibration_result = {
                'success': False,
                'error': str(e)
            }
            self.status_message = f"Calibration failed: {e}"

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Camera Calibration</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #1a1a1a;
            color: white;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #4CAF50;
        }
        .info-panel {
            background: #2a2a2a;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .info-row {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
        }
        .video-container {
            text-align: center;
            background: #000;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }
        #camera-feed {
            max-width: 100%;
            border: 3px solid #444;
            border-radius: 5px;
        }
        .controls {
            text-align: center;
            margin: 20px 0;
        }
        button {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 15px 30px;
            font-size: 18px;
            border-radius: 5px;
            cursor: pointer;
            margin: 0 10px;
        }
        button:hover {
            background: #45a049;
        }
        button:disabled {
            background: #666;
            cursor: not-allowed;
        }
        .status {
            text-align: center;
            font-size: 24px;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .status.detected {
            background: #2d5016;
            color: #90EE90;
        }
        .status.waiting {
            background: #503016;
            color: #FFA500;
        }
        .status.complete {
            background: #165050;
            color: #00FFFF;
        }
        .progress {
            font-size: 32px;
            font-weight: bold;
            color: #4CAF50;
        }
        .result {
            background: #2a2a2a;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Camera Calibration Tool</h1>
        
        <div class="info-panel">
            <div class="info-row">
                <span><strong>Checkerboard Size:</strong> {{ checkerboard_size }}</span>
                <span><strong>Target Images:</strong> {{ target_images }}</span>
            </div>
            <div class="info-row">
                <span class="progress">Progress: <span id="progress">0/{{ target_images }}</span></span>
            </div>
        </div>
        
        <div id="status-message" class="status waiting">
            Waiting for checkerboard...
        </div>
        
        <div class="video-container">
            <img id="camera-feed" src="" alt="Camera Feed">
        </div>
        
        <div class="controls">
            <button id="capture-btn" onclick="captureImage()" disabled>
                📸 CAPTURE IMAGE
            </button>
            <button id="finish-btn" onclick="finishCalibration()">
                ✅ FINISH CALIBRATION (min 10 images)
            </button>
        </div>
        
        <div id="result" class="result" style="display: none;">
            <h2>Calibration Results</h2>
            <p id="result-text"></p>
        </div>
    </div>
    
    <script>
        let patternDetected = false;
        let calibrationComplete = false;
        
        function updateFrame() {
            if (calibrationComplete) return;
            
            fetch('/api/frame')
                .then(response => response.json())
                .then(data => {
                    if (data.frame) {
                        document.getElementById('camera-feed').src = 'data:image/jpeg;base64,' + data.frame;
                        patternDetected = data.pattern_detected;
                        
                        // Update progress
                        document.getElementById('progress').textContent = 
                            data.images_captured + '/' + data.target_images;
                        
                        // Update status
                        const statusDiv = document.getElementById('status-message');
                        statusDiv.textContent = data.status;
                        
                        if (data.calibration_complete) {
                            statusDiv.className = 'status complete';
                            calibrationComplete = true;
                            showResults(data.calibration_result);
                        } else if (data.pattern_detected) {
                            statusDiv.className = 'status detected';
                        } else {
                            statusDiv.className = 'status waiting';
                        }
                        
                        // Enable/disable capture button
                        document.getElementById('capture-btn').disabled = !patternDetected || calibrationComplete;
                    }
                })
                .catch(error => console.error('Error:', error));
        }
        
        function captureImage() {
            fetch('/api/capture', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        console.log('Image captured:', data.captured);
                    }
                })
                .catch(error => console.error('Error:', error));
        }
        
        function finishCalibration() {
            if (confirm('Finish calibration now? (Need at least 10 images)')) {
                fetch('/api/calibrate', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        if (!data.success) {
                            alert(data.message);
                        }
                    })
                    .catch(error => console.error('Error:', error));
            }
        }
        
        function showResults(result) {
            const resultDiv = document.getElementById('result');
            const resultText = document.getElementById('result-text');
            
            if (result.success) {
                resultText.innerHTML = `
                    <strong>✅ Calibration Successful!</strong><br>
                    <strong>RMS Error:</strong> ${result.rms_error.toFixed(4)}<br>
                    <strong>Output File:</strong> ${result.output_file}<br>
                    <br>
                    You can now run the camera_calibrated_node to publish undistorted images.
                `;
            } else {
                resultText.innerHTML = `
                    <strong>❌ Calibration Failed</strong><br>
                    <strong>Error:</strong> ${result.error}
                `;
            }
            
            resultDiv.style.display = 'block';
            document.getElementById('capture-btn').disabled = true;
            document.getElementById('finish-btn').disabled = true;
        }
        
        // Update frame every 100ms
        setInterval(updateFrame, 100);
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(event) {
            if (event.code === 'Space' && patternDetected && !calibrationComplete) {
                event.preventDefault();
                captureImage();
            }
        });
    </script>
</body>
</html>
'''

def main(args=None):
    rclpy.init(args=args)
    node = CameraCalibrator()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()