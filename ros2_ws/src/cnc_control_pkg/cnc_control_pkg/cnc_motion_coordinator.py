#!/usr/bin/env python3
"""
CNC Motion Coordinator Node

A ROS2 node that provides a web interface for CNC machine control.
Features:
- Send and receive CNC commands (GRBL protocol)
- Track machine position relative to user-defined origin
- Home position (0,0,0) + 4 preset waypoints
- Web GUI for monitoring and control
- Display detected objects from camera
- Gripper-to-camera transformation for pick-and-place

Author: [Your Name]
Date: 2026-01-09
"""

import re
import threading
import time
import json
import math
import numpy as np
from typing import Dict, Optional, List

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from flask import Flask, render_template_string, request, jsonify
from cv_bridge import CvBridge
import cv2
import base64
from std_msgs.msg import String, Int32 




# ============================================================================
# HTML Template for Web Interface
# ============================================================================
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>CNC Motion Coordinator</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 1400px; 
            margin: 20px auto; 
            padding: 0 20px;
            background-color: #f0f0f0;
        }
        h1 { color: #333; text-align: center; }
        .container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .section { 
            background: white; 
            padding: 20px; 
            margin: 15px 0; 
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .full-width {
            grid-column: 1 / -1;
        }
        .position { 
            font-size: 28px; 
            font-weight: bold; 
            margin: 15px 0;
            text-align: center;
            color: #007bff;
        }
        button { 
            padding: 10px 20px; 
            margin: 5px; 
            cursor: pointer;
            border-radius: 5px;
            font-size: 14px;
            transition: all 0.3s;
        }
        button:hover { opacity: 0.8; }
        .preset { 
            display: flex; 
            align-items: center; 
            gap: 10px; 
            margin: 10px 0;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
        }
        .preset input { 
            width: 80px; 
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .preset strong { 
            min-width: 60px;
            color: #555;
        }
        .btn-primary { background: #007bff; color: white; border: none; }
        .btn-success { background: #28a745; color: white; border: none; }
        .btn-warning { background: #ffc107; border: none; }
        .btn-danger { background: #dc3545; color: white; border: none; }
        .btn-home { background: #17a2b8; color: white; border: none; font-weight: bold; }
        .btn-catch { background: #ff6b6b; color: white; border: none; font-weight: bold; font-size: 18px; padding: 15px 30px; }
        .btn-place { background: #28a745; color: white; border: none; font-weight: bold; font-size: 18px; padding: 15px 30px; }
        .home-section {
            text-align: center;
            padding: 15px;
            background: #e7f3ff;
            border-radius: 5px;
            margin-bottom: 15px;
        }
        .camera-view {
            width: 100%;
            max-width: 640px;
            border: 2px solid #ddd;
            border-radius: 8px;
            margin: 10px auto;
            display: block;
        }
        .object-list {
            max-height: 400px;
            overflow-y: auto;
        }
        .object-item {
            background: #f8f9fa;
            padding: 10px;
            margin: 8px 0;
            border-radius: 5px;
            border-left: 4px solid #007bff;
        }
        .object-item h4 {
            margin: 0 0 8px 0;
            color: #333;
        }
        .object-item p {
            margin: 4px 0;
            font-size: 13px;
            color: #666;
        }
        .transform-info {
            background: #fff3cd;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            font-size: 13px;
        }
        .catch-section {
            text-align: center;
            padding: 20px;
            background: #ffe7e7;
            border-radius: 8px;
            margin: 15px 0;
        }
        .place-section {
            text-align: center;
            padding: 20px;
            background: #e7ffe7;
            border-radius: 8px;
            margin: 15px 0;
        }
    </style>
</head>
<body>
    <h1>🔧 CNC Motion Coordinator</h1>
  
    <div class="container">
        <!-- Left Column: CNC Control -->
        <div>
            <!-- Current Position Display -->
            <div class="section">
                <h2>📍 Current Position (Relative to Origin)</h2>
                <div class="position">
                    X: <span id="pos-x">0.000</span> mm | 
                    Y: <span id="pos-y">0.000</span> mm | 
                    Z: <span id="pos-z">0.000</span> mm
                </div>
                <div style="text-align: center;">
                    <button class="btn-warning" onclick="setOrigin()">
                        🎯 Set Current Position as Origin
                    </button>
                </div>
            </div>
          
            <!-- Waypoint Control -->
            <div class="section">
                <h2>📌 Waypoint Management</h2>
              
                <!-- Home Position (Always 0,0,0) -->
                <div class="home-section">
                    <h3 style="margin-top: 0;">🏠 Home Position</h3>
                    <div style="font-size: 18px; margin: 10px 0;">
                        X: 0.000 | Y: 0.000 | Z: 0.000
                    </div>
                    <button class="btn-home" onclick="goToHome()">
                        Go To Home
                    </button>
                </div>
              
                <!-- Preset Waypoints P1-P4 -->
                <h3>Custom Waypoints</h3>
                {% for i in range(4) %}
                <div class="preset">
                    <strong>P{{i+1}}:</strong>
                    <label>X:</label>
                    <input type="number" step="0.1" id="preset{{i}}-x" placeholder="0.0" value="0">
                    <label>Y:</label>
                    <input type="number" step="0.1" id="preset{{i}}-y" placeholder="0.0" value="0">
                    <label>Z:</label>
                    <input type="number" step="0.1" id="preset{{i}}-z" placeholder="0.0" value="0">
                    <button class="btn-primary" onclick="saveCurrent({{i}})">
                        💾 Save Current
                    </button>
                    <button class="btn-success" onclick="goTo({{i}})">
                        ➡️ Go To P{{i+1}}
                    </button>
                </div>
                {% endfor %}
            </div>

            <!-- Gripper Transform Info -->
            <div class="section">
                <h2>🤖 Gripper Offset</h2>
                <div class="transform-info">
                    <strong>Camera → Gripper Offset:</strong><br>
                    X: <span id="tf-x">0</span> mm | 
                    Y: <span id="tf-y">0</span> mm | 
                    Z: <span id="tf-z">0</span> mm
                </div>
            </div>
        </div>

        <!-- Right Column: Camera & Objects -->
        <div>
            <!-- Camera View -->
            <div class="section">
                <h2>📷 Camera View</h2>
                <img id="camera-image" class="camera-view" src="" alt="Camera feed loading...">
            </div>

            <!-- Catch Button -->
            <div class="catch-section">
                <button class="btn-catch" onclick="catchObject()">
                    🦾 CATCH!
                </button>
            </div>

            <!-- Place Button -->
            <div class="place-section">
                <button class="btn-place" onclick="placeObject()">
                    📦 PLACE!
                </button>
            </div>

            <!-- Detected Objects -->
            <div class="section">
                <h2>🎯 Detected Objects</h2>
                <div id="object-list" class="object-list">
                    <p style="text-align: center; color: #999;">No objects detected</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Update position and presets from server
        function updatePosition() {
            fetch('/api/position')
                .then(r => r.json())
                .then(data => {
                    // Update current position display
                    document.getElementById('pos-x').textContent = data.x.toFixed(3);
                    document.getElementById('pos-y').textContent = data.y.toFixed(3);
                    document.getElementById('pos-z').textContent = data.z.toFixed(3);
                  
                    // Update preset values
                    for (let i = 0; i < 4; i++) {
                        document.getElementById('preset'+i+'-x').value = data.presets[i].x.toFixed(3);
                        document.getElementById('preset'+i+'-y').value = data.presets[i].y.toFixed(3);
                        document.getElementById('preset'+i+'-z').value = data.presets[i].z.toFixed(3);
                    }
                })
                .catch(err => console.error('Error fetching position:', err));
        }

        // Update camera image
        function updateCamera() {
            fetch('/api/camera_image')
                .then(r => r.json())
                .then(data => {
                    if (data.image) {
                        document.getElementById('camera-image').src = 'data:image/jpeg;base64,' + data.image;
                    }
                })
                .catch(err => console.error('Error fetching camera:', err));
        }

        // Update detected objects
        function updateObjects() {
            fetch('/api/objects')
                .then(r => r.json())
                .then(data => {
                    const listDiv = document.getElementById('object-list');
                  
                    if (data.objects && data.objects.length > 0) {
                        let html = '';
                        data.objects.forEach((obj, idx) => {
                            html += `
                                <div class="object-item">
                                    <h4>${obj.name || 'Object #' + obj.object_id}</h4>
                                    <p><strong>Type:</strong> ${obj.classification?.type || 'N/A'}</p>
                                    <p><strong>Body Diameter:</strong> ${obj.body_diameter_mm?.toFixed(2) || 'N/A'} mm</p>
                                    <p><strong>Body Length:</strong> ${obj.body_length_mm?.toFixed(2) || 'N/A'} mm</p>
                                    <p><strong>Pickup Point (Rel):</strong> (${obj.pickup_point_relative_to_camera_mm?.[0]?.toFixed(1) || 'N/A'}, ${obj.pickup_point_relative_to_camera_mm?.[1]?.toFixed(1) || 'N/A'}) mm</p>
                                    <p><strong>Jagginess:</strong> ${obj.jagginess?.overall_px?.toFixed(3) || 'N/A'}</p>
                                </div>
                            `;
                        });
                        listDiv.innerHTML = html;
                    } else {
                        listDiv.innerHTML = '<p style="text-align: center; color: #999;">No objects detected</p>';
                    }
                })
                .catch(err => console.error('Error fetching objects:', err));
        }

        // Update gripper transform display
        function updateTransform() {
            fetch('/api/gripper_transform')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('tf-x').textContent = data.offset_x.toFixed(1);
                    document.getElementById('tf-y').textContent = data.offset_y.toFixed(1);
                    document.getElementById('tf-z').textContent = data.offset_z.toFixed(1);
                })
                .catch(err => console.error('Error fetching transform:', err));
        }
      
        // Set current position as origin (zero point)
        function setOrigin() {
            fetch('/api/set_origin', {method: 'POST'})
                .then(() => {
                    updatePosition();
                })
                .catch(err => console.error('Error setting origin:', err));
        }
      
        // Save current position to preset slot
        function saveCurrent(idx) {
            fetch('/api/save_preset/' + idx, {method: 'POST'})
                .then(() => {
                    updatePosition();
                })
                .catch(err => console.error('Error saving preset:', err));
        }
      
        // Go to home position (0,0,0)
        function goToHome() {
            fetch('/api/goto', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({x: 0, y: 0, z: 0})
            })
            .catch(err => console.error('Error going to home:', err));
        }
      
        // Go to preset position
        function goTo(idx) {
            const x = parseFloat(document.getElementById('preset'+idx+'-x').value);
            const y = parseFloat(document.getElementById('preset'+idx+'-y').value);
            const z = parseFloat(document.getElementById('preset'+idx+'-z').value);
          
            fetch('/api/goto', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({x: x, y: y, z: z})
            })
            .catch(err => console.error('Error going to preset:', err));
        }

        // Catch object
        function catchObject() {
            fetch('/api/catch', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    console.log('Catch executed:', data.message);
                    alert('Catch: ' + data.message);
                })
                .catch(err => console.error('Error catching:', err));
        }

        // Place object
        function placeObject() {
            fetch('/api/place', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    console.log('Place executed:', data);
                })
                .catch(err => console.error('Error placing:', err));
        }
      
        // Auto-update all data
        setInterval(updatePosition, 500);
        setInterval(updateCamera, 200);  // 5 FPS
        setInterval(updateObjects, 500);
        setInterval(updateTransform, 2000);
      
        // Initial updates
        updatePosition();
        updateCamera();
        updateObjects();
        updateTransform();
    </script>
</body>
</html>
'''


# ============================================================================
# CNC Motion Coordinator Node
# ============================================================================
class CncMotionCoordinator(Node):
    """
    Main ROS2 node for CNC motion coordination.
  
    Responsibilities:
    - Communicate with CNC controller via serial topics
    - Track machine position and user-defined origin
    - Manage waypoints (Home + 4 presets)
    - Provide web interface for monitoring and control
    - Display camera feed and detected objects
    - Handle gripper-to-camera offset for pick-and-place
    """
  
    def __init__(self):
        super().__init__('cnc_motion_coordinator')
      
        # ====================================================================
        # ROS2 Parameters
        # ====================================================================
        self.declare_parameter('feed_rate', 2500)           # mm/min
        self.declare_parameter('web_port', 8002)            # Web server port
        self.declare_parameter('position_poll_rate', 2.0)   # Hz
      
        # Gripper-to-camera offset parameters
        self.declare_parameter('gripper_offset_x', -245.0)  # mm (gripper offset in X)
        self.declare_parameter('gripper_offset_y', 0.0)     # mm (gripper offset in Y)
        self.declare_parameter('gripper_offset_z', -0.0)    # mm (gripper offset in Z)

        # Servo control parameters
        self.declare_parameter('servo_wrist_horizontal', 0)    # Wrist horizontal position (degrees)
        self.declare_parameter('servo_gripper_open', 0)        # Gripper fully open (degrees)
        self.declare_parameter('servo_gripper_closed', 180)    # Gripper closed (degrees)
        self.declare_parameter('pick_height_offset', 10.0)     # Height above object before descending (mm)
        self.declare_parameter('pick_descent_speed', 500)      # Speed for Z descent (mm/min)
        self.declare_parameter('motion_delay', 4)              # Delay between motion steps (seconds)
        self.declare_parameter('pick_down_mm', 10.0)           # How much to descend in Z during pick (mm)
        self.declare_parameter('safe_plane_z', 70.0)           # Safe Z height after pick (mm, relative to origin)
        self.declare_parameter('place_delay', 15.0)            # Delay before releasing object during place (seconds)
      
        self.feed_rate = self.get_parameter('feed_rate').value
        self.web_port = self.get_parameter('web_port').value
        self.position_poll_rate = self.get_parameter('position_poll_rate').value
      
        # Gripper offset
        self.gripper_offset_x = self.get_parameter('gripper_offset_x').value
        self.gripper_offset_y = self.get_parameter('gripper_offset_y').value
        self.gripper_offset_z = self.get_parameter('gripper_offset_z').value

        # Servo and motion parameters
        self.servo_wrist_horizontal = self.get_parameter('servo_wrist_horizontal').value
        self.servo_gripper_open = self.get_parameter('servo_gripper_open').value
        self.servo_gripper_closed = self.get_parameter('servo_gripper_closed').value
        self.pick_height_offset = self.get_parameter('pick_height_offset').value
        self.pick_descent_speed = self.get_parameter('pick_descent_speed').value
        self.motion_delay = self.get_parameter('motion_delay').value
        self.pick_down_mm = self.get_parameter('pick_down_mm').value
        self.safe_plane_z = self.get_parameter('safe_plane_z').value
        self.place_delay = self.get_parameter('place_delay').value
      
        # ====================================================================
        # ROS2 Publishers & Subscribers
        # ====================================================================
        # Publisher: Send commands to CNC controller
        self.pub_cnc_cmd = self.create_publisher(
            String, 
            '/serial_cnc_in', 
            10
        )

        # Publisher: Control wrist servo (publishes to /servo2 topic)
        self.pub_wrist_servo = self.create_publisher(
            Int32,
            '/servo2',
            10
        )
      
        # Publisher: Control gripper servo (publishes to /servo1 topic)
        self.pub_gripper_servo = self.create_publisher(
            Int32,
            '/servo1',
            10
        )
      
        # Subscriber: Receive status/feedback from CNC controller
        self.sub_cnc_status = self.create_subscription(
            String,
            '/serial_cnc_out',
            self.cnc_status_callback,
            10
        )
      
        # Subscriber: Camera image with annotations
        self.sub_camera_image = self.create_subscription(
            Image,
            '/camera/physical_features',
            self.camera_image_callback,
            10
        )
      
        # Subscriber: Object physical features (JSON)
        self.sub_object_features = self.create_subscription(
            String,
            '/camera/object_physical_features',
            self.object_features_callback,
            10
        )
      
        # ====================================================================
        # State Variables
        # ====================================================================
        # Machine position in absolute coordinates (from CNC controller)
        self.machine_x = 0.0
        self.machine_y = 0.0
        self.machine_z = 0.0
      
        # User-defined origin point (for relative positioning)
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.origin_z = 0.0

        # GRBL state tracking (Idle, Run, Hold, Alarm, etc.)
        self.grbl_state = "Unknown"
        self.grbl_state_lock = threading.Lock()
      
        # Preset waypoints (P1-P4) - stored as relative coordinates
        self.presets = [
            {'x': 0.0, 'y': 0.0, 'z': 0.0},
            {'x': 0.0, 'y': 0.0, 'z': 0.0},
            {'x': 0.0, 'y': 0.0, 'z': 0.0},
            {'x': 0.0, 'y': 0.0, 'z': 0.0}
        ]
      
        # Camera and object data
        self.latest_camera_image = None
        self.latest_objects = []
        self.bridge = CvBridge()
      
        # Thread safety locks
        self.position_lock = threading.Lock()
        self.camera_lock = threading.Lock()
        self.objects_lock = threading.Lock()
        self.catch_lock = threading.Lock()       
        self.catch_in_progress = False 
      
        # ====================================================================
        # Timers
        # ====================================================================
        # Periodic position polling
        self.position_timer = self.create_timer(
            1.0 / self.position_poll_rate,
            self.poll_cnc_position
        )
      
        # ====================================================================
        # Web Server
        # ====================================================================
        self.start_web_server()
      
        # ====================================================================
        # Initialization
        # ====================================================================
        self.get_logger().info('=' * 60)
        self.get_logger().info('CNC Motion Coordinator Node Started')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'Feed Rate: {self.feed_rate} mm/min')
        self.get_logger().info(f'Position Poll Rate: {self.position_poll_rate} Hz')
        self.get_logger().info(f'Pick Down Distance: {self.pick_down_mm} mm')
        self.get_logger().info(f'Safe Plane Z Height: {self.safe_plane_z} mm')
        self.get_logger().info(f'Place Delay: {self.place_delay} seconds')
        self.get_logger().info(f'Web Interface: http://localhost:{self.web_port}')
        self.get_logger().info('-' * 60)
        self.get_logger().info('Gripper Offset:')
        self.get_logger().info(f'  Offset X: {self.gripper_offset_x} mm')
        self.get_logger().info(f'  Offset Y: {self.gripper_offset_y} mm')
        self.get_logger().info(f'  Offset Z: {self.gripper_offset_z} mm')
        self.get_logger().info('=' * 60)
      
        # Send GRBL unlock command after 5 seconds
        threading.Timer(5.0, self.send_grbl_unlock).start()
  
    # ========================================================================
    # CNC Communication Methods
    # ========================================================================
  
    def send_cnc_command(self, command: str, log: bool = True) -> None:
        """
        Send a command to the CNC controller.
      
        Args:
            command: GRBL command string (e.g., "G0 X10 Y20")
            log: Whether to log this command (default True)
        """
        msg = String()
        msg.data = command if command.endswith('\n') else command + '\n'
        self.pub_cnc_cmd.publish(msg)
        # Only log if requested and not a position query
        if log and command.strip() != '?':
            self.get_logger().info(f'📤 Sent CNC command: {command.strip()}')
  
    def cnc_status_callback(self, msg: String) -> None:
        """
        Process status messages from CNC controller.
      
        Parses position data and machine state from GRBL status reports.
        Format: <Idle|MPos:10.000,20.000,5.000|...>
                <Run|MPos:10.000,20.000,5.000|...>
      
        Args:
            msg: Status message from CNC controller
        """
        # Parse machine state (Idle, Run, Hold, Alarm, etc.)
        state_match = re.search(r'<([A-Za-z]+)\|', msg.data)
        if state_match:
            with self.grbl_state_lock:
                self.grbl_state = state_match.group(1)
      
        # Parse machine position from status report
        match = re.search(r'MPos:([-\d\.]+),([-\d\.]+),([-\d\.]+)', msg.data)
        if match:
            with self.position_lock:
                self.machine_x = float(match.group(1))
                self.machine_y = float(match.group(2))
                self.machine_z = float(match.group(3))

    def wait_until_idle(self, timeout_sec: float = 15.0, check_interval: float = 0.05) -> bool:
        """
        Wait until GRBL reports 'Idle' state (motion complete).
      
        Args:
            timeout_sec: Maximum time to wait (seconds)
            check_interval: How often to check state (seconds)
          
        Returns:
            True if machine reached Idle, False if timeout
        """
        start_time = time.time()
      
        while (time.time() - start_time) < timeout_sec:
            with self.grbl_state_lock:
                current_state = self.grbl_state
          
            # Check if machine is idle
            if current_state == "Idle":
                self.get_logger().info(f'✓ Machine idle (waited {time.time() - start_time:.2f}s)')
                return True
          
            # Log if we're still running
            if current_state == "Run":
                self.get_logger().debug(f'⏳ Waiting for motion to complete... ({current_state})')
          
            time.sleep(check_interval)
      
        # Timeout reached
        with self.grbl_state_lock:
            final_state = self.grbl_state
      
        self.get_logger().error(
            f'❌ Timeout waiting for Idle state! '
            f'Current state: {final_state} (waited {timeout_sec}s)'
        )
        return False

    def wait_for_motion_complete(self, timeout_sec: float = 20.0, poll_interval: float = 0.1) -> bool:
        """
        Robust wait: forces GRBL status updates and requires seeing non-Idle before accepting Idle.
        Prevents 'stale Idle' causing instant return.
        
        Args:
            timeout_sec: Maximum time to wait (seconds)
            poll_interval: How often to poll status (seconds)
          
        Returns:
            True if motion completed (saw non-Idle then Idle), False if timeout
        """
        start = time.time()
        saw_non_idle = False

        while (time.time() - start) < timeout_sec:
            # Force a fresh GRBL status response
            self.send_cnc_command('?', log=False)

            with self.grbl_state_lock:
                state = self.grbl_state

            if state != "Idle":
                saw_non_idle = True

            if saw_non_idle and state == "Idle":
                self.get_logger().info(f'✓ Motion complete (waited {time.time() - start:.2f}s)')
                return True

            time.sleep(poll_interval)

        self.get_logger().error(f'❌ Timeout waiting motion complete (state={state}, saw_non_idle={saw_non_idle})')
        return False

    def wait_for_servo(self, delay_sec: float = 0.5) -> None:
        """
        Wait for servo to reach target position.
      
        Servos don't report completion, so we use a fixed delay.
      
        Args:
            delay_sec: Time to wait for servo movement (seconds)
        """
        self.get_logger().debug(f'⏳ Waiting {delay_sec}s for servo...')
        time.sleep(delay_sec)

    def poll_cnc_position(self) -> None:
        """
        Request current position from CNC controller.
      
        Sends '?' query command to GRBL.
        Called periodically by timer.
        """
        self.send_cnc_command('?', log=False)  # Don't log position queries
  
    def send_grbl_unlock(self) -> None:
        """
        Send unlock command to GRBL controller.
      
        The $X command clears alarm states and allows motion.
        Called once at startup after 5 second delay.
        """
        self.send_cnc_command('$X')
        self.get_logger().info('🔓 GRBL unlock command sent ($X)')
  
    # ========================================================================
    # Camera & Object Callbacks
    # ========================================================================
  
    def camera_image_callback(self, msg: Image) -> None:
        """
        Process camera image with annotations.
      
        Args:
            msg: Image message from /camera/physical_features
        """
        try:
            with self.camera_lock:
                self.latest_camera_image = msg
        except Exception as e:
            self.get_logger().error(f'Error processing camera image: {e}')
  
    def object_features_callback(self, msg: String) -> None:
        """
        Process detected object features (JSON).
      
        Args:
            msg: JSON string from /camera/object_physical_features
        """
        try:
            data = json.loads(msg.data)
            with self.objects_lock:
                self.latest_objects = data.get('objects', [])
        except Exception as e:
            self.get_logger().error(f'Error parsing object features: {e}')
  
    # ========================================================================
    # Position & Waypoint Management
    # ========================================================================
  
    def get_relative_position(self) -> Dict[str, float]:
        """
        Get current position relative to user-defined origin.
      
        Returns:
            Dictionary with 'x', 'y', 'z' keys (in mm)
        """
        with self.position_lock:
            return {
                'x': self.machine_x - self.origin_x,
                'y': self.machine_y - self.origin_y,
                'z': self.machine_z - self.origin_z
            }
  
    def set_origin(self) -> None:
        """
        Set current machine position as the origin (0,0,0).
      
        All subsequent relative positions will be calculated from this point.
        """
        with self.position_lock:
            self.origin_x = self.machine_x
            self.origin_y = self.machine_y
            self.origin_z = self.machine_z
  
    def save_preset(self, preset_index: int) -> None:
        """
        Save current relative position to a preset slot.
      
        Args:
            preset_index: Preset slot number (0-3 for P1-P4)
        """
        if not (0 <= preset_index < 4):
            self.get_logger().error(f'Invalid preset index: {preset_index}')
            return
      
        rel_pos = self.get_relative_position()
        self.presets[preset_index] = rel_pos.copy()
  
    def goto_position(self, rel_x: float, rel_y: float, rel_z: float) -> None:
        """
        Move CNC to specified position (relative to origin) using relative movement.
      
        Args:
            rel_x: Target X position relative to origin (mm)
            rel_y: Target Y position relative to origin (mm)
            rel_z: Target Z position relative to origin (mm)
        """
        # Get current relative position
        current_rel = self.get_relative_position()
      
        # Calculate the delta (difference between target and current)
        delta_x = rel_x - current_rel['x']
        delta_y = rel_y - current_rel['y']
        delta_z = rel_z - current_rel['z']
      
        # Send G-code command (G91 = relative positioning, G1 = linear move)
        cmd = f'G91 G1 X{delta_x:.4f} Y{delta_y:.4f} Z{delta_z:.4f} F{self.feed_rate}'
        self.send_cnc_command(cmd)
      
        # Switch back to absolute mode for safety
        self.send_cnc_command('G90')

    # ========================================================================
    # Servo Control Methods
    # ========================================================================

    def set_wrist_angle(self, angle_degrees: int) -> None:
        """
        Set wrist servo to specified angle.
      
        Args:
            angle_degrees: Target angle in degrees (0-180)
        """
        msg = Int32()
        msg.data = int(angle_degrees)
        self.pub_wrist_servo.publish(msg)
        self.get_logger().info(f'🔄 Wrist servo set to {angle_degrees}°')

    def set_gripper(self, angle_degrees: int) -> None:
        """
        Set gripper servo to specified angle.
      
        Args:
            angle_degrees: Target angle in degrees (0-180)
                        0 = fully open, 180 = closed
        """
        msg = Int32()
        msg.data = int(angle_degrees)
        self.pub_gripper_servo.publish(msg)
      
        state = "OPEN" if angle_degrees <= 10 else "CLOSED"
        self.get_logger().info(f'🤏 Gripper set to {angle_degrees}° ({state})')

    def open_gripper(self) -> None:
        """Open the gripper fully."""
        self.set_gripper(self.servo_gripper_open)

    def close_gripper(self) -> None:
        """Close the gripper."""
        self.set_gripper(self.servo_gripper_closed)

    def compute_object_orientation(self, obj: dict) -> Optional[float]:
        """
        Compute the orientation angle of the object's principal axis.
      
        Args:
            obj: Object dictionary containing moment_vector
          
        Returns:
            Orientation angle in degrees (0-180), or None if unavailable
        """
        moment_vector = obj.get("moment_vector")
        if not moment_vector:
            self.get_logger().warn('Object has no moment_vector')
            return None
      
        vx = moment_vector.get("vx", 1.0)
        vy = moment_vector.get("vy", 0.0)
      
        # Calculate angle from horizontal (in degrees)
        angle_rad = math.atan2(vy, vx)
        angle_deg = math.degrees(angle_rad)
      
        # Normalize to 0-180 range (since wrist servo is 0-180)
        # Add 90 to convert from [-90, 90] to [0, 180]
        wrist_angle = (angle_deg + 90) % 180
      
        self.get_logger().info(
            f'📐 Object orientation: vx={vx:.3f}, vy={vy:.3f} → '
            f'angle={angle_deg:.1f}° → wrist={wrist_angle:.1f}°'
        )
      
        return wrist_angle

    def goto_position_xy_only(self, rel_x: float, rel_y: float) -> None:
        """
        Move CNC to specified XY position (relative to origin), keeping Z unchanged.
      
        Args:
            rel_x: Target X position relative to origin (mm)
            rel_y: Target Y position relative to origin (mm)
        """
        # Get current relative position
        current_rel = self.get_relative_position()
      
        # Calculate the delta (difference between target and current)
        delta_x = rel_x - current_rel['x']
        delta_y = rel_y - current_rel['y']
      
        # Send G-code command (G91 = relative positioning, G1 = linear move)
        # Only move in XY plane
        cmd = f'G91 G1 X{delta_x:.4f} Y{delta_y:.4f} F{self.feed_rate}'
        self.send_cnc_command(cmd)
      
        # Switch back to absolute mode for safety
        self.send_cnc_command('G90')
      
        self.get_logger().info(
            f'🚀 Moving in XY plane: X={rel_x:.3f} Y={rel_y:.3f} | '
            f'Delta: X={delta_x:.3f} Y={delta_y:.3f}'
        )

    def goto_position_z_only(self, rel_z: float, feed_rate: Optional[int] = None) -> None:
        """
        Move CNC to specified Z position (relative to origin), keeping XY unchanged.
      
        Args:
            rel_z: Target Z position relative to origin (mm)
            feed_rate: Optional custom feed rate (mm/min), uses default if None
        """
        # Get current relative position
        current_rel = self.get_relative_position()
      
        # Calculate the delta
        delta_z = rel_z - current_rel['z']
      
        # Use custom feed rate or default
        speed = feed_rate if feed_rate is not None else self.feed_rate
      
        # Send G-code command
        cmd = f'G91 G1 Z{delta_z:.4f} F{speed}'
        self.send_cnc_command(cmd)
      
        # Switch back to absolute mode
        self.send_cnc_command('G90')
      
        self.get_logger().info(
            f'⬇️ Moving in Z axis: Z={rel_z:.3f} | Delta: Z={delta_z:.3f} @ {speed} mm/min'
        )
  
    # ========================================================================
    # Gripper Pick Target Calculation
    # ========================================================================
  
    def compute_pick_target_rel(self, pickup_xy_mm: List[float]) -> Dict[str, float]:
        """
        Convert an object's pickup point (camera frame, mm) into a CNC target position
        in *relative-to-origin* coordinates.
      
        Simply subtracts the pickup point from current position and adds gripper offsets.

        Args:
            pickup_xy_mm: [x_mm, y_mm] in camera frame (relative to camera center)
          
        Returns:
            Dictionary with 'x', 'y', 'z' keys (target position relative to origin)
        """
        if not (isinstance(pickup_xy_mm, (list, tuple)) and len(pickup_xy_mm) >= 2):
            raise ValueError(f"Invalid pickup_point_relative_to_camera_mm: {pickup_xy_mm}")

        # Get pickup point coordinates (relative to camera center)
        pickup_x = float(pickup_xy_mm[0])
        pickup_y = float(pickup_xy_mm[1])

        # Current CNC position (relative to origin)
        cur = self.get_relative_position()

        # Target = current + gripper_offset - pickup (with axis swap)
        target_x = cur["x"] + self.gripper_offset_x + pickup_y
        target_y = cur["y"] + self.gripper_offset_y + pickup_x
        target_z = cur["z"] + self.gripper_offset_z

        # Print detailed computation
        self.get_logger().info('=' * 70)
        self.get_logger().info('🎯 PICK TARGET COMPUTATION')
        self.get_logger().info('=' * 70)
        self.get_logger().info(f'📍 Current Position (relative to origin):')
        self.get_logger().info(f'   X = {cur["x"]:.3f} mm')
        self.get_logger().info(f'   Y = {cur["y"]:.3f} mm')
        self.get_logger().info(f'   Z = {cur["z"]:.3f} mm')
        self.get_logger().info('-' * 70)
        self.get_logger().info(f'📷 Pickup Point (camera frame, relative to center):')
        self.get_logger().info(f'   pickup_x = {pickup_x:.3f} mm')
        self.get_logger().info(f'   pickup_y = {pickup_y:.3f} mm')
        self.get_logger().info('-' * 70)
        self.get_logger().info(f'🤖 Gripper Offsets:')
        self.get_logger().info(f'   offset_x = {self.gripper_offset_x:.3f} mm')
        self.get_logger().info(f'   offset_y = {self.gripper_offset_y:.3f} mm')
        self.get_logger().info(f'   offset_z = {self.gripper_offset_z:.3f} mm')
        self.get_logger().info('-' * 70)
        self.get_logger().info(f'🧮 Computation (with axis swap):')
        self.get_logger().info(f'   target_x = cur_x + offset_x + pickup_y')
        self.get_logger().info(f'   target_x = {cur["x"]:.3f} + {self.gripper_offset_x:.3f} + {pickup_y:.3f} = {target_x:.3f}')
        self.get_logger().info(f'   target_y = cur_y + offset_y + pickup_x')
        self.get_logger().info(f'   target_y = {cur["y"]:.3f} + {self.gripper_offset_y:.3f} + {pickup_x:.3f} = {target_y:.3f}')
        self.get_logger().info(f'   target_z = cur_z + offset_z')
        self.get_logger().info(f'   target_z = {cur["z"]:.3f} + {self.gripper_offset_z:.3f} = {target_z:.3f}')
        self.get_logger().info('-' * 70)
        self.get_logger().info(f'✅ Final Target Position:')
        self.get_logger().info(f'   X = {target_x:.3f} mm')
        self.get_logger().info(f'   Y = {target_y:.3f} mm')
        self.get_logger().info(f'   Z = {target_z:.3f} mm')
        self.get_logger().info('=' * 70)

        return {
            "x": target_x,
            "y": target_y,
            "z": target_z,
        }
  
    def catch_object(self) -> str:
        """
        Execute a complete pick operation:
        1. Move to XY position above the object
        2. Open gripper
        3. Rotate wrist to match object orientation
        4. Descend in Z by fixed offset
        5. Close gripper
        6. Ascend to safe plane height
      
        Each motion step waits for completion before proceeding.
        Thread-safe: Only one catch operation can run at a time.
      
        Returns:
            Status message string
        """
        # ========================================================================
        # THREAD SAFETY: Check if catch is already in progress
        # ========================================================================
        with self.catch_lock:
            if self.catch_in_progress:
                self.get_logger().warn('⚠️  Catch already in progress, ignoring request')
                return 'Catch already in progress'
            self.catch_in_progress = True
      
        try:
            self.get_logger().info('=' * 70)
            self.get_logger().info('🦾 CATCH SEQUENCE INITIATED')
            self.get_logger().info('=' * 70)
          
            # ========================================================================
            # STEP 1: Validate object detection
            # ========================================================================
            with self.objects_lock:
                if not self.latest_objects:
                    self.get_logger().warn('❌ No objects detected to catch')
                    return 'No objects detected'
              
                obj = self.latest_objects[0]
          
            obj_name = obj.get('name', 'Unknown')
            self.get_logger().info(f'🎯 Target object: {obj_name}')
          
            # ========================================================================
            # STEP 2: Get pickup point coordinates
            # ========================================================================
            pickup = obj.get("pickup_point_relative_to_camera_mm")
            if pickup is None:
                self.get_logger().warn('❌ Target object has no pickup_point_relative_to_camera_mm')
                return 'Target has no pickup point'
          
            self.get_logger().info(
                f'📍 Pickup point (camera frame): X={pickup[0]:.3f} mm, Y={pickup[1]:.3f} mm'
            )
          
            # ========================================================================
            # STEP 3: Compute target position
            # ========================================================================
            try:
                target = self.compute_pick_target_rel(pickup)
            except Exception as e:
                self.get_logger().error(f'❌ Failed to compute pick target: {e}')
                return f'Failed to compute target: {e}'
          
            self.get_logger().info(
                f'📊 Target position: X={target["x"]:.3f} Y={target["y"]:.3f} Z={target["z"]:.3f}'
            )
          
            # ========================================================================
            # STEP 4: Compute object orientation
            # ========================================================================
            wrist_angle = self.compute_object_orientation(obj)
            if wrist_angle is None:
                # Default to horizontal if orientation unavailable
                wrist_angle = self.servo_wrist_horizontal
                self.get_logger().warn(f'⚠️  Using default wrist angle: {wrist_angle}°')
          
            # ========================================================================
            # STEP 5: Move to XY position (keeping Z safe)
            # ========================================================================
            self.get_logger().info('-' * 70)
            self.get_logger().info('STEP 1/6: Moving to XY position above object...')
            self.goto_position_xy_only(target["x"], target["y"])
          
            # WAIT FOR MOTION TO COMPLETE
            if not self.wait_for_motion_complete(timeout_sec=30.0):
                return "❌ Timeout waiting for XY motion to complete"
          
            self.get_logger().info('✓ XY position reached')
          
            # ========================================================================
            # STEP 6: Open gripper
            # ========================================================================
            self.get_logger().info('-' * 70)
            self.get_logger().info('STEP 2/6: Opening gripper...')
            self.open_gripper()
          
            # WAIT FOR SERVO
            self.wait_for_servo(self.motion_delay)
            self.get_logger().info('✓ Gripper opened')
          
            # ========================================================================
            # STEP 7: Rotate wrist to match object orientation
            # ========================================================================
            self.get_logger().info('-' * 70)
            self.get_logger().info(f'STEP 3/6: Rotating wrist to {wrist_angle:.1f}°...')
            self.set_wrist_angle(int(wrist_angle))
          
            # WAIT FOR SERVO
            self.wait_for_servo(self.motion_delay)
            self.get_logger().info('✓ Wrist rotated')
          
            # ========================================================================
            # STEP 8: Descend by fixed offset (slow speed)
            # ========================================================================
            self.get_logger().info('-' * 70)
            cur_z = self.get_relative_position()["z"]
            pick_z = cur_z - float(self.pick_down_mm)

            self.get_logger().info(f'STEP 4/6: Descending by {self.pick_down_mm:.1f}mm (from Z={cur_z:.3f} to Z={pick_z:.3f})...')
            self.goto_position_z_only(pick_z, feed_rate=self.pick_descent_speed)
          
            # WAIT FOR MOTION TO COMPLETE
            if not self.wait_for_motion_complete(timeout_sec=20.0):
                return "❌ Timeout waiting for Z descent to complete"
          
            self.get_logger().info('✓ Pick height reached')
          
            # ========================================================================
            # STEP 9: Close gripper to grab object
            # ========================================================================
            self.get_logger().info('-' * 70)
            self.get_logger().info('STEP 5/6: Closing gripper to grab object...')
            self.close_gripper()
          
            # WAIT FOR SERVO (extra time for secure grip)
            self.wait_for_servo(self.motion_delay * 1.5)
            self.get_logger().info('✓ Object gripped')
          
            # ========================================================================
            # STEP 10: Ascend to safe plane height
            # ========================================================================
            self.get_logger().info('-' * 70)
            self.get_logger().info(f'STEP 6/6: Ascending to safe plane (Z={self.safe_plane_z:.3f})...')
            self.goto_position_z_only(self.safe_plane_z)
          
            # WAIT FOR MOTION TO COMPLETE
            if not self.wait_for_motion_complete(timeout_sec=20.0):
                return "❌ Timeout waiting for Z ascent to complete"
          
            self.get_logger().info('✓ Safe plane reached')
          
            # ========================================================================
            # COMPLETION - Object is now held at safe height
            # ========================================================================
            self.get_logger().info('=' * 70)
            self.get_logger().info('✅ CATCH SEQUENCE COMPLETE - Object secured at safe plane')
            self.get_logger().info('=' * 70)
          
            return 'Object caught and held at safe plane'
      
        finally:
            # ========================================================================
            # ALWAYS release the lock, even if there's an error
            # ========================================================================
            with self.catch_lock:
                self.catch_in_progress = False
            self.get_logger().debug('🔓 Catch lock released')

    def decision_place(self, obj: dict) -> str:
        """
        Decide where to place the object based on its type and size.
      
        Decision logic:
        - Metal screw, body_length <= 10mm  → Point 1 (P1)
        - Metal screw, body_length > 10mm   → Point 2 (P2)
        - Wood screw, body_length <= 10mm   → Point 3 (P3)
        - Wood screw, body_length > 10mm    → Point 4 (P4)
      
        Args:
            obj: Object dictionary containing classification and body_length_mm
          
        Returns:
            Status message string
        """
        # ========================================================================
        # Extract object properties
        # ========================================================================
        classification = obj.get('classification', {})
        material_type = classification.get('type', 'unknown').lower()
        body_length = obj.get('body_length_mm', 0.0)
        obj_name = obj.get('name', 'Unknown')
      
        self.get_logger().info('=' * 70)
        self.get_logger().info('🧠 DECISION PLACE - Determining placement location...')
        self.get_logger().info('=' * 70)
        self.get_logger().info(f'📦 Object properties:')
        self.get_logger().info(f'   Name: {obj_name}')
        self.get_logger().info(f'   Material: {material_type}')
        self.get_logger().info(f'   Body Length: {body_length:.2f} mm')
      
        # ========================================================================
        # Decision logic
        # ========================================================================
        if 'metal' in material_type:
            if body_length <= 10.0:
                target_point = 1  # P1
                target_name = "P1"
                target_pos = self.presets[0]
            else:
                target_point = 2  # P2
                target_name = "P2"
                target_pos = self.presets[1]
      
        elif 'wood' in material_type:
            if body_length <= 10.0:
                target_point = 3  # P3
                target_name = "P3"
                target_pos = self.presets[2]
            else:
                target_point = 4  # P4
                target_name = "P4"
                target_pos = self.presets[3]
      
        else:
            # Unknown material type - default to P4
            self.get_logger().warn(f'⚠️  Unknown material type: {material_type}, defaulting to P4')
            target_point = 4  # P4
            target_name = "P4 (Unknown)"
            target_pos = self.presets[3]
      
        # ========================================================================
        # Log decision
        # ========================================================================
        self.get_logger().info('-' * 70)
        self.get_logger().info(f'🎯 DECISION:')
        self.get_logger().info(f'   Material: {material_type.upper()}')
        self.get_logger().info(f'   Body Length: {body_length:.2f} mm {"<= 10mm" if body_length <= 10.0 else "> 10mm"}')
        self.get_logger().info(f'   → Destination: {target_name}')
        self.get_logger().info(f'   → Position: X={target_pos["x"]:.3f} Y={target_pos["y"]:.3f} Z={target_pos["z"]:.3f}')
        self.get_logger().info('-' * 70)
      
        # ========================================================================
        # Move to placement location (XY only, keep Z safe)
        # ========================================================================
        self.get_logger().info(f'🚀 Moving to {target_name} in XY plane...')
        self.goto_position_xy_only(target_pos['x'], target_pos['y'])
      
        # WAIT FOR MOTION TO COMPLETE
        if not self.wait_for_motion_complete(timeout_sec=30.0):
            return f"❌ Timeout moving to {target_name}"
      
        self.get_logger().info(f'✓ Reached {target_name} XY position')

        '''
      
        # ========================================================================
        # Descend to placement Z height
        # ========================================================================
        self.get_logger().info('-' * 70)
        self.get_logger().info(f'⬇️ Descending to placement height (Z={target_pos["z"]:.3f})...')
        self.goto_position_z_only(target_pos['z'], feed_rate=self.pick_descent_speed)
      
        # WAIT FOR MOTION TO COMPLETE
        if not self.wait_for_motion_complete(timeout_sec=20.0):
            return f"❌ Timeout descending to {target_name}"
      
        self.get_logger().info('✓ Placement height reached')
      
        # ========================================================================
        # Wait before releasing (configurable delay for stability)
        # ========================================================================
        self.get_logger().info('-' * 70)
        self.get_logger().info(f'⏳ Waiting {self.place_delay:.1f}s for stability before release...')
        time.sleep(self.place_delay)
      
        # ========================================================================
        # Open gripper to release object
        # ======================================================================== 
        self.get_logger().info('🤏 Opening gripper to release object...')
        self.open_gripper()
      
        # WAIT FOR SERVO
        self.wait_for_servo(self.motion_delay)
        self.get_logger().info('✓ Object released')
      
        # ========================================================================
        # Ascend back to safe plane
        # ========================================================================
        self.get_logger().info('-' * 70)
        self.get_logger().info(f'⬆️ Ascending to safe plane (Z={self.safe_plane_z:.3f})...')
        self.goto_position_z_only(self.safe_plane_z)
        '''

         # ========================================================================
        # Open gripper to release object
        # ======================================================================== 
        self.get_logger().info('🤏 Opening gripper to release object...')
        self.open_gripper()
      
        # WAIT FOR SERVO
        self.wait_for_servo(self.motion_delay)
        self.get_logger().info('✓ Object released')


        # WAIT FOR MOTION TO COMPLETE
        if not self.wait_for_motion_complete(timeout_sec=20.0):
            return f"❌ Timeout ascending from {target_name}"
      
        self.get_logger().info('✓ Safe plane reached')
      
        # ========================================================================
        # Return success message
        # ========================================================================
        self.get_logger().info('=' * 70)
        self.get_logger().info(f'✅ PLACE SEQUENCE COMPLETE')
        self.get_logger().info('=' * 70)
      
        return f'Successfully placed {obj_name} ({material_type}, {body_length:.1f}mm) at {target_name}'

  
    # ========================================================================
    # Web Server (Flask)
    # ========================================================================
  
    def start_web_server(self) -> None:
        """
        Start Flask web server in a separate thread.
      
        Provides REST API and web interface for CNC control.
        """
        # Disable Flask's HTTP request logging
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
      
        app = Flask(__name__)
        node = self
        # ---- Web Routes ----
      
        @app.route('/')
        def index():
            """Serve main web interface"""
            return render_template_string(HTML_TEMPLATE)
      
        @app.route('/api/position')
        def get_position():
            """API: Get current position and presets"""
            rel_pos = node.get_relative_position()
            return jsonify({
                'x': rel_pos['x'],
                'y': rel_pos['y'],
                'z': rel_pos['z'],
                'presets': node.presets
            })
      
        @app.route('/api/camera_image')
        def get_camera_image():
            """API: Get latest camera image as base64 JPEG"""
            with node.camera_lock:
                if node.latest_camera_image is None:
                    return jsonify({'image': None})
              
                try:
                    # Convert ROS Image to OpenCV
                    cv_image = node.bridge.imgmsg_to_cv2(
                        node.latest_camera_image, 
                        desired_encoding='bgr8'
                    )
                  
                    # Encode as JPEG
                    _, buffer = cv2.imencode('.jpg', cv_image, [cv2.IMWRITE_JPEG_QUALITY, 80])
                  
                    # Convert to base64
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                  
                    return jsonify({'image': img_base64})
                except Exception as e:
                    node.get_logger().error(f'Error encoding image: {e}')
                    return jsonify({'image': None})
      
        @app.route('/api/objects')
        def get_objects():
            """API: Get detected objects"""
            with node.objects_lock:
                return jsonify({'objects': node.latest_objects})
      
        @app.route('/api/gripper_transform')
        def get_gripper_transform():
            """API: Get gripper offset parameters"""
            return jsonify({
                'offset_x': node.gripper_offset_x,
                'offset_y': node.gripper_offset_y,
                'offset_z': node.gripper_offset_z
            })
      
        @app.route('/api/set_origin', methods=['POST'])
        def set_origin():
            """API: Set current position as origin"""
            node.set_origin()
            return jsonify({'status': 'ok'})
      
        @app.route('/api/save_preset/<int:idx>', methods=['POST'])
        def save_preset(idx):
            """API: Save current position to preset"""
            if 0 <= idx < 4:
                node.save_preset(idx)
                return jsonify({'status': 'ok'})
            return jsonify({'status': 'error', 'message': 'Invalid preset index'}), 400
      
        @app.route('/api/goto', methods=['POST'])
        def goto():
            """API: Move to specified position"""
            data = request.json
            node.goto_position(
                float(data['x']),
                float(data['y']),
                float(data['z'])
            )
            return jsonify({'status': 'ok'})
      
        @app.route('/api/catch', methods=['POST'])
        def catch():
            """API: Execute catch operation"""
            message = node.catch_object()
            return jsonify({'status': 'ok', 'message': message})
      
        @app.route('/api/place', methods=['POST'])
        def place():
            """API: Execute placement operation"""
            with node.objects_lock:
                if not node.latest_objects:
                    return jsonify({'status': 'error', 'message': 'No object to place'}), 400
                obj = node.latest_objects[0]
            
            message = node.decision_place(obj)
            return jsonify({'status': 'ok', 'message': message})
      
        # Start Flask in separate thread
        web_thread = threading.Thread(
            target=lambda: app.run(
                host='0.0.0.0',
                port=node.web_port,
                debug=False,
                use_reloader=False
            ),
            daemon=True
        )
        web_thread.start()


# ============================================================================
# Main Entry Point
# ============================================================================
def main(args=None):
    """
    Main entry point for the CNC Motion Coordinator node.
  
    Initializes ROS2, creates the node, and starts the executor.
    """
    rclpy.init(args=args)
    node = CncMotionCoordinator()
  
    executor = MultiThreadedExecutor()
    executor.add_node(node)
  
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down CNC Motion Coordinator...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


