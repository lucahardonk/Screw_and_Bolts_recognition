#!/usr/bin/env python3
"""
ROS2 Node for CNC control via joystick with web interface.
Provides jogging, preset positions, servo control, and lighting.
"""

import re
import threading
import time
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String, Int32
from sensor_msgs.msg import Joy
from flask import Flask, render_template_string, request, jsonify

from services_pkg.srv import SetLightColor


# ============================================================================
# HTML Template
# ============================================================================
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>CNC Controller</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 20px auto; padding: 0 20px; }
        .section { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 8px; }
        .position { font-size: 24px; font-weight: bold; margin: 10px 0; }
        button { padding: 10px 20px; margin: 5px; cursor: pointer; }
        .preset { display: flex; align-items: center; gap: 10px; margin: 10px 0; }
        .preset input { width: 70px; padding: 5px; }
        .btn-primary { background: #007bff; color: white; border: none; }
        .btn-success { background: #28a745; color: white; border: none; }
        .btn-warning { background: #ffc107; border: none; }
    </style>
</head>
<body>
    <h1>CNC Joystick Controller</h1>
    
    <div class="section">
        <h2>Current Position (Relative to Origin)</h2>
        <div class="position">
            X: <span id="pos-x">0.000</span> | 
            Y: <span id="pos-y">0.000</span> | 
            Z: <span id="pos-z">0.000</span>
        </div>
        <button class="btn-warning" onclick="setOrigin()">Set Origin (Zero All)</button>
    </div>
    
    <div class="section">
        <h2>Preset Points</h2>
        {% for i in range(4) %}
        <div class="preset">
            <strong>P{{i+1}}:</strong>
            <input type="number" step="0.1" id="preset{{i}}-x" placeholder="X" value="0">
            <input type="number" step="0.1" id="preset{{i}}-y" placeholder="Y" value="0">
            <input type="number" step="0.1" id="preset{{i}}-z" placeholder="Z" value="0">
            <button class="btn-primary" onclick="saveCurrent({{i}})">Save Current</button>
            <button class="btn-success" onclick="goTo({{i}})">Go To</button>
        </div>
        {% endfor %}
    </div>

    <script>
        function updatePosition() {
            fetch('/api/position').then(r => r.json()).then(data => {
                document.getElementById('pos-x').textContent = data.x.toFixed(3);
                document.getElementById('pos-y').textContent = data.y.toFixed(3);
                document.getElementById('pos-z').textContent = data.z.toFixed(3);
                for (let i = 0; i < 4; i++) {
                    document.getElementById('preset'+i+'-x').value = data.presets[i].x.toFixed(3);
                    document.getElementById('preset'+i+'-y').value = data.presets[i].y.toFixed(3);
                    document.getElementById('preset'+i+'-z').value = data.presets[i].z.toFixed(3);
                }
            });
        }
        
        function setOrigin() {
            fetch('/api/set_origin', {method: 'POST'}).then(() => updatePosition());
        }
        
        function saveCurrent(idx) {
            fetch('/api/save_preset/' + idx, {method: 'POST'}).then(() => updatePosition());
        }
        
        function goTo(idx) {
            const x = document.getElementById('preset'+idx+'-x').value;
            const y = document.getElementById('preset'+idx+'-y').value;
            const z = document.getElementById('preset'+idx+'-z').value;
            fetch('/api/goto', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({x: parseFloat(x), y: parseFloat(y), z: parseFloat(z)})
            });
        }
        
        setInterval(updatePosition, 500);
        updatePosition();
    </script>
</body>
</html>
'''

class JoystickCncControl(Node):
    def __init__(self):
        super().__init__('joystick_cnc_control')

        # --- Parameters ---
        self.declare_parameter('feed_rate', 2500)
        self.declare_parameter('max_step', 0.6)
        self.declare_parameter('deadzone', 0.1)
        self.declare_parameter('jog_update_rate', 50.0)  # CNC jog rate
        self.declare_parameter('servo_update_rate', 20.0)  # Servo update rate
        self.declare_parameter('web_port', 5000)
        self.declare_parameter('position_poll_rate', 2.0)

        self.feed_rate = self.get_parameter('feed_rate').value
        self.max_step = self.get_parameter('max_step').value
        self.deadzone = self.get_parameter('deadzone').value
        self.jog_update_period = 1.0 / self.get_parameter('jog_update_rate').value
        self.servo_update_period = 1.0 / self.get_parameter('servo_update_rate').value
        self.web_port = self.get_parameter('web_port').value
        self.position_poll_period = 1.0 / self.get_parameter('position_poll_rate').value

        # --- ROS Publishers & Subscribers ---
        self.pub_cmd = self.create_publisher(String, '/serial_cnc_in', 10)
        self.sub_status = self.create_subscription(String, '/serial_cnc_out', self.serial_callback, 10)
        self.subscription = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        
        # --- Servo Publishers (Topic-based) ---
        self.pub_servo1 = self.create_publisher(Int32, '/servo1', 10)
        self.pub_servo2 = self.create_publisher(Int32, '/servo2', 10)

        # --- State ---
        self.joy_axes = [0.0] * 8
        self.joy_buttons = [0] * 12
        self.machine_x = 0.0
        self.machine_y = 0.0
        self.machine_z = 0.0
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.origin_z = 0.0
        self.presets = [{'x':0.0,'y':0.0,'z':0.0} for _ in range(4)]
        
        # Servo/light state with thread safety
        self.servo1_target = 90
        self.servo2_target = 90
        self.servo1_current = 90
        self.servo2_current = 90
        self.lights_on = False
        self.prev_buttons = [0] * 12
        
        # Thread locks for safety
        self.servo_lock = threading.Lock()
        self.jog_lock = threading.Lock()
        
        # Control flags for threads
        self.running = True

        # --- Services (Lights only) ---
        self.light_client = self.create_client(SetLightColor, '/set_light_color')

        # --- Thread 1: CNC Control (dedicated) ---
        self.cnc_thread = threading.Thread(target=self._cnc_control_loop, daemon=True)
        self.cnc_thread.start()

        # --- Thread 2: Servo & Lights Control (dedicated) ---
        self.servo_light_thread = threading.Thread(target=self._servo_light_loop, daemon=True)
        self.servo_light_thread.start()

        # --- Thread 3: Flask Web Server (dedicated) ---
        self.start_web_server()

        self.get_logger().info(f"Joystick CNC Control Node started with 3 independent threads")
        self.get_logger().info(f"  - Thread 1: CNC Control (50Hz)")
        self.get_logger().info(f"  - Thread 2: Servo & Lights (20Hz)")
        self.get_logger().info(f"  - Thread 3: Web GUI at http://localhost:{self.web_port}")
        self.get_logger().info(f"  - Servo control via topics: /servo1, /servo2")

        # --- GRBL Unlock at startup (delayed 5 seconds) ---
        threading.Timer(5.0, self.send_grbl_unlock_delayed).start()

    # ========================================================================
    # THREAD 1: CNC Control Loop (50Hz)
    # ========================================================================
    def _cnc_control_loop(self):
        """Dedicated thread for CNC jogging and position polling"""
        poll_counter = 0
        poll_interval = int(self.position_poll_period / self.jog_update_period)
        
        while self.running:
            # Send jog commands at 50Hz
            self.send_jog_command()
            
            # Poll position at 2Hz
            poll_counter += 1
            if poll_counter >= poll_interval:
                self.poll_position()
                poll_counter = 0
            
            time.sleep(self.jog_update_period)

    def send_jog_command(self):
        """Send jog command based on joystick input"""
        with self.jog_lock:
            if len(self.joy_axes) < 5:
                return
            x = 0 if abs(self.joy_axes[0]) < self.deadzone else self.joy_axes[0]
            y = 0 if abs(self.joy_axes[1]) < self.deadzone else self.joy_axes[1]
            z = 0 if abs(self.joy_axes[4]) < self.deadzone else self.joy_axes[4]

            if x == 0 and y == 0 and z == 0:
                return

            step_x = x * self.max_step
            step_y = -y * self.max_step
            step_z = z * self.max_step

            cmd = f"$J=G91 G21 X{step_x:.4f} Y{step_y:.4f} Z{step_z:.4f} F{self.feed_rate}\n"
            msg = String()
            msg.data = cmd
            self.pub_cmd.publish(msg)

    def poll_position(self):
        """Poll CNC position"""
        msg = String()
        msg.data = "?"
        self.pub_cmd.publish(msg)

    # ========================================================================
    # THREAD 2: Servo & Lights Control Loop (20Hz)
    # ========================================================================
    def _servo_light_loop(self):
        """Dedicated thread for servo and light updates"""
        while self.running:
            self.update_servos()
            time.sleep(self.servo_update_period)

    def update_servos(self):
        """Update servos based on targets - now using topics"""
        with self.servo_lock:
            # Only send if position changed significantly
            if abs(self.servo1_target - self.servo1_current) > 2:
                self.publish_servo(1, int(self.servo1_target))
                self.servo1_current = self.servo1_target
            
            if abs(self.servo2_target - self.servo2_current) > 2:
                self.publish_servo(2, int(self.servo2_target))
                self.servo2_current = self.servo2_target

    def publish_servo(self, servo_id, position):
        """Publish servo position to topic (0-180)"""
        msg = Int32()
        msg.data = int(position)
        
        if servo_id == 1:
            self.pub_servo1.publish(msg)
        elif servo_id == 2:
            self.pub_servo2.publish(msg)

    def toggle_all_lights(self, on):
        """Turn all 8 lights on (white) or off"""
        color = (255, 255, 255) if on else (0, 0, 0)
        for light_id in range(8):
            self.set_light(light_id, *color)
            time.sleep(0.5)

    def set_light(self, light_id, r, g, b):
        """Set individual light color"""
        if not self.light_client.service_is_ready():
            return
        req = SetLightColor.Request()
        req.light_id = int(light_id)
        req.r = int(r)
        req.g = int(g)
        req.b = int(b)
        self.light_client.call_async(req)

    # ========================================================================
    # THREAD 3: Flask Web Server & ROS Callbacks (Main Thread)
    # ========================================================================
    def start_web_server(self):
        """Start Flask web server in dedicated thread"""
        app = Flask(__name__)
        node = self

        @app.route('/')
        def index():
            return render_template_string(HTML_TEMPLATE)

        @app.route('/api/position')
        def get_position():
            rel = node.get_relative_position()
            return jsonify({'x': rel['x'], 'y': rel['y'], 'z': rel['z'], 'presets': node.presets})

        @app.route('/api/set_origin', methods=['POST'])
        def set_origin():
            node.set_origin()
            return jsonify({'status': 'ok'})

        @app.route('/api/save_preset/<int:idx>', methods=['POST'])
        def save_preset(idx):
            if 0 <= idx < 4:
                node.save_preset(idx)
            return jsonify({'status': 'ok'})

        @app.route('/api/goto', methods=['POST'])
        def goto():
            data = request.json
            node.goto_position(data['x'], data['y'], data['z'])
            return jsonify({'status': 'ok'})

        web_thread = threading.Thread(
            target=lambda: app.run(host='0.0.0.0', port=node.web_port, debug=False, use_reloader=False),
            daemon=True
        )
        web_thread.start()

    # ----------------- Joystick Callback (Main ROS Thread) -----------------
    def joy_callback(self, msg: Joy):
        """Process joystick input - runs in main ROS thread"""
        self.joy_axes = msg.axes
        self.joy_buttons = msg.buttons

        # Update servo targets (thread-safe)
        with self.servo_lock:
            # Servo 1: Trigger axis (axis 2)
            if len(msg.axes) >= 3:
                trigger = msg.axes[2]
                self.servo1_target = int((1.0 - trigger) / 2.0 * 180)  # Map -1..1 to 180..0

            # Servo 2: Right stick horizontal (axis 3)
            if len(msg.axes) >= 4:
                stick = msg.axes[3]
                if abs(stick) > self.deadzone:
                    self.servo2_target += stick * 3.0
                    self.servo2_target = max(0, min(180, self.servo2_target))

        # Lights: Toggle with button 3 (index 3)
        if len(msg.buttons) >= 4:
            if msg.buttons[3] == 1 and self.prev_buttons[3] == 0:
                self.lights_on = not self.lights_on
                self.toggle_all_lights(self.lights_on)

        self.prev_buttons = list(msg.buttons)

    def serial_callback(self, msg: String):
        """Process serial feedback - runs in main ROS thread"""
        match = re.search(r'MPos:([-\d\.]+),([-\d\.]+),([-\d\.]+)', msg.data)
        if match:
            self.machine_x = float(match.group(1))
            self.machine_y = float(match.group(2))
            self.machine_z = float(match.group(3))

    # ----------------- GRBL Unlock -----------------
    def send_grbl_unlock_delayed(self):
        """Send $X to unlock GRBL after 5 second delay"""
        msg = String()
        msg.data = "$X\n"
        self.pub_cmd.publish(msg)
        self.get_logger().info(f"Sent GRBL unlock: '{msg.data.strip()}'")

    # ----------------- Position / Presets -----------------
    def get_relative_position(self):
        return {
            'x': self.machine_x - self.origin_x,
            'y': self.machine_y - self.origin_y,
            'z': self.machine_z - self.origin_z
        }

    def set_origin(self):
        self.origin_x = self.machine_x
        self.origin_y = self.machine_y
        self.origin_z = self.machine_z

    def save_preset(self, idx):
        rel = self.get_relative_position()
        self.presets[idx] = {'x': rel['x'], 'y': rel['y'], 'z': rel['z']}

    def goto_position(self, rel_x, rel_y, rel_z):
        abs_x = self.origin_x + rel_x
        abs_y = self.origin_y + rel_y
        abs_z = self.origin_z + rel_z
        cmd = f"G90 G1 X{abs_x:.4f} Y{abs_y:.4f} Z{abs_z:.4f} F{self.feed_rate}"
        msg = String()
        msg.data = cmd
        self.pub_cmd.publish(msg)

    # ----------------- Cleanup -----------------
    def destroy_node(self):
        """Clean shutdown of threads"""
        self.running = False
        if hasattr(self, 'cnc_thread'):
            self.cnc_thread.join(timeout=1.0)
        if hasattr(self, 'servo_light_thread'):
            self.servo_light_thread.join(timeout=1.0)
        super().destroy_node()


# ============================================================================
# Main Entry Point
# ============================================================================
def main(args=None):
    """Main entry point for the node."""
    rclpy.init(args=args)
    node = JoystickCncControl()
    
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()