import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Joy
from flask import Flask, render_template_string, request, jsonify
import threading
import time

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

        # Parameters
        self.declare_parameter('feed_rate', 500)
        self.declare_parameter('max_step', 0.5)
        self.declare_parameter('deadzone', 0.1)
        self.declare_parameter('update_rate', 20.0)
        self.declare_parameter('web_port', 5000)

        self.feed_rate = self.get_parameter('feed_rate').value
        self.max_step = self.get_parameter('max_step').value
        self.deadzone = self.get_parameter('deadzone').value
        self.update_period = 1.0 / self.get_parameter('update_rate').value
        self.web_port = self.get_parameter('web_port').value

        # Publisher
        self.pub_cmd = self.create_publisher(String, '/serial_cnc_in', 10)

        # Subscriber
        self.subscription = self.create_subscription(Joy, '/joy', self.joy_callback, 10)

        # Joystick state
        self.joy_x = 0.0
        self.joy_y = 0.0
        self.joy_z = 0.0

        # Position tracking
        self.machine_x = 0.0
        self.machine_y = 0.0
        self.machine_z = 0.0

        # Origin offset
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.origin_z = 0.0

        # 4 preset points (relative to origin)
        self.presets = [{'x': 0.0, 'y': 0.0, 'z': 0.0} for _ in range(4)]

        # Timer
        self.timer = self.create_timer(self.update_period, self.send_jog_command)

        # Start Flask in separate thread
        self.start_web_server()

        self.get_logger().info(f"Joystick CNC Control Node Started. Web GUI at http://localhost:{self.web_port}")

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
        # Calculate absolute machine position
        abs_x = self.origin_x + rel_x
        abs_y = self.origin_y + rel_y
        abs_z = self.origin_z + rel_z
        
        # Send GRBL absolute move command
        cmd = f"$J=G90 X{abs_x:.4f} Y{abs_y:.4f} Z{abs_z:.4f} F{self.feed_rate}"
        msg = String()
        msg.data = cmd
        self.pub_cmd.publish(msg)
        self.get_logger().info(f"Go to: {cmd}")

    def joy_callback(self, msg: Joy):
        if len(msg.axes) >= 2:
            self.joy_x = msg.axes[0]
            self.joy_y = msg.axes[1]
        if len(msg.axes) >= 5:
            self.joy_z = msg.axes[4]

    def send_jog_command(self):
        x = 0 if abs(self.joy_x) < self.deadzone else self.joy_x
        y = 0 if abs(self.joy_y) < self.deadzone else self.joy_y
        z = 0 if abs(self.joy_z) < self.deadzone else self.joy_z

        if x == 0 and y == 0 and z == 0:
            return

        step_x = x * self.max_step
        step_y = -y * self.max_step  # Invert Y
        step_z = z * self.max_step

        # Update tracked machine position
        self.machine_x += step_x
        self.machine_y += step_y
        self.machine_z += step_z

        cmd = f"$J=G91 X{step_x:.4f} Y{step_y:.4f} Z{step_z:.4f} F{self.feed_rate}"
        msg = String()
        msg.data = cmd
        self.pub_cmd.publish(msg)
        self.get_logger().info(f"Sent jog: {cmd}")

    def start_web_server(self):
        app = Flask(__name__)
        node = self

        @app.route('/')
        def index():
            return render_template_string(HTML_TEMPLATE)

        @app.route('/api/position')
        def get_position():
            rel = node.get_relative_position()
            return jsonify({
                'x': rel['x'], 'y': rel['y'], 'z': rel['z'],
                'presets': node.presets
            })

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

        thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=node.web_port, debug=False, use_reloader=False), daemon=True)
        thread.start()


def main(args=None):
    rclpy.init(args=args)
    node = JoystickCncControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
