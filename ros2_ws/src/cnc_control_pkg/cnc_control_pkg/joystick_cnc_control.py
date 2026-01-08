#!/usr/bin/env python3
"""
ROS2 Node: Joystick CNC Control (smoothed jog)
- Single-threaded
- Low-pass filtering (EMA) of joystick axes
- Fixed-rate jog command sender (timer) decoupled from joystick callbacks
- Servos and lights handled in joystick callback (immediate)
- GRBL unlock ($X) sent once after 5s
- Detailed debug logging for all actions
"""

import re
from typing import Dict

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32
from sensor_msgs.msg import Joy

from services_pkg.srv import SetLightColor


class JoystickCncControl(Node):
    def __init__(self):
        super().__init__('joystick_cnc_control')

        # --- Parameters ---
        self.declare_parameter('feed_rate', 2500)
        self.declare_parameter('max_step', 0.5)
        self.declare_parameter('deadzone', 0.15)
        self.declare_parameter('jog_rate', 20.0)       # Hz at which jog commands are sent
        self.declare_parameter('smoothing', 0.2)       # EMA alpha (0..1)

        self.feed_rate = float(self.get_parameter('feed_rate').value)
        self.max_step = float(self.get_parameter('max_step').value)
        self.deadzone = float(self.get_parameter('deadzone').value)
        self.jog_rate = float(self.get_parameter('jog_rate').value)
        self.smoothing = float(self.get_parameter('smoothing').value)

        # --- ROS Publishers & Subscribers ---
        self.pub_cmd = self.create_publisher(String, '/serial_cnc_in', 10)
        self.sub_status = self.create_subscription(String, '/serial_cnc_out', self.serial_callback, 10)
        self.subscription = self.create_subscription(Joy, '/joy', self.joy_callback, 10)

        # --- Servo Publishers ---
        self.pub_servo1 = self.create_publisher(Int32, '/servo1', 10)
        self.pub_servo2 = self.create_publisher(Int32, '/servo2', 10)

        # --- Light service client ---
        self.light_client = self.create_client(SetLightColor, '/set_light_color')

        # --- State ---
        self.machine_x = 0.0
        self.machine_y = 0.0
        self.machine_z = 0.0

        # Filtered joystick values (EMA)
        self.filtered_x = 0.0
        self.filtered_y = 0.0
        self.filtered_z = 0.0

        # Raw last seen joystick (for debugging)
        self.last_raw_x = 0.0
        self.last_raw_y = 0.0
        self.last_raw_z = 0.0

        self.servo2_position = 90
        self.lights_on = False
        self.prev_buttons = [0] * 12

        # Jogging state
        self.is_jogging = False

        # Unlock once flag
        self.unlock_sent = False

        # --- Timers ---
        # One-shot unlock after 5 seconds
        self.unlock_timer = self.create_timer(5.0, self.send_grbl_unlock)
        # Periodic jog sender (decoupled from /joy frequency)
        self.jog_timer = self.create_timer(1.0 / self.jog_rate, self.jog_timer_cb)

        # Startup logs
        self.get_logger().info("=" * 60)
        self.get_logger().info("Joystick CNC Control (smoothed) node started")
        self.get_logger().info(f"feed_rate={self.feed_rate}  max_step={self.max_step}  deadzone={self.deadzone}")
        self.get_logger().info(f"jog_rate={self.jog_rate} Hz  smoothing(alpha)={self.smoothing}")
        self.get_logger().info("=" * 60)

    # ------------------------------------------------------------------
    # GRBL unlock (one-shot)
    # ------------------------------------------------------------------
    def send_grbl_unlock(self):
        """Send $X once after startup to unlock GRBL"""
        if not self.unlock_sent:
            msg = String()
            msg.data = "$X\n"
            self.pub_cmd.publish(msg)
            self.get_logger().info("🔓 Sent GRBL unlock: $X")
            self.unlock_sent = True
            # cancel the unlock timer (one-shot)
            self.unlock_timer.cancel()

    # ------------------------------------------------------------------
    # JOY callback: update filtered values, handle servos & lights
    # ------------------------------------------------------------------
    def joy_callback(self, msg: Joy):
        """Update filters and immediate controls (servos, lights)."""
        # Read raw axes with deadzone
        raw_x = msg.axes[0] if len(msg.axes) > 0 else 0.0
        raw_y = msg.axes[1] if len(msg.axes) > 1 else 0.0
        raw_z = msg.axes[4] if len(msg.axes) > 4 else 0.0

        raw_x = raw_x if abs(raw_x) > self.deadzone else 0.0
        raw_y = raw_y if abs(raw_y) > self.deadzone else 0.0
        raw_z = raw_z if abs(raw_z) > self.deadzone else 0.0

        # Save raw for debug
        self.last_raw_x, self.last_raw_y, self.last_raw_z = raw_x, raw_y, raw_z

        # Exponential moving average (EMA) filtering
        a = self.smoothing
        self.filtered_x = a * raw_x + (1.0 - a) * self.filtered_x
        self.filtered_y = a * raw_y + (1.0 - a) * self.filtered_y
        self.filtered_z = a * raw_z + (1.0 - a) * self.filtered_z

        self.get_logger().debug(
            f"🎮 Raw axes: x={raw_x:.3f} y={raw_y:.3f} z={raw_z:.3f} | "
            f"Filtered: x={self.filtered_x:.3f} y={self.filtered_y:.3f} z={self.filtered_z:.3f}"
        )

        # --- Servo 1: Left trigger (axis 2) ---
        if len(msg.axes) >= 3:
            trigger = msg.axes[2]
            servo1_pos = int((1.0 - trigger) / 2.0 * 180)  # Map -1..1 -> 180..0
            servo1_msg = Int32()
            servo1_msg.data = servo1_pos
            self.pub_servo1.publish(servo1_msg)
            self.get_logger().debug(f"🔧 Servo1 -> {servo1_pos}° (trigger={trigger:.3f})")

        # --- Servo 2: Right stick horizontal (axis 3) ---
        if len(msg.axes) >= 4:
            stick = msg.axes[3]
            if abs(stick) > self.deadzone:
                old_pos = self.servo2_position
                self.servo2_position = max(0, min(180, self.servo2_position + stick * 3.0))
                servo2_msg = Int32()
                servo2_msg.data = int(self.servo2_position)
                self.pub_servo2.publish(servo2_msg)
                self.get_logger().debug(f"🔧 Servo2 -> {int(self.servo2_position)}° (was {int(old_pos)}°, stick={stick:.3f})")

        # --- Lights: toggle (button index 4) ---
        if len(msg.buttons) > 4:
            if msg.buttons[4] == 1 and self.prev_buttons[4] == 0:
                self.lights_on = not self.lights_on
                self.get_logger().info(f"💡 Button 4 pressed - Lights {'ON' if self.lights_on else 'OFF'}")
                self.toggle_all_lights(self.lights_on)

        # update prev buttons
        self.prev_buttons = list(msg.buttons)

    # ------------------------------------------------------------------
    # Jog timer: sends jog commands at fixed rate using filtered values
    # ------------------------------------------------------------------
    def jog_timer_cb(self):
        """Send jog commands according to filtered joystick values at a fixed rate."""
        x = self.filtered_x
        y = self.filtered_y
        z = self.filtered_z

        # consider very small values as zero (avoid tiny jitter)
        small_thresh = 0.001
        x = 0.0 if abs(x) < small_thresh else x
        y = 0.0 if abs(y) < small_thresh else y
        z = 0.0 if abs(z) < small_thresh else z

        if x != 0.0 or y != 0.0 or z != 0.0:
            step_x = x * self.max_step * -1.0
            step_y = -y * self.max_step * -1.0
            step_z = z * self.max_step

            # Build a clean, single-space command string
            cmd = f"$J=G91 G21 X{step_x:.4f} Y{step_y:.4f} Z{step_z:.4f} F{int(self.feed_rate)}\n"
            msg = String()
            msg.data = cmd
            self.pub_cmd.publish(msg)

            if not self.is_jogging:
                self.is_jogging = True
                self.get_logger().info("🎮 Jogging started (timer)")

            self.get_logger().debug(f"📨 Sending jog: {cmd.strip()}")
        else:
            # If we were jogging and axes now zero, send the GRBL jog-cancel (0x85)
            if self.is_jogging:
                cancel_msg = String()
                cancel_msg.data = "\x85"
                self.pub_cmd.publish(cancel_msg)
                self.is_jogging = False
                self.get_logger().info("⏹️  Jog cancelled (axes ~ 0)")

    # ------------------------------------------------------------------
    # Lights: call service to set each light (non-blocking, no sleep)
    # ------------------------------------------------------------------
    def toggle_all_lights(self, on: bool):
        """Turn all 8 lights on (white) or off (fast, no sleeps)."""
        if not self.light_client.service_is_ready():
            self.get_logger().warn("⚠️ Light service not ready")
            return

        color = (255, 255, 255) if on else (0, 0, 0)
        self.get_logger().info(f"💡 Setting all lights to {'WHITE' if on else 'OFF'} (RGB={color})")

        for light_id in range(8):
            req = SetLightColor.Request()
            req.light_id = int(light_id)
            req.r = int(color[0])
            req.g = int(color[1])
            req.b = int(color[2])
            # call async, do not block the node
            self.light_client.call_async(req)
            self.get_logger().debug(f"   → Light {light_id} set request sent: RGB{color}")

        self.get_logger().info(f"✅ Light set requests dispatched for all 8 lights")

    # ------------------------------------------------------------------
    # Serial callback: parse MPos and debug other messages
    # ------------------------------------------------------------------
    def serial_callback(self, msg: String):
        """Parse serial output (MPos) and debug-report other messages."""
        data = msg.data.strip()
        match = re.search(r'MPos:([-\d\.]+),([-\d\.]+),([-\d\.]+)', data)
        if match:
            self.machine_x = float(match.group(1))
            self.machine_y = float(match.group(2))
            self.machine_z = float(match.group(3))
            self.get_logger().debug(
                f"📊 Machine position parsed: X={self.machine_x:.3f} Y={self.machine_y:.3f} Z={self.machine_z:.3f}"
            )
        else:
            # Log other serial replies (ok/error/msgs)
            self.get_logger().debug(f"📡 Serial: {data}")

def main(args=None):
    rclpy.init(args=args)
    node = JoystickCncControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Shutdown requested")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()