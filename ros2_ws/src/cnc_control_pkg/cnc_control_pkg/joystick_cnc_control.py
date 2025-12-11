import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Joy
import time

class JoystickCncControl(Node):
    def __init__(self):
        super().__init__('joystick_cnc_control')

        # Parameters
        self.declare_parameter('feed_rate', 500)        # mm/min
        self.declare_parameter('max_step', 0.5)         # mm per update
        self.declare_parameter('deadzone', 0.1)         # joystick deadzone
        self.declare_parameter('update_rate', 20.0)     # Hz

        self.feed_rate = self.get_parameter('feed_rate').value
        self.max_step = self.get_parameter('max_step').value
        self.deadzone = self.get_parameter('deadzone').value
        self.update_period = 1.0 / self.get_parameter('update_rate').value

        # Publisher → send GRBL jog commands
        self.pub_cmd = self.create_publisher(String, '/serial_cnc_in', 10)

        # Subscriber → joystick input
        self.subscription = self.create_subscription(
            Joy,
            '/joy',
            self.joy_callback,
            10
        )

        self.last_cmd_time = 0
        self.joy_x = 0.0
        self.joy_y = 0.0

        # Timer → send commands at fixed update rate
        self.timer = self.create_timer(self.update_period, self.send_jog_command)

        self.get_logger().info("Joystick CNC Control Node Started.")

    def joy_callback(self, msg: Joy):
        # axes: msg.axes[0] = X, msg.axes[1] = Y
        if len(msg.axes) >= 2:
            self.joy_x = msg.axes[0]
            self.joy_y = msg.axes[1]

    def send_jog_command(self):
        # Apply deadzone
        x = 0 if abs(self.joy_x) < self.deadzone else self.joy_x
        y = 0 if abs(self.joy_y) < self.deadzone else self.joy_y

        # If joystick is idle → stop sending
        if x == 0 and y == 0:
            return

        # Map joystick (-1..1) to step size
        step_x = x * self.max_step
        step_y = y * self.max_step

        # Build GRBL incremental jog command ($J=G91 ...)
        # Negative Y because joystick usually has up = -1
        cmd = f"$J=G91 X{step_x:.4f} Y{-step_y:.4f} F{self.feed_rate}"

        msg = String()
        msg.data = cmd
        self.pub_cmd.publish(msg)

        self.get_logger().info(f"Sent jog: {cmd}")


def main(args=None):
    rclpy.init(args=args)
    node = JoystickCncControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
