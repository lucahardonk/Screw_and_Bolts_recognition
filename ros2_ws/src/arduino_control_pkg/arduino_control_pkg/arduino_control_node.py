#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import serial
import serial.tools.list_ports

from services_pkg.srv import SetServo, SetLightColor

class ArduinoControlNode(Node):
    def __init__(self):
        super().__init__('arduino_control_node')

        # Parameters (you can expose them via ROS2 params)
        port = self.declare_parameter('port', '/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0').get_parameter_value().string_value
        baud = self.declare_parameter('baud', 115200).get_parameter_value().integer_value

        try:
            self.serial = serial.Serial(port, baudrate=baud, timeout=1.0)
            self.get_logger().info(f'Opened serial port {port} at {baud} baud')
        except Exception as e:
            self.get_logger().error(f'Failed to open serial port {port}: {e}')
            self.serial = None

        # Create services
        self.servo_srv = self.create_service(SetServo, 'set_servo', self.handle_set_servo)
        self.light_srv = self.create_service(SetLightColor, 'set_light_color', self.handle_set_light)

    def send_serial_command(self, cmd: str) -> bool:
        """Send a command string to Arduino, return True if sent."""
        if self.serial is None or not self.serial.is_open:
            self.get_logger().error('Serial port is not open')
            return False

        try:
            line = (cmd.strip() + '\n').encode('utf-8')
            self.serial.write(line)
            self.get_logger().info(f'Sent command: {cmd}')
            return True
        except Exception as e:
            self.get_logger().error(f'Error writing to serial: {e}')
            return False

    def handle_set_servo(self, request, response):
        servo_id = request.servo_id
        position = request.position

        # Basic validation
        if servo_id not in (1, 2):
            response.success = False
            response.message = f'Invalid servo_id {servo_id}, must be 1 or 2'
            return response

        if not (0 <= position <= 180):
            response.success = False
            response.message = f'Invalid position {position}, must be 0..180'
            return response

        # Example protocol: "SERVO,<id>,<pos>"
        cmd = f'SERVO,{servo_id},{position}'
        ok = self.send_serial_command(cmd)

        response.success = ok
        response.message = 'OK' if ok else 'Failed to send command'
        return response

    def handle_set_light(self, request, response):
        light_id = request.light_id
        r = request.r
        g = request.g
        b = request.b

        if light_id != 1:
            response.success = False
            response.message = f'Invalid light_id {light_id}, only 1 supported for now'
            return response

        if not all(0 <= x <= 255 for x in (r, g, b)):
            response.success = False
            response.message = 'RGB values must be in 0..255'
            return response

        # Example protocol: "LED,<id>,<r>,<g>,<b>"
        cmd = f'LED,{light_id},{r},{g},{b}'
        ok = self.send_serial_command(cmd)

        response.success = ok
        response.message = 'OK' if ok else 'Failed to send command'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = ArduinoControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.serial is not None and node.serial.is_open:
            node.serial.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
