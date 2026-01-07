import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial
import threading
import time

class SerialCncNode(Node):
    def __init__(self):
        super().__init__('serial_cnc_node')

        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('baud_rate', 115200)

        self.port = self.get_parameter('port').get_parameter_value().string_value
        self.baud_rate = self.get_parameter('baud_rate').get_parameter_value().integer_value

        self.get_logger().info(f'Connecting to GRBL on port {self.port} at {self.baud_rate} baud...')
        try:
            self.ser = serial.Serial(self.port, self.baud_rate, timeout=0.1)
            self.get_logger().info('Successfully connected to GRBL.')
            # Wake up GRBL
            self.ser.write(b"\r\n\r\n")
            time.sleep(2)  # Wait for GRBL to initialize
            self.ser.reset_input_buffer()
            self.get_logger().info('GRBL initialized and ready.')
        except serial.SerialException as e:
            self.get_logger().error(f'Failed to connect to GRBL: {e}')
            self.ser = None

        self.publisher_ = self.create_publisher(String, 'serial_cnc_out', 10)
        self.subscription = self.create_subscription(String, 'serial_cnc_in', self.command_callback, 10)

        self._stop_event = threading.Event()
        self._reader_thread = threading.Thread(target=self.read_serial)
        self._reader_thread.daemon = True
        self._reader_thread.start()

    def command_callback(self, msg):
        if self.ser and self.ser.is_open:
            command = msg.data.strip()
            self.get_logger().info(f'Sending command: {command}')
            self.ser.write((command + '\n').encode())
        else:
            self.get_logger().warn('Serial port not connected. Cannot send command.')

    def read_serial(self):
        while not self._stop_event.is_set():
            if self.ser and self.ser.in_waiting > 0:
                try:
                    line = self.ser.readline().decode(errors='ignore').strip()
                    if line:
                        msg = String()
                        msg.data = line
                        self.publisher_.publish(msg)
                        self.get_logger().info(f'Received: {line}')
                except Exception as e:
                    self.get_logger().error(f'Error reading from serial: {e}')
            else:
                time.sleep(0.05)

    def destroy_node(self):
        self._stop_event.set()
        self._reader_thread.join()
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.get_logger().info('Serial connection closed.')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SerialCncNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()