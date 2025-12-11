#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

import serial
import serial.tools.list_ports
import time
from typing import Optional

from services_pkg.srv import SetServo, SetLightColor

'''
full parameter command:
ros2 run arduino_control_pkg arduino_control_server --ros-args -p port:=/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0 -p baud:=115200 -p timeout:=1.0 -p auto_reconnect:=true

Example service calls:
ros2 service call /set_servo services_pkg/srv/SetServo "{servo_id: 1, position: 90}"
ros2 service call /set_servo services_pkg/srv/SetServo "{servo_id: 2, position: 180}"

ros2 service call /set_light_color services_pkg/srv/SetLightColor "{light_id: 0, r: 255, g: 255, b: 255}"
ros2 service call /set_light_color services_pkg/srv/SetLightColor "{light_id: 1, r: 0, g: 255, b: 0}"
ros2 service call /set_light_color services_pkg/srv/SetLightColor "{light_id: 7, r: 0, g: 0, b: 255}"
'''


class ArduinoControlNode(Node):
    """
    ROS2 node for controlling Arduino via serial communication.
    Provides services for servo control and LED color setting.
    All validation is done on the microcontroller side.
    """
    
    # Constants
    SERIAL_TIMEOUT = 1.0
    RECONNECT_INTERVAL = 5.0  # seconds
    
    def __init__(self):
        super().__init__('arduino_control_node')
        
        # Declare parameters
        self.declare_parameter('port', '/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0')
        self.declare_parameter('baud', 115200)
        self.declare_parameter('timeout', self.SERIAL_TIMEOUT)
        self.declare_parameter('auto_reconnect', True)
        
        # Get parameters
        self.port = self.get_parameter('port').get_parameter_value().string_value
        self.baud = self.get_parameter('baud').get_parameter_value().integer_value
        self.timeout = self.get_parameter('timeout').get_parameter_value().double_value
        self.auto_reconnect = self.get_parameter('auto_reconnect').get_parameter_value().bool_value
        
        # Serial connection
        self.serial: Optional[serial.Serial] = None
        
        # Initialize serial connection
        self._connect_serial()
        
        # Create callback group for concurrent service calls
        self.callback_group = ReentrantCallbackGroup()
        
        # Create services
        self.servo_srv = self.create_service(
            SetServo, 
            'set_servo', 
            self.handle_set_servo,
            callback_group=self.callback_group
        )
        
        self.light_srv = self.create_service(
            SetLightColor, 
            'set_light_color', 
            self.handle_set_light,
            callback_group=self.callback_group
        )
        
        # Create reconnection timer if auto-reconnect is enabled
        if self.auto_reconnect:
            self.reconnect_timer = self.create_timer(
                self.RECONNECT_INTERVAL,
                self._check_and_reconnect
            )
        
        self.get_logger().info('Arduino Control Node initialized')
        self.get_logger().info(f'Services available: /set_servo, /set_light_color')
    
    def _connect_serial(self) -> bool:
        """
        Attempt to connect to the serial port.
        Returns True if successful, False otherwise.
        """
        try:
            if self.serial is not None and self.serial.is_open:
                self.serial.close()
            
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                timeout=self.timeout,
                write_timeout=self.timeout
            )
            
            # Wait for Arduino to reset (common after serial connection)
            time.sleep(2.0)
            
            # Flush any startup messages
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
            
            self.get_logger().info(
                f'Successfully opened serial port {self.port} at {self.baud} baud'
            )
            return True
            
        except serial.SerialException as e:
            self.get_logger().error(f'Failed to open serial port {self.port}: {e}')
            self.serial = None
            return False
        except Exception as e:
            self.get_logger().error(f'Unexpected error opening serial port: {e}')
            self.serial = None
            return False
    
    def _check_and_reconnect(self):
        """
        Periodic callback to check serial connection and reconnect if needed.
        """
        if self.serial is None or not self.serial.is_open:
            self.get_logger().warn('Serial connection lost. Attempting to reconnect...')
            self._connect_serial()
    
    def _is_serial_ready(self) -> bool:
        """Check if serial connection is ready."""
        return self.serial is not None and self.serial.is_open
    
    def send_serial_command(self, cmd: str) -> tuple[bool, str]:
        """
        Send a command string to Arduino.
        
        Args:
            cmd: Command string to send
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        if not self._is_serial_ready():
            msg = 'Serial port is not open'
            self.get_logger().error(msg)
            return False, msg
        
        try:
            # Prepare command
            line = (cmd.strip() + '\n').encode('utf-8')
            
            # Send command
            bytes_written = self.serial.write(line)
            self.serial.flush()  # Ensure data is sent
            
            if bytes_written != len(line):
                msg = f'Incomplete write: {bytes_written}/{len(line)} bytes'
                self.get_logger().warn(msg)
                return False, msg
            
            self.get_logger().info(f'Sent command: {cmd}')
            
            # Optional: Read response from Arduino (if your protocol supports it)
            # response = self.serial.readline().decode('utf-8').strip()
            # self.get_logger().debug(f'Arduino response: {response}')
            
            return True, 'Command sent successfully'
            
        except serial.SerialTimeoutException:
            msg = 'Serial write timeout'
            self.get_logger().error(msg)
            return False, msg
            
        except serial.SerialException as e:
            msg = f'Serial error: {e}'
            self.get_logger().error(msg)
            # Mark connection as broken
            if self.serial is not None:
                try:
                    self.serial.close()
                except:
                    pass
                self.serial = None
            return False, msg
            
        except Exception as e:
            msg = f'Unexpected error writing to serial: {e}'
            self.get_logger().error(msg)
            return False, msg
    
    def handle_set_servo(self, request, response):
        """
        Service callback for setting servo position.
        No validation - all checks done on microcontroller.
        
        Args:
            request: SetServo request with servo_id and position
            response: SetServo response with success and message
        """
        servo_id = request.servo_id
        position = request.position
        
        self.get_logger().info(
            f'Received set_servo request: servo_id={servo_id}, position={position}'
        )
        
        # Send command to Arduino (no validation)
        cmd = f'SERVO,{servo_id},{position}'
        success, message = self.send_serial_command(cmd)
        
        response.success = success
        response.message = message
        
        if success:
            self.get_logger().info(f'Servo {servo_id} command sent: position={position}')
        else:
            self.get_logger().error(f'Failed to send servo {servo_id} command: {message}')
        
        return response
    
    def handle_set_light(self, request, response):
        """
        Service callback for setting LED color.
        No validation - all checks done on microcontroller.
        
        Args:
            request: SetLightColor request with light_id, r, g, b
            response: SetLightColor response with success and message
        """
        light_id = request.light_id
        r = request.r
        g = request.g
        b = request.b
        
        self.get_logger().info(
            f'Received set_light_color request: '
            f'light_id={light_id}, RGB=({r},{g},{b})'
        )
        
        # Send command to Arduino (no validation)
        cmd = f'LED,{light_id},{r},{g},{b}'
        success, message = self.send_serial_command(cmd)
        
        response.success = success
        response.message = message
        
        if success:
            self.get_logger().info(f'Light {light_id} command sent: RGB({r},{g},{b})')
        else:
            self.get_logger().error(f'Failed to send light {light_id} command: {message}')
        
        return response
    
    def destroy_node(self):
        """Clean up resources before node shutdown."""
        self.get_logger().info('Shutting down Arduino Control Node...')
        
        # Close serial connection
        if self.serial is not None and self.serial.is_open:
            try:
                self.serial.close()
                self.get_logger().info('Serial port closed')
            except Exception as e:
                self.get_logger().error(f'Error closing serial port: {e}')
        
        super().destroy_node()


def main(args=None):
    """Main entry point for the node."""
    rclpy.init(args=args)
    
    node = None
    try:
        node = ArduinoControlNode()
        
        # Use MultiThreadedExecutor for concurrent service calls
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        
        try:
            executor.spin()
        except KeyboardInterrupt:
            node.get_logger().info('Keyboard interrupt received')
        finally:
            executor.shutdown()
            
    except Exception as e:
        print(f'Fatal error: {e}')
        
    finally:
        if node is not None:
            node.destroy_node()
        
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()