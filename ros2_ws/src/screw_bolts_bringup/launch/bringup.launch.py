from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

        # Joystick driver
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            output='screen'
        ),

        # CNC joystick control
        Node(
            package='cnc_control_pkg',
            executable='joystick_cnc_control',
            name='joystick_cnc_control',
            output='screen'
        ),

        # CNC serial controller
        Node(
            package='cnc_control_pkg',
            executable='cnc_serial_controller',
            name='cnc_serial_controller',
            output='screen'
        ),

        # Arduino control server
        Node(
            package='arduino_control_pkg',
            executable='arduino_control_server',
            name='arduino_control_server',
            output='screen'
        ),
    ])
