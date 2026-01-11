from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the camera launch file path
    camera_launch_file = os.path.join(
        get_package_share_directory('camera_pkg'),
        'launch',
        'camera_launch.py'
    )

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

        # CNC motion coordinator
        Node(
            package='cnc_control_pkg',
            executable='cnc_motion_coordinator',
            name='cnc_motion_coordinator',
            output='screen'
        ),

        # Camera launch file
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(camera_launch_file)
        ),
    ])