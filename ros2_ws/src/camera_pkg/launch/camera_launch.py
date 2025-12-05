#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node

def generate_launch_description():

    # --- Run Windows camera script directly (not in wsl) ---
    windows_path = r"%USERPROFILE%\Documents\Projects\Screw_and_Bolts_recognition\ros2_ws\src\camera_pkg\camera_pkg\first_camera_node_windows_native.py"

    windows_camera = ExecuteProcess(
        cmd=[
            "cmd.exe", "/C",
            "python",
            windows_path
        ],
        shell=False
    )

    # --- camera_calibrated_node ---
    # this node performs camera calibration using the provided parameters YAML file
    camera_calibrated_node = Node(
        package='camera_pkg',
        executable='camera_calibrated_node',
        name='camera_calibrated_node',
        output='screen',
        parameters=[]
    )

    # Add 10 second delay after launch start before camera_calibrated_node
    delayed_camera_calibrated = TimerAction(
        period=10.0,
        actions=[camera_calibrated_node]
    )

    # --- camera_visualizer_node ---
    # localhost web gui to visualize each camera topic
    camera_visualizer_node = Node(
        package='camera_pkg',
        executable='camera_visualizer_node',
        name='camera_visualizer_node',
        output='screen',
        parameters=[{"topics": ["/camera/calibrated", "/camera/gaussian_blurred", "/camera/canny", "/camera/background_removed"]}]
    )

    return LaunchDescription([
        windows_camera,            # starts immediately
        delayed_camera_calibrated, # starts 10s after launch
        camera_visualizer_node,    # starts immediately at launch
    ])