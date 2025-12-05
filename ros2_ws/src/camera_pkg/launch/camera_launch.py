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

    #delay 5 second with TimerAction delayed_camera_calibrated

    # --- camera_calibrated_node ---
    # this node performs camera calibration using the provided parameters YAML file
    camera_calibrated_node = Node(
        package='camera_pkg',
        executable='camera_calibrated_node',
        name='camera_calibrated_node',
        output='screen',
        parameters=[]
    )

    # --- background_removal_node ---
    background_removal_node = Node(
        package='camera_pkg',
        executable='background_removal_node',
        name='background_removal_node',
        output='screen',
        parameters=[
            {"input_image_topic": "/camera/calibrated"},
            {"output_image_topic": "/camera/background_removed"}
        ]
    )

    # --- grey_scaled_node ---
    grey_scaled_node = Node(
        package='camera_pkg',
        executable='grey_scaled_node',
        name='grey_scaled_node',
        output='screen',
        parameters=[
            {"input_image_topic": "/camera/background_removed"},
            {"output_image_topic": "/camera/background_grayed"}
        ]
    )


    # --- camera_visualizer_node ---
    # localhost web gui to visualize each camera topic
    camera_visualizer_node = Node(
        package='camera_pkg',
        executable='camera_visualizer_node',
        name='camera_visualizer_node',
        output='screen',
        parameters=[{"topics": ["/camera/calibrated", "/camera/gaussian_blurred", "/camera/canny", "/camera/background_removed", "/camera/background_grayed", "/camera/otsu", "/camera/closure"]}]
    )

    




    # Add 5 second delay after launch start before camera_calibrated_node
    delayed_camera_calibrated = TimerAction(
        period=5.0,
        actions=[camera_calibrated_node]
    )
    

    return LaunchDescription([
        windows_camera,           # starts immediately
        delayed_camera_calibrated, # starts 5s after launch
        background_removal_node,
        grey_scaled_node,


        camera_visualizer_node,    
    ])