#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node

def generate_launch_description():

    # --- Run Linux camera node by absolute path (same as your working command) ---
    linux_camera = Node(
        package='camera_pkg',
        executable='first_camera_node_linux_native',
        name='first_camera_node_linux_native',
        output='screen',
        parameters=[]
    )

    camera_calibrated_node = Node(
        package='camera_pkg',
        executable='camera_calibrated_node',
        name='camera_calibrated_node',
        output='screen',
        parameters=[]
    )

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

    
    otsu_node = Node(
        package='camera_pkg',
        executable='otsu_node',
        name='otsu_node',
        output='screen',
        parameters=[
            {"input_image_topic": "/camera/background_grayed"},
            {"output_image_topic": "/camera/otsu"}
        ]
    )


    morphological_closure_node = Node(
        package='camera_pkg',
        executable='morphological_closure_node',
        name='morphological_closure_node',
        output='screen',
        parameters=[
            {"input_image_topic": "/camera/otsu"},
            {"output_image_topic": "/camera/closure"},
            {"kernel_size": 5},
            {"iterations": 2},
        ]
    )


    contour_detection_node = Node(
        package='camera_pkg',
        executable='contour_detection_node',
        name='contour_detection_node',
        output='screen',
        parameters=[
            {"input_image_topic": "/camera/closure"},
            {"output_image_topic": "/camera/contour"},
            {"output_binary_topic": "/camera/contour/binary_mask"},
            {"contour_color_g": 255},
            {"contour_thickness": 2},
            {"min_contour_area": 100.0},
        ]
    )

    min_rect_area_node = Node(
        package='camera_pkg',
        executable='min_rect_area_node',
        name='min_rect_area_node',
        output='screen',
        parameters=[
            {"input_image_topic": "/camera/closure"},
            {"output_image_topic": "/camera/objects"},
            {"output_info_topic": "/camera/object_information"},
            {"min_contour_area": 500.0},
        ]
    )

    physical_features_node = Node(
        package='camera_pkg',
        executable='physical_features_node',
        name='physical_features_node',
        output='screen',
        parameters=[
            {"input_image_topic": "/camera/closure"},
            {"input_info_topic": "/camera/object_information"},
            {"output_image_topic": "/camera/physical_features"},
        ]
    )

    camera_visualizer_node = Node(
        package='camera_pkg',
        executable='camera_visualizer_node',
        name='camera_visualizer_node',
        output='screen',
        parameters=[{
            "topics": [
                "/camera/calibrated",
                "/camera/gaussian_blurred",
                "/camera/canny",
                "/camera/background_removed",
                "/camera/background_grayed",
                "/camera/otsu",
                "/camera/closure",
                "/camera/contour",
                "/camera/objects",
                "/camera/physical_features"
                
            ]
        }]
    )

    delayed_camera_calibrated = TimerAction(
        period=5.0,
        actions=[camera_calibrated_node]
    )

    return LaunchDescription([
        linux_camera,
        delayed_camera_calibrated,
        background_removal_node,
        grey_scaled_node,
        otsu_node,
        morphological_closure_node,
        contour_detection_node,
        min_rect_area_node,
        physical_features_node,

        camera_visualizer_node,
    ])