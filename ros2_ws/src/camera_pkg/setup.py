from setuptools import setup
import os
from glob import glob

package_name = 'camera_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        # Register package
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),

        # package.xml
        ('share/' + package_name, ['package.xml']),

        # Calibration YAML file
        ('share/' + package_name, ['fisheye_camera.yaml']),

        # Install launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Camera package with calibration, visualization, and filtering nodes.',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'first_camera_node_linux_native = camera_pkg.first_camera_node_linux_native:main',
            'camera_calibrated_node = camera_pkg.camera_calibrated_node:main',
            'gaussian_blur_node = camera_pkg.gaussian_blur:main',
            'canny_edge_node = camera_pkg.canny_node:main',
            'background_removal_node = camera_pkg.background_removal_node:main',
            'grey_scaled_node = camera_pkg.grey_scaled_node:main',
            'otsu_node = camera_pkg.otsu_thresholding:main',
            'morphological_closure_node = camera_pkg.morphological_closure:main',
            'contour_detection_node = camera_pkg.contour_detection_node:main',
            'min_rect_area_node = camera_pkg.min_rect_area:main',
            'physical_features_node = camera_pkg.physical_features_extraction:main',
            
            'camera_visualizer_node = camera_pkg.camera_visualizer_node:main',
            
            

        ],
    },
)
