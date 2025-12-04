from setuptools import setup
import os
from glob import glob

package_name = 'camera_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include the calibration YAML file
        ('share/' + package_name, ['fisheye_camera.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Camera calibration and undistortion package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_calibrated_node = camera_pkg.camera_calibrated_node:main',
            'gaussian_blur_node = camera_pkg.gaussian_blur:main',
            'camera_visualizer_node = camera_pkg.camera_visualizer_node:main',
            'canny_edge_node = camera_pkg.canny_node:main',
        ],
    },
)