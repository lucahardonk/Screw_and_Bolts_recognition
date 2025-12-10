from setuptools import setup

package_name = 'arduino_control_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='YOUR_NAME',
    maintainer_email='your@email',
    description='Arduino servo+LED control via ROS2 services',
    license='TODO',
    entry_points={
        'console_scripts': [
            'arduino_control_server = arduino_control_pkg.arduino_control_node:main',
        ],
    },
)