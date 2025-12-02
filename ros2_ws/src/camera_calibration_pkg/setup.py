from setuptools import setup

package_name = 'camera_calibration_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'opencv-python', 'numpy', 'pyyaml'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Fisheye camera calibration package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fisheye_calibration_service = camera_calibration_pkg.fisheye_calibration_service:main',
        ],
    },
)