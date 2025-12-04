# Automated Screws & Bolts Sorting System
image prossecing project to categorize screws from bolts from nuts and order them by size
also a 3 axis cnc machine with a custom 3D printed gripper has to pick and place to organize the mess of screws

## Project Architecture
the project will be aminly in python. as opencv offers a great library to work with images. 
also to maintain modularity i would like to use ros2, also very useful to saparate image processing, calibation and cnc kinematics in different packages. 
also some triks to make comunication efficnt betwheen different os like windos and wsl some web based comunication and debugging will be implemented.

---
# 1 Camera Package
## 1.1 first_camera_node_windows_native.py

`first_camera_node_windows_native` is the Windows-based camera acquisition node of the project.  
Instead of forwarding the USB camera into WSL2 (which is cumberson every time you need to connect the camera),  
this node captures frames **directly on Windows (don't launch in wsl)**  and exposes them through two lightweight HTTP endpoints.

This design eliminates all USB passthrough issues and provides a stable, high-performance interface for any
WSL2 or ROS2 node that needs access to camera images.

---

The node creates *two independent camera endpoints*, each with a different purpose:
http://localhost:8000

### ** `/video` — Live Downscaled MJPEG Stream**
- Resolution: **1280×720**
- JPEG compressed at **quality 60**
- Designed for:
  - Real-time visualization in a browser
  - Monitoring the camera feed
  - Any low-latency UI or dashboard component

This endpoint is optimized for speed and responsiveness for operator feedback only, not for iamge processing.

### ** `/frame` — Full-Resolution Snapshot (On Demand)**
- Resolution: **3264×2448** (the highest supported by the camera)
- JPEG encoded at **quality 100**
- Always returns the latest available frame from the capture thread
- Designed for:
  - Image processing pipelines
  - Calibration procedures
  - inference nodes
  - Building datasets and AI

This endpoint is intentionally **pull-based**: a processing node can request frames exactly when it needs them.

### Running the Node
Start the node on Windows:

```bash
python.exe first_camera_node_windows_native.py
```

---

## 1.2 camera_calibrated_node.py — Fisheye Undistortion & ROS2 Publishing

camera_calibrated_node.py is the ROS2 component responsible for
converting the high-resolution JPEG frames exposed by the Windows-side
camera server into rectified, undistorted, and cropped images suitable
for robotics processing inside the ROS ecosystem.

It reads the raw images from:

    http://localhost:8000/frame

(or whatever IP you configure), then applies the calibration parameters
previously generated using the calibration package described in the next
section.

------------------------------------------------------------------------

✅ What This Node Does

1. Pulls Full-Resolution Frames From /frame

The Windows camera node exposes the latest available high-quality JPEG
snapshot at:

-   http://<camera_ip>:8000/frame

This node fetches those frames at a configurable rate (default: 5 Hz).

------------------------------------------------------------------------

2. Loads Calibration From fisheye_camera.yaml

The node automatically loads:

-   Intrinsic matrix K
-   Distortion coefficients D
-   Original resolution
-   Any relevant metadata

from the calibration file:

    fisheye_camera.yaml

This file contains both the intrinsic parameters and distortion model
needed to correct the fisheye distortion.

------------------------------------------------------------------------

3. Applies Fisheye Undistortion

Using OpenCV’s cv2.fisheye model, the node:

-   Computes a rectification matrix
-   Generates undistort + rectify maps
-   Warps the full image into a distortion-free view

This transforms the raw fisheye frame into a geometrically correct
perspective.

------------------------------------------------------------------------

4. Performs a Center Crop

After undistortion, the image is:

-   Cropped to a fixed 1500×1500 region
-   Centered based on the corrected optical center
-   Guaranteed to contain no warped or black regions
-   Suitable for all downstream geometry-based processes

This ensures the final image has clean, usable content only.

------------------------------------------------------------------------

5. Publishes to ROS2 Topics

After undistorting and cropping, the node publishes:

📸 /camera/calibrated

The processed image as a ROS2 sensor_msgs/Image message (BGR8).

📄 /camera/calibrated/camera_info

A matching CameraInfo message containing:

-   Updated and cropped intrinsic matrix
-   Zero distortion (since the image is already rectified)
-   Projection matrix of the corrected camera
-   Exact width/height of the cropped image

This guarantees correct integration with:

-   Computer vision algorithms
-   SLAM
-   Object detection
-   Pose estimation
-   Any ROS2 node requiring accurate camera geometry

------------------------------------------------------------------------

## 1.3 camera_calibrated_node.py — Fisheye Undistortion & ROS2 Publishing



