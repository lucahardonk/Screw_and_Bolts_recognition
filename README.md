# Automated Screws & Bolts Sorting System 
An automated vision-guided system that classifies screws, bolts, and nuts, estimates size/orientation, and sorts them using a 3-axis CNC machine with a custom 3D-printed gripper. Using computer vision, robotics, and modular software design, the system transforms a chaotic pile of fasteners into an organized storage layout.

## Project Architecture
The project is built mainly in Python, using OpenCV for image processing and ROS2 for modularity. ROS2 separates the system into clean components:
- Image acquisition & processing
- Camera calibration
- Object classification & measurement
- CNC kinematics & motion planning
- Pick-and-place execution

Because the workflow spans both Windows and WSL2, the system uses lightweight web-based communication (HTTP streams, debugging dashboards, etc.) to ensure reliable cross-platform data flow.

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
camera server into **rectified, undistorted, and cropped** images suitable
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

## 1.3 camera_blur.py — Gaussian Blur Filtering Node

camera_blur.py (implemented as gaussian_blur_node.py) is a ROS2
processing node that subscribes to the previous camera stream and applies a configurable Gaussian blur filter.
It then republishes the smoothed image along with its corresponding CameraInfo.

This node demonstrates how additional filtering blocks can be chained
into the perception pipeline, forming a modular image-processing
architecture where each step enhances or transforms the camera data for
downstream robotics algorithms.

------------------------------------------------------------------------

✅ Purpose of the Node

The Gaussian blur node takes the distortion-corrected frames from:

-   /camera/calibrated

and applies a Gaussian blurring kernel, reducing high-frequency noise
and smoothing textures. This is useful for:

-   Noise reduction
-   Preprocessing before edge detection
-   Stabilizing feature extraction
-   Softening segmentation inputs

It then republishes the result back into the ROS ecosystem at:

-   /camera/gaussian_blurred

------------------------------------------------------------------------

🧠 How It Works

1. Apply Gaussian Blur

The filter uses:

-   An odd-sized kernel (3, 5, 7, … 15, etc.)
-   A configurable sigma value (controls blur amount). The sigma (σ) parameter controls how strong or soft the blur is. It represents the standard deviation of the Gaussian kerne

Example:

    blurred = cv2.GaussianBlur(cv_image, (kernel_size, kernel_size), sigma)

Effects:

-   Larger kernel = wider blur
-   Larger sigma = stronger smoothing
-   Sigma 0.0 = automatically computed by OpenCV

------------------------------------------------------------------------

⚙️ Configurable Parameters

All parameters can be set at runtime using --ros-args.

  ----------------------------------------------------------------------------------------------
  Parameter                  Default                                Description
  -------------------------- -------------------------------------- ----------------------------
  input_image_topic          /camera/calibrated                     Source image topic

  input_camera_info_topic    /camera/calibrated/camera_info         Source CameraInfo topic

  output_image_topic         /camera/gaussian_blurred               Output blurred image

  gaussian_kernel_size       15                                     Must be odd, controls blur
                                                                    size

  gaussian_sigma             0.0                                    Sigma value (0 = OpenCV
                                                                    auto)
                                                                    
  ----------------------------------------------------------------------------------------------

If an even kernel is provided, the node automatically adjusts it to the
nearest odd number.

------------------------------------------------------------------------

▶️ Example Run Command

```bash

ros2 run camera_pkg gaussian_blur_node --ros-args -p input_image_topic:=/camera/calibrated -p output_image_topic:=/camera/gaussian_blurred -p gaussian_kernel_size:=15 -p gaussian_sigma:=0.0

```

------------------------------------------------------------------------


## 1.4 canny_edge_detector.py — Canny Edge Detection Node

The `canny_edge_node.py` is a ROS2 processing node that applies the Canny edge detection algorithm. It then republishes the edge-detected image along with its corresponding CameraInfo.

------------------------------------------------------------------------

✅ Purpose of the Node

The Canny edge detection node takes the distortion-corrected frames from:

-   `/camera/calibrated`

and applies the Canny edge detection algorithm, identifying sharp intensity gradients that correspond to object boundaries, lines, and contours. This is useful for:

-   Line and lane detection
-   Object boundary extraction
-   Feature-based navigation
-   Preprocessing for shape recognition
-   Vision-based control systems

It then republishes the result back into the ROS ecosystem at:

-   `/camera/canny`

------------------------------------------------------------------------

🧠 How It Works

1. **Apply Canny Edge Detection**

The filter uses a multi-stage algorithm:

-   Gradient calculation using Sobel operators
-   Non-maximum suppression to thin edges
-   Double thresholding with hysteresis to classify edges

Example:

```python
edges = cv2.Canny(cv_image, low_threshold, high_threshold, 
                  apertureSize=aperture_size, L2gradient=l2_gradient)
```

Effects:

-   **Lower `low_threshold`** = more sensitive, detects weaker edges
-   **Higher `high_threshold`** = only strong edges are retained
-   **Larger `aperture_size`** = smoother gradients, less noise sensitivity
-   **`L2gradient=True`** = more accurate but slower gradient computation

------------------------------------------------------------------------

⚙️ Configurable Parameters

All parameters can be set at runtime using `--ros-args`.

| Parameter        | Default | Description                                                                 |
|------------------|---------|-----------------------------------------------------------------------------|
| `low_threshold`  | 50      | Lower hysteresis threshold; controls edge sensitivity                       |
| `high_threshold` | 150     | Upper hysteresis threshold; defines strong edges                            |
| `aperture_size`  | 3       | Sobel kernel size (must be 3, 5, or 7); controls gradient smoothing         |
| `l2_gradient`    | false   | Use L2-norm for gradient magnitude (true = accurate, false = faster)        |

**Note:** The node automatically validates that `aperture_size` is one of `3`, `5`, or `7`. Invalid values will cause the node to terminate with an error.

------------------------------------------------------------------------

🔍 Core Functionality

The main processing happens in:

**`image_callback(self, msg: Image)`**

This callback:
1. Converts the incoming ROS `sensor_msgs/Image` to an OpenCV image using `CvBridge`
2. Applies Canny edge detection (`cv2.Canny`) with the configured thresholds and aperture size
3. Converts the resulting single-channel edge image back to BGR for visualization (edges appear white on black background)
4. Publishes the edge-detected image to `/camera/canny`
5. Republishes the latest `CameraInfo` (if available) on `/camera/canny/camera_info` with synchronized header timestamps

This is the **most important function**, as it encapsulates the entire vision-processing pipeline from subscription to publication.

------------------------------------------------------------------------

▶️ Example Run Command

**Default parameters:**

```bash
ros2 run camera_pkg canny_edge_node
```

**With all parameters explicitly set:**

```bash

ros2 run camera_pkg canny_edge_node --ros-args -p input_image_topic:=/camera/calibrated -p output_image_topic:=/camera/canny -p low_threshold:=120 -p high_threshold:=225 -p aperture_size:=3 -p l2_gradient:=true



```

**Topics:**

- **Subscribes:**
  - `/camera/calibrated` (`sensor_msgs/Image`)
  - `/camera/calibrated/camera_info` (`sensor_msgs/CameraInfo`)

- **Publishes:**
  - `/camera/canny` (`sensor_msgs/Image`)
  - `/camera/canny/camera_info` (`sensor_msgs/CameraInfo`)

------------------------------------------------------------------------


