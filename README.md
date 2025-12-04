# Screw_and_Bolts_recognition
image prossecing project to categorize screws from bolts from nuts and order them by size + cnc pick and place application to fisically reorganize the mess

## 0.3 Why This Architecture

Running the camera directly inside WSL2 causes limitations:

- WSL cannot reliably handle high-bandwidth USB video devices.
- High-resolution modes often fail to negotiate.
- Frame drops, unstable FPS, or OpenCV driver issues are common.

By running the capture on Windows and providing HTTP endpoints instead:

- USB stays native → **full resolution works reliably**
- No special drivers needed in WSL
- Any language / framework can consume the frames
- ROS2 nodes can work in WSL2 without touching Windows camera APIs
- The camera pipeline becomes modular, isolated, and easy to debug

This node acts as a **bridge**:
- Windows handles the hardware  
- WSL/ROS handles the processing  
---

## 1.1 first_camera_node_windows_native

`first_camera_node_windows_native` is the Windows-based camera acquisition node of the project.  
Instead of forwarding the USB camera into WSL2 (which is cumberson every time you need to connect the camera),  
this node captures frames **directly on Windows (don't launch in wsl)**  and exposes them through two lightweight HTTP endpoints.

This design eliminates all USB passthrough issues and provides a stable, high-performance interface for any
WSL2 or ROS2 node that needs access to camera images.

---

The node creates *two independent camera endpoints*, each with a different purpose:
http://localhost:8000

### **1. `/video` — Live Downscaled MJPEG Stream**
- Resolution: **1280×720**
- JPEG compressed at **quality 60**
- Designed for:
  - Real-time visualization in a browser
  - Monitoring the camera feed
  - Any low-latency UI or dashboard component

This endpoint is optimized for speed and responsiveness for operator feedback only, not for iamge processing.

### **2. `/frame` — Full-Resolution Snapshot (On Demand)**
- Resolution: **3264×2448** (the highest supported by the camera)
- JPEG encoded at **quality 100**
- Always returns the latest available frame from the capture thread
- Designed for:
  - Image processing pipelines
  - Calibration procedures
  - inference nodes
  - Building datasets and AI

This endpoint is intentionally **pull-based**: a processing node can request frames exactly when it needs them.


## 1.5 Running the Node

Start the node on Windows:

```bash
python.exe first_camera_node_windows_native.py

```