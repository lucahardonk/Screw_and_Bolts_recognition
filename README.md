# 🛠️ CNC Pick & Place Demonstration

A computer vision-based robotic pick-and-place system combining ROS2 and OpenCV for automated object detection and manipulation.

---

## 📹 Demo Video

[![Watch the video](https://img.youtube.com/vi/r9lemT6S0Cc/0.jpg)](https://youtu.be/EGnl93jBOss)

---

## 📖 Project Overview

This project was developed as the final assignment for the Signal, Image and Video Processing course at the University of Trento. It combines computer vision techniques with robotic control to create an automated pick-and-place system using a CNC machine.

---

## 🏗️ Architecture

The system leverages two primary frameworks:

* **ROS2**: Provides modular architecture for separating vision and robotic control, improving code quality and development workflow

* **Python + OpenCV**: Handles all image processing and computer vision tasks

### ROS2 Package Structure

The project is organized into the following ROS2 packages (source code located in the `src` folder):

For those unfamiliar with ROS2:

* **Packages**: Macro-level folders containing related functionality

* **Nodes**: Individual programs performing specific tasks

* **Topics**: Asynchronous message passing between nodes (publish/subscribe)

* **Services**: Synchronous client-server communication


#### 📦 **Arduino Control Package**

Interfaces with Arduino hardware to control:

* Servo motors

* Camera lighting system

#### 📦 **Camera Calibration Package**

Performs camera calibration using:

* Chessboard pattern detection

* OpenCV calibration algorithms

#### 📦 **CNC Control Package**

Manages CNC machine communication:

* G-code command interface

* ROS2 publishers/subscribers for movement control

* Exposes CNC functionality to the ROS2 ecosystem

#### 📦 **Camera Package**

Comprehensive vision processing module containing multiple nodes for:

* Camera interfacing via OpenCV

* Step by step image processing filter application

* Advanced image processing features we will discuss in detail below

#### 📦 **Services & Bringup Packages**

Utility packages providing:

* Bringup: Launch files for multi-node startup

* Custom services for light control

* Gripper servo control services

---

## Hardware components
The hardware setup is straightforward yet effective. At its core, we have a 3-axis CNC machine that serves as the movement platform. Mounted on the Z-axis is a camera that captures the workspace from above, providing the visual input for our computer vision system. An Arduino board acts as the interface between the high-level control and the physical actuators, managing both the servo motors for the gripper and the lighting system that ensures consistent image quality. All of this is orchestrated by a computer running the ROS2 nodes and executing the computer vision algorithms in real-time. 

![Project Hardware Components](documentation_media/project_hardware_components.png)

---

## Software Pipeline
In this section, we'll dive into the vision processing pipeline that powers the system. The Camera Package contains the core nodes responsible for image acquisition, processing, and object detection. These nodes communicate seamlessly through ROS2 topics, creating a distributed yet cohesive system. We'll also touch on the CNC control mechanisms for those interested in the mechanical side of the operation.

Below is the rqt_graph visualization showing how all the nodes interact within the system. I know the image is clear enough, I will describe everything in detail anyway:
![rqt_graph visualization](documentation_media/rqt_graph.png)


