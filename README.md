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

* Object detection algorithms

* Position estimation

* Additional image processing features

#### 📦 **Services & Bringup Packages**

Utility packages providing:

* Launch files for multi-node startup

* Custom services for light control

* Gripper servo control services

---

## 🔧 ROS2 Communication

For those unfamiliar with ROS2:

* **Packages**: Macro-level folders containing related functionality

* **Nodes**: Individual programs performing specific tasks

* **Topics**: Asynchronous message passing between nodes (publish/subscribe)

* **Services**: Synchronous client-server communication

