# 👓 Spectacle AR Try-On Demo

## 🚀 Overview

This project is a real-time Augmented Reality (AR) application that allows a user to **"try on" various spectacle styles** using a standard webcam feed. Built as a proof-of-concept (PoC) for computer vision expertise, the application accurately detects and tracks facial features to achieve a highly convincing AR effect.

This project demonstrates strong capability in handling complex dependencies, image processing, and real-time geometric transformations.

****

---

## ✨ Key Features

* **Real-Time Tracking:** Utilizes **dlib's 68-point shape predictor** for highly accurate facial landmark detection, ensuring smooth tracking of the user's head movements.
* **Perspective AR Overlay:** Implements **perspective transformation** (`cv2.findHomography`) to warp the 2D spectacle image. This makes the glasses appear as if they are a 3D object tracking the user's head rotation and viewing angle.
* **Robust Image Handling:** Custom logic ensures reliable loading and channel conversion of transparent PNG images, resolving common issues encountered with OpenCV's alpha channel handling.
* **Interactive Selection:** Allows the user to cycle through different spectacle models using keyboard input (currently implemented via 'A' and 'D' keys).

---

## 🛠️ Tech Stack

| Category | Tool / Library | Purpose |
| :--- | :--- | :--- |
| **Language** | Python (3.8+) | Core scripting language. |
| **Computer Vision** | OpenCV (`cv2`) | Video stream handling, image manipulation, and blending. |
| **Detection** | `dlib` | Face detection and 68-point facial landmark prediction. |
| **Numerical Ops** | NumPy | High-performance array and matrix operations for transformations. |
| **Environment** | `venv` / `pip` | Isolated dependency management. |

---

## 💻 Installation & Setup

These instructions will guide you through setting up the project on your local machine.

### Prerequisites

* Python 3.8+
* Ac