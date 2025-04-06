# 3D Image Driven Gaming

A real-time, dual-camera system that maps 3D hand gestures to game controls using computer vision. Built using Python, OpenCV, and MediaPipe, this project enables immersive gesture-driven control for PC games by interpreting hand orientation and movement into keyboard inputs.

## 🔧 Features

- Dual-camera setup for front and side gesture tracking  
- Real-time gesture-to-key mapping (steering, acceleration, braking, nitro)  
- MediaPipe Hands for landmark detection  
- Angle-based gesture recognition  
- pyautogui for keyboard control  
- Multithreaded processing for parallel camera input

## 🎮 Demo Use Case

Configured for a car racing game with the following gesture mappings:

- **Turn Left / Right** → Tilt hand left/right (Laptop Camera)  
- **Accelerate / Brake** → Move hand forward/backward (Webcam)  
- **Nitro Boost** → Bring thumb and pinky close together (Laptop Camera)

## 🛠️ Tech Stack

**Backend:**
- Python  
- OpenCV (Video processing)  
- MediaPipe (Hand landmark detection)  
- pyautogui (Keyboard simulation)  
- threading, math, time (Control logic and performance)

## 🧠 How It Works

1. **Camera Input**: Laptop camera (front view) and external webcam (side view) capture hand gestures.
2. **Landmark Detection**: MediaPipe extracts 21 2D hand landmarks from each camera feed.
3. **Gesture Logic**: Angles between specific hand points (e.g., wrist to middle fingertip) are used to determine tilt and trigger specific game actions.
4. **Key Control**: Based on the gesture state, corresponding keyboard keys (W, A, S, D, Shift) are pressed or released using pyautogui.
5. **Multithreading**: Separate threads handle both camera inputs for smooth and synchronized performance.

## 📷 Setup

Ensure the following before running:
- Webcam connected (for side view)
- Game window in focus to receive keyboard input
- Python 3.7+ installed

Install dependencies:
```bash
pip install opencv-python mediapipe pyautogui

▶️ Run:
To start the gesture controller, run this file:

    python b4.py
