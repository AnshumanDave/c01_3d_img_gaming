import cv2
import mediapipe as mp
import time
import threading
from collections import deque
import numpy as np
from pynput.keyboard import Controller  # Replaces 'keyboard' module

# Initialize keyboard control
keyboard = Controller()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, 
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open video capture (single camera)
cap = cv2.VideoCapture(0)  

# Variables to track hand movement
previous_x, previous_y = None, None
threshold = 0.05  # Movement threshold

def detect_hand_movement():
    global previous_x, previous_y

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Extract hand landmark data
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the palm center (wrist point)
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                hand_x, hand_y = wrist.x, wrist.y  # Normalized (0 to 1)

                if previous_x is not None and previous_y is not None:
                    dx = hand_x - previous_x
                    dy = hand_y - previous_y

                    # Movement detection
                    if dy < -threshold:  # Move up → Press "W"
                        keyboard.press('w')
                    else:
                        keyboard.release('w')

                    if dy > threshold:  # Move down → Press "S"
                        keyboard.press('s')
                    else:
                        keyboard.release('s')

                    if dx < -threshold:  # Move left → Press "A"
                        keyboard.press('a')
                    else:
                        keyboard.release('a')

                    if dx > threshold:  # Move right → Press "D"
                        keyboard.press('d')
                    else:
                        keyboard.release('d')

                # Update previous position
                previous_x, previous_y = hand_x, hand_y

        # Show camera feed
        cv2.imshow("Hand Control for Krunker.io", frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the function
detect_hand_movement()
