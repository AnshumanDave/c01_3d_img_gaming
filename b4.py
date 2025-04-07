import cv2
import mediapipe as mp
import math
import threading
import time
import pyautogui

# Initialize MediaPipe Hands solution.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Fixed thresholds (adjust these values based on your testing)
TURN_THRESHOLD_DEG = 15         # Angle threshold for turning (laptop camera)
ACCEL_THRESHOLD_DEG = 15        # Angle threshold for acceleration/deceleration (side camera)
NITRO_DISTANCE_THRESHOLD = 0.1  # Normalized distance threshold for thumb to pinky detection

# Mapping of actions to keys for your game.
KEYS = {
    "turn_left": "a",
    "turn_right": "d",
    "accelerate": "w",
    "decelerate": "s",
    "nitro": "shift"
}

# Global dictionary to track key press states (to avoid repeated keyDown events).
key_states = {
    "a": False,
    "d": False,
    "w": False,
    "s": False,
    "shift": False
}

def press_key(key):
    """Press a key if it is not already pressed."""
    if not key_states[key]:
        pyautogui.keyDown(key)
        key_states[key] = True

def release_key(key):
    """Release a key if it is currently pressed."""
    if key_states[key]:
        pyautogui.keyUp(key)
        key_states[key] = False

def process_laptop_camera():
    """
    Process the laptop (front) camera to detect left/right tilt for turning.
    It calculates the angle between the wrist (landmark 0) and the middle finger tip (landmark 12)
    relative to the horizontal axis. Also, it checks for nitro boost using thumb (landmark 4) and pinky (landmark 20).
    """
    cap = cv2.VideoCapture(1)
    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            # Flip for a mirror view and convert color to RGB.
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Process hand landmarks.
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Using landmark 0 (wrist, approx. palm bottom) and landmark 12 (tip of middle finger)
                    palm_point = hand_landmarks.landmark[0]
                    middle_tip = hand_landmarks.landmark[12]

                    # Calculate the vector from the palm to the middle finger tip.
                    dx = middle_tip.x - palm_point.x
                    dy = middle_tip.y - palm_point.y

                    # Compute angle relative to the horizontal axis.
                    angle_rad = math.atan2(dy, dx)
                    angle_deg = math.degrees(angle_rad)
                    print(f"[Laptop Camera] Turning Angle: {angle_deg:.2f}")

                    # Option 1: Use the computed angle directly.
                    if angle_deg + 90 < -TURN_THRESHOLD_DEG:
                        press_key(KEYS["turn_left"])
                        release_key(KEYS["turn_right"])
                    elif 90 + angle_deg > TURN_THRESHOLD_DEG:
                        press_key(KEYS["turn_right"])
                        release_key(KEYS["turn_left"])
                    else:
                        release_key(KEYS["turn_left"])
                        release_key(KEYS["turn_right"])

                    # Nitro boost detection using thumb and pinky.
                    thumb_tip = hand_landmarks.landmark[4]
                    pinky_tip = hand_landmarks.landmark[20]
                    dist = math.hypot(thumb_tip.x - pinky_tip.x, thumb_tip.y - pinky_tip.y)
                    print(f"[Laptop Camera] Nitro Distance: {dist:.3f}")
                    if dist < NITRO_DISTANCE_THRESHOLD:
                        press_key(KEYS["nitro"])
                    else:
                        release_key(KEYS["nitro"])

                    # Draw landmarks for visual feedback.
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow("Laptop Camera - Turn & Nitro Control", image)
            if cv2.waitKey(1) & 0xFF == 27:  # Press Esc to exit.
                break

    cap.release()
    cv2.destroyWindow("Laptop Camera - Turn & Nitro Control")

def process_side_camera():
    """
    Process the side (left) camera to detect forward/backward tilt for acceleration/deceleration.
    It calculates the angle between the wrist (landmark 0) and the middle finger tip (landmark 12)
    relative to the vertical axis.
    """
    cap = cv2.VideoCapture(2)
    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            # For side camera, no flip is applied; convert color to RGB.
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Using landmark 0 (wrist, approx. palm bottom) and landmark 12 (middle finger tip)
                    palm_point = hand_landmarks.landmark[0]
                    middle_tip = hand_landmarks.landmark[12]

                    # Calculate the vector from the palm to the middle finger tip.
                    dx = middle_tip.x - palm_point.x
                    dy = middle_tip.y - palm_point.y

                    # For vertical tilt, compute the angle relative to the vertical axis.
                    # Using atan2 with dx first so that when dx=0, angle=0.
                    angle_rad = math.atan2(dx, dy)
                    angle_deg = math.degrees(angle_rad)
                    print(f"[Side Camera] Accel/Decel Angle: {angle_deg:.2f}")

                    # Determine forward (accelerate) or backward (decelerate) movement.
                    if (angle_deg < 0) & (angle_deg + 180 > ACCEL_THRESHOLD_DEG):
                        press_key(KEYS["accelerate"])
                        release_key(KEYS["decelerate"])
                    elif 180 - angle_deg > ACCEL_THRESHOLD_DEG:
                        press_key(KEYS["decelerate"])
                        release_key(KEYS["accelerate"])
                    else:
                        release_key(KEYS["accelerate"])
                        release_key(KEYS["decelerate"])

                    # Draw landmarks for visual feedback.
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow("Side Camera - Acceleration/Deceleration Control", image)
            if cv2.waitKey(1) & 0xFF == 27:  # Press Esc to exit.
                break

    cap.release()
    cv2.destroyWindow("Side Camera - Acceleration/Deceleration Control")

def main():
    # Start threads for processing each camera feed concurrently.
    t1 = threading.Thread(target=process_laptop_camera)
    t2 = threading.Thread(target=process_side_camera)

    t1.start()
    t2.start()

    try:
        while t1.is_alive() and t2.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        # Ensure all keys are released when the program exits.
        for key in key_states:
            release_key(key)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
