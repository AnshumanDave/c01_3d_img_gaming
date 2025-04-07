import cv2
import mediapipe as mp
import time
from pynput.keyboard import Controller

# Initialize MediaPipe and keyboard
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

keyboard = Controller()
cap = cv2.VideoCapture(0)

previous_x, previous_y = None, None
threshold = 0.002  # INSANELY LOW threshold for UNREALISTIC sensitivity
last_key_pressed = None

def press_key(key):
    global last_key_pressed
    if key != last_key_pressed:
        if last_key_pressed:
            keyboard.release(last_key_pressed)
        keyboard.press(key)
        last_key_pressed = key

def release_all_keys():
    global last_key_pressed
    if last_key_pressed:
        keyboard.release(last_key_pressed)
        last_key_pressed = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            point = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = point.x, point.y

            if previous_x is not None and previous_y is not None:
                dx = x - previous_x
                dy = y - previous_y

                if abs(dx) < threshold and abs(dy) < threshold:
                    # Still press the last key to keep movement going
                    if last_key_pressed:
                        keyboard.press(last_key_pressed)
                else:
                    # Hyper-sensitive direction detection
                    if abs(dx) > abs(dy):
                        if dx > 0:
                            press_key('d')
                        else:
                            press_key('a')
                    else:
                        if dy > 0:
                            press_key('w')
                        else:
                            press_key('w')

            previous_x, previous_y = x, y
    else:
        release_all_keys()
        previous_x, previous_y = None, None

    cv2.imshow("Hand Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

release_all_keys()
cap.release()
cv2.destroyAllWindows()
