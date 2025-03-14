import cv2
import mediapipe as mp
import time
import threading
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, 
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS  # Get hand skeletal connections

# Open video streams
cap1 = cv2.VideoCapture(0)  # Laptop Camera
cap2 = cv2.VideoCapture(1)  # External Camera (e.g., Phone via Iriun)

# Buffers to store frames and timestamps
buffer1 = deque()
buffer2 = deque()
threshold = 30  # Maximum timestamp difference in milliseconds
lock = threading.Lock()

# Frame storage for animation
hand_frames = deque(maxlen=50)  # Store up to 50 frames for smooth animation

# Exit flag to stop threads and animation
exit_flag = threading.Event()

def capture_frames(cap, buffer):
    """Continuously captures frames with timestamps and stores them in a buffer."""
    while cap.isOpened() and not exit_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        if timestamp == 0:  # If timestamp isn't provided, use system time
            timestamp = time.time() * 1000  # Convert seconds to milliseconds
        with lock:
            buffer.append((timestamp, frame))

def rotate_coordinates(hand_landmarks, angle=20):
    """Applies a rotation transformation to normalize the 'up' direction of the hand."""
    angle = np.radians(angle)
    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle), np.cos(angle), 0],
                  [0, 0, 1]])  # Rotation around Z-axis
    return [tuple(np.dot(R, np.array(pt))) for pt in hand_landmarks]

def visualize_3d_hand():
    """Animates the 3D plot of hand keypoints over time."""
    plt.ion()  # Enable interactive mode
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    while not exit_flag.is_set():
        ax.clear()
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        if len(hand_frames) == 0:
            plt.pause(0.1)
            continue

        hand_landmarks = hand_frames[-1]  # Get latest frame
        rotated_hand_landmarks = rotate_coordinates(hand_landmarks)

        x = [point[0] for point in rotated_hand_landmarks]
        y = [point[1] for point in rotated_hand_landmarks]
        z = [point[2] for point in rotated_hand_landmarks]

        ax.scatter(x, y, z, c='r', marker='o')

        for connection in HAND_CONNECTIONS:
            ax.plot([x[connection[0]], x[connection[1]]],
                    [y[connection[0]], y[connection[1]]],
                    [z[connection[0]], z[connection[1]]], 'b')

        plt.draw()
        plt.pause(0.1)

    plt.ioff()
    plt.close(fig)  # Close the figure window

def synchronize_and_display():
    """Synchronizes frames from both streams and processes hand tracking."""
    while not exit_flag.is_set():
        with lock:
            if not buffer1 or not buffer2:
                continue

            t1, f1 = buffer1[0]
            t2, f2 = buffer2[0]

            if abs(t1 - t2) <= threshold:
                buffer1.popleft()
                buffer2.popleft()

                h, w, c = f1.shape
                f1 = cv2.resize(f1, (w, h))
                f2 = cv2.resize(f2, (w, h))

                f1_rgb = cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)
                f2_rgb = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)

                results1 = hands.process(f1_rgb)
                results2 = hands.process(f2_rgb)

                keypts1 = []
                keypts2 = []

                if results1.multi_hand_landmarks:
                    for hand_landmarks in results1.multi_hand_landmarks:
                        mp_draw.draw_landmarks(f1, hand_landmarks, HAND_CONNECTIONS)
                        for lndmrk in hand_landmarks.landmark:
                            cx, cy, cz = lndmrk.x * 2 - 1, lndmrk.y * 2 - 1, lndmrk.z * 2 - 1
                            keypts1.append((cx, cy, cz))

                if results2.multi_hand_landmarks:
                    for hand_landmarks in results2.multi_hand_landmarks:
                        mp_draw.draw_landmarks(f2, hand_landmarks, HAND_CONNECTIONS)
                        for lndmrk in hand_landmarks.landmark:
                            cx, cy, cz = lndmrk.x * 2 - 1, lndmrk.y * 2 - 1, lndmrk.z * 2 - 1
                            keypts2.append((cx, cy, cz))

                if len(keypts1) == 21:
                    hand_frames.append(keypts1)
                if len(keypts2) == 21:
                    hand_frames.append(keypts2)

                combined_frame = cv2.hconcat([f1, f2])
                cv2.imshow('Synchronized Hand Tracking', combined_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    exit_flag.set()  # Stop all threads
                    break

            elif t1 < t2:
                buffer1.popleft()
            else:
                buffer2.popleft()

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

# Start capturqqe threads
thread1 = threading.Thread(target=capture_frames, args=(cap1, buffer1), daemon=True)
thread2 = threading.Thread(target=capture_frames, args=(cap2, buffer2), daemon=True)
sync_thread = threading.Thread(target=synchronize_and_display, daemon=True)

thread1.start()
thread2.start()
sync_thread.start()

# Run Matplotlib animation on the main thread
visualize_3d_hand()

# Ensure all threads terminate cleanly
thread1.join()
thread2.join()
sync_thread.join()
