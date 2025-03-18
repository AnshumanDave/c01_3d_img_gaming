import cv2
import mediapipe as mp
import time
import threading
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import keyboard  # Requires 'pip install keyboard'

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
exit_flag = threading.Event()
apply_transformation_flag = True  # Toggle transformation
x_angle, y_angle, z_angle = 0, 0, 20  # Default rotation values
translation_vector = np.array([0, 0, 0])


def get_rotation_matrix(x_angle=0, y_angle=0, z_angle=0):
    """Generates a combined rotation matrix for given angles."""
    x_rad, y_rad, z_rad = np.radians([x_angle, y_angle, z_angle])
    
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(x_rad), -np.sin(x_rad)],
                    [0, np.sin(x_rad), np.cos(x_rad)]])
    
    R_y = np.array([[np.cos(y_rad), 0, np.sin(y_rad)],
                    [0, 1, 0],
                    [-np.sin(y_rad), 0, np.cos(y_rad)]])
    
    R_z = np.array([[np.cos(z_rad), -np.sin(z_rad), 0],
                    [np.sin(z_rad), np.cos(z_rad), 0],
                    [0, 0, 1]])
    
    return R_z @ R_y @ R_x  # Combined rotation matrix


def apply_transformation(points, rotation_matrix, translation_vector):
    """Applies a rotation and translation to a set of 3D points."""
    points_np = np.array(points).T  # Convert list of tuples into a 3xN NumPy array
    transformed_points = rotation_matrix @ points_np + translation_vector.reshape(3, 1)
    return list(map(tuple, transformed_points.T))


def visualize_3d_hand():
    """Animates the 3D plot of hand keypoints over time with transformations."""
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    global apply_transformation_flag, x_angle, y_angle, z_angle

    while not exit_flag.is_set():
        ax.clear()
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Hand Tracking Visualization")

        # Handle key inputs for transformations
        if keyboard.is_pressed("t"):  # Toggle transformation
            apply_transformation_flag = not apply_transformation_flag
            time.sleep(0.3)  # Prevent rapid toggling

        if keyboard.is_pressed("up"):
            x_angle += 5
        if keyboard.is_pressed("down"):
            x_angle -= 5
        if keyboard.is_pressed("left"):
            y_angle -= 5
        if keyboard.is_pressed("right"):
            y_angle += 5

        rotation_matrix = get_rotation_matrix(x_angle, y_angle, z_angle)

        if len(hand_frames) == 0:
            plt.pause(0.1)
            continue

        hand_landmarks = hand_frames[-1]
        transformed_landmarks = apply_transformation(hand_landmarks, rotation_matrix, translation_vector) if apply_transformation_flag else hand_landmarks

        x, y, z = zip(*transformed_landmarks)
        ax.scatter(x, y, z, c='r', marker='o')
        for connection in HAND_CONNECTIONS:
            ax.plot([x[connection[0]], x[connection[1]]],
                    [y[connection[0]], y[connection[1]]],
                    [z[connection[0]], z[connection[1]]], 'b')

        plt.draw()
        plt.pause(0.1)

    plt.ioff()
    plt.close(fig)


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


# Start capture threads
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
