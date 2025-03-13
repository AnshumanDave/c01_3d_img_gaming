import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

# Define hand skeleton connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]

# Open two video streams
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

# Buffers for synchronization
buffer1, buffer2 = deque(), deque()
thresh_ms = 50


def current_milli_time():
    return int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)


def extract_hand_landmarks(frame):
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    keypoints = []
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                keypoints.append((lm.x * w, lm.y * h, lm.z))
    return keypoints


def triangulate_point_dlt(P1, P2, point1, point2):
    A = np.array([
        point1[0] * P1[2] - P1[0],
        point1[1] * P1[2] - P1[1],
        point2[0] * P2[2] - P2[0],
        point2[1] * P2[2] - P2[1]
    ])
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


# Set up Matplotlib for real-time plotting
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121, projection='3d')
ax.set_xlim([-0.5, 0.5])
ax.set_ylim([-0.5, 0.5])
ax.set_zlim([-0.5, 0.5])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
scatter = ax.scatter([], [], [])
lines = [ax.plot([], [], [])[0] for _ in HAND_CONNECTIONS]
plt.ion()
plt.show()


def transform_coordinates(points, rotation_matrix, translation_vector):
    return np.dot(rotation_matrix, points.T).T + translation_vector


def update_3d_plot(data):
    if data is not None and len(data) == 21:
        scatter._offsets3d = (data[:, 0], data[:, 1], data[:, 2])
        for i, (start, end) in enumerate(HAND_CONNECTIONS):
            lines[i].set_data([data[start, 0], data[end, 0]], [data[start, 1], data[end, 1]])
            lines[i].set_3d_properties([data[start, 2], data[end, 2]])
        plt.draw()
        plt.pause(0.001)


while cap1.isOpened() and cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        break

    ts = current_milli_time()
    buffer1.append((ts, frame1))
    buffer2.append((ts, frame2))

    while buffer1 and buffer2:
        t1, f1 = buffer1[0]
        t2, f2 = buffer2[0]

        if abs(t1 - t2) <= thresh_ms:
            buffer1.popleft()
            buffer2.popleft()
            keypoints1 = extract_hand_landmarks(f1)
            keypoints2 = extract_hand_landmarks(f2)
            if len(keypoints1) == 21 and len(keypoints2) == 21:
                proj_matrix1 = np.eye(3, 4)
                proj_matrix2 = np.array([[1, 0, 0, -0.1], [0, 1, 0, 0], [0, 0, 1, 0]])
                hand_3d_frame = np.array(
                    [triangulate_point_dlt(proj_matrix1, proj_matrix2, keypoints1[i], keypoints2[i]) for i in
                     range(21)])

                # Apply rotation transformation
                rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  # Example: Align Z upwards
                translation_vector = np.array([0, 0, 0])
                hand_3d_frame = transform_coordinates(hand_3d_frame, rotation_matrix, translation_vector)

                update_3d_plot(hand_3d_frame)

            combined_frame = cv2.hconcat([f1, f2])
            cv2.imshow("Camera Feeds", combined_frame)
        elif t1 < t2:
            buffer1.popleft()
        else:
            buffer2.popleft()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
