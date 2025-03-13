import cv2
import mediapipe as mp
import numpy as np
import socket
import json
import time
import threading

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize Video Capture
cap1 = cv2.VideoCapture(0)  # Laptop Webcam
cap2 = cv2.VideoCapture(1)  # Phone Camera via Iriun

# Buffers and threading lock
buffer1, buffer2 = [], []
threshold = 0.03  # 30ms threshold for synchronization
lock = threading.Lock()

# Define Projection Matrices (Replace with real values)
P1 = np.array([[1000, 0, 320, 0], [0, 1000, 240, 0], [0, 0, 1, 0]])  
P2 = np.array([[1000, 0, 320, -100], [0, 1000, 240, 0], [0, 0, 1, 0]])  

# Start Socket Connection
HOST, PORT = "localhost", 65432
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))

def extract_landmarks(frame):
    """Extracts hand landmarks from a given frame and returns a list of (x, y) coordinates."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            return [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in hand_landmarks.landmark]
    return None

def capture_frames(cap, buffer):
    """Continuously captures frames with timestamps and stores them in a buffer."""
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = time.time()
        with lock:
            buffer.append((timestamp, frame))

# Start capture threads
thread1 = threading.Thread(target=capture_frames, args=(cap1, buffer1))
thread2 = threading.Thread(target=capture_frames, args=(cap2, buffer2))
thread1.start()
thread2.start()

def triangulate_dlt(P1, P2, point1, point2):
    """Computes 3D coordinates from two 2D points using Direct Linear Transform (DLT)."""
    x1, y1 = point1
    x2, y2 = point2

    A = np.array([
        x1 * P1[2] - P1[0],
        y1 * P1[2] - P1[1],
        x2 * P2[2] - P2[0],
        y2 * P2[2] - P2[1]
    ])

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X /= X[3]  # Normalize to homogeneous coordinates

    return X[:3]  # Return (X, Y, Z) coordinates

def send_data(data):
    """Sends JSON-serializable 3D data over the socket."""
    try:
        json_data = json.dumps(data.tolist() if isinstance(data, np.ndarray) else data)
        sock.sendall(json_data.encode())
    except Exception as e:
        print(f"Socket Error: {e}")

def synchronize_and_send():
    """Synchronizes frames, extracts landmarks, and computes 3D coordinates."""
    while True:
        with lock:
            if not buffer1 or not buffer2:
                continue

            t1, f1 = buffer1[0]
            t2, f2 = buffer2[0]

            if abs(t1 - t2) <= threshold:
                buffer1.pop(0)
                buffer2.pop(0)

                landmarks1 = extract_landmarks(f1)
                landmarks2 = extract_landmarks(f2)

                if landmarks1 and landmarks2 and len(landmarks1) == len(landmarks2):
                    frame_3d = [triangulate_dlt(P1, P2, landmarks1[i], landmarks2[i]) for i in range(len(landmarks1))]
                    send_data(frame_3d)  # ✅ Send data to `plot_3d.py`
                    print(f"Sent 3D Points: {frame_3d}")  # Debugging output

                cv2.imshow("Synchronized Hand Tracking", cv2.hconcat([f1, f2]))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            elif t1 < t2:
                buffer1.pop(0)
            else:
                buffer2.pop(0)

# Start data synchronization
synchronize_and_send()


# Cleanup
cap1.release()
cap2.release()
sock.close()
cv2.destroyAllWindows()
