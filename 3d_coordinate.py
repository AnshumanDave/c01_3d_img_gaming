import cv2
import mediapipe as mp
import time
import threading
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize Video Capture
cap1 = cv2.VideoCapture(0)  # Laptop Webcam
cap2 = cv2.VideoCapture(1)  # Phone Camera via Iriun

# Buffers to store frames and timestamps
buffer1, buffer2 = [], []
threshold = 0.03  # 30ms threshold for synchronization
lock = threading.Lock()

# Projection matrices for cameras (Replace with actual values after calibration)
P1 = np.array([[1000, 0, 320, 0], [0, 1000, 240, 0], [0, 0, 1, 0]])  # Example
P2 = np.array([[1000, 0, 320, -100], [0, 1000, 240, 0], [0, 0, 1, 0]])  # Example

def extract_landmarks(frame):
    """Extracts hand landmarks from a given frame and returns a list of (x, y) coordinates."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            return [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in hand_landmarks.landmark]  # Convert to pixel coords
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
    """
    Computes 3D coordinates from two 2D points and their respective projection matrices using Direct Linear Transform (DLT).
    
    Steps:
    1. Accepts projection matrices (P1, P2) and 2D points (point1, point2).
    2. Constructs a linear system based on the projection equations.
    3. Stacks the equations into a 4×4 matrix.
    4. Computes the SVD (Singular Value Decomposition) of the matrix.
    5. Extracts and normalizes the last row of the V matrix to obtain the homogeneous 3D coordinate.
    6. Returns the (X, Y, Z) Cartesian coordinate.
    """
    x1, y1 = point1
    x2, y2 = point2

    # Construct the system of equations
    A = np.array([
        x1 * P1[2] - P1[0],  # First equation from Camera 1
        y1 * P1[2] - P1[1],  # Second equation from Camera 1
        x2 * P2[2] - P2[0],  # First equation from Camera 2
        y2 * P2[2] - P2[1]   # Second equation from Camera 2
    ])

    # Perform SVD (Singular Value Decomposition)
    U, S, Vt = np.linalg.svd(A)
    X = Vt[-1]  # Extract last row of V (corresponds to the smallest singular value)

    # Convert homogeneous coordinates to Cartesian by normalizing
    X /= X[3]  # Normalize by the last coordinate

    return X[:3]  # Return (X, Y, Z) coordinates

def synchronize_and_display():
    """Synchronizes frames from both streams, extracts landmarks, and computes 3D coordinates."""
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

                if landmarks1 and landmarks2:
                    if len(landmarks1) == len(landmarks2):  # Ensure matching points
                        for i in range(len(landmarks1)):
                            point_3d = triangulate_dlt(P1, P2, landmarks1[i], landmarks2[i])
                            print(f"3D Point {i}: {point_3d}")  

                combined_frame = cv2.hconcat([f1, f2])
                cv2.imshow("Synchronized Hand Tracking", combined_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            elif t1 < t2:
                buffer1.pop(0)
            else:
                buffer2.pop(0)

# Start synchronization and display
synchronize_and_display()

# Release resources
cap1.release()
cap2.release()
cv2.destroyAllWindows()