import cv2
import mediapipe as mp
import time
import threading

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize Video Capture
cap1 = cv2.VideoCapture(0)  # Laptop Webcam
cap2 = cv2.VideoCapture(1)  # Phone Camera via Iriun

# Buffers to store frames and timestamps
buffer1 = []
buffer2 = []
threshold = 0.03  # 30ms threshold for synchronization
lock = threading.Lock()

def process_frame(frame):
    """Extracts hand landmarks from a given frame."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    return frame

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

def synchronize_and_display():
    """Synchronizes frames from both streams and displays them side by side."""
    while True:
        with lock:
            if not buffer1 or not buffer2:
                continue
            
            # Match frames based on timestamp difference
            t1, f1 = buffer1[0]
            t2, f2 = buffer2[0]
            
            if abs(t1 - t2) <= threshold:
                buffer1.pop(0)
                buffer2.pop(0)
                f1 = process_frame(f1)
                f2 = process_frame(f2)
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
