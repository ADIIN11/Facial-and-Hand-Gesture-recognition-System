import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# --- CONFIGURATION ---
CSV_FILE = 'custom_gestures.csv'
# 33 Pose + 468 Face + 21 Left Hand + 21 Right Hand = 543 landmarks. 
# Each has 4 values (x, y, z, visibility) -> 2172 columns.

# --- 1. SETUP CSV HEADERS ---
if not os.path.exists(CSV_FILE):
    landmarks = ['class']
    for val in range(1, 543 + 1):
        landmarks += [f'x{val}', f'y{val}', f'z{val}', f'v{val}']
    
    with open(CSV_FILE, mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(landmarks)

# --- 2. GET LABEL FROM USER ---
print("========================================")
class_name = input("Enter the name of the gesture you are recording (e.g., 'Angry', 'Victory'): ")
print(f"\nReady to record: '{class_name}'")
print("Press 'r' to start/pause recording.")
print("Press 'q' to quit.")
print("========================================")

cap = cv2.VideoCapture(0)
is_recording = False
frame_count = 0

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw Landmarks (so you know what the AI sees)
        if results.face_landmarks:
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                     mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1))
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # --- 3. DATA EXTRACTION & LOGGING ---
        if is_recording:
            try:
                # Extract coordinates. If a body part isn't visible, we pad it with zeros 
                # so the CSV columns don't break.
                pose = list(np.array([[p.x, p.y, p.z, p.visibility] for p in results.pose_landmarks.landmark]).flatten()) if results.pose_landmarks else list(np.zeros(33*4))
                face = list(np.array([[p.x, p.y, p.z, p.visibility] for p in results.face_landmarks.landmark]).flatten()) if results.face_landmarks else list(np.zeros(468*4))
                lh = list(np.array([[p.x, p.y, p.z, p.visibility] for p in results.left_hand_landmarks.landmark]).flatten()) if results.left_hand_landmarks else list(np.zeros(21*4))
                rh = list(np.array([[p.x, p.y, p.z, p.visibility] for p in results.right_hand_landmarks.landmark]).flatten()) if results.right_hand_landmarks else list(np.zeros(21*4))

                # Combine all rows and add the class name at the beginning
                row = pose + face + lh + rh
                row.insert(0, class_name)

                # Append to CSV
                with open(CSV_FILE, mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row)
                
                frame_count += 1
                cv2.putText(image, f"RECORDING '{class_name}': {frame_count} frames", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            except Exception as e:
                pass # Skip frames where an error occurs
        else:
            cv2.putText(image, "PAUSED - Press 'r' to record", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Data Collector', image)

        # Key Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            is_recording = not is_recording # Toggle recording
        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()