import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# Initialize MediaPipe (Legacy Version)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# --- CONFIGURATION ---
CSV_FILE = 'multi_target_data.csv'

# --- 1. SETUP DUAL-HEADER CSV ---
if not os.path.exists(CSV_FILE):
    print(f"[*] Creating new dataset file: {CSV_FILE}")
    # Notice we now have TWO label columns at the start
    landmarks = ['gesture_class', 'expression_class']
    for val in range(1, 543 + 1):
        landmarks += [f'x{val}', f'y{val}', f'z{val}', f'v{val}']
    
    with open(CSV_FILE, mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(landmarks)
else:
    # If the file exists, let's count the rows to give you peace of mind!
    with open(CSV_FILE, 'r') as f:
        row_count = sum(1 for row in f) - 1 # Subtract 1 so we don't count the header row
    print(f"[*] Found existing dataset '{CSV_FILE}'.")
    print(f"[*] Currently contains {row_count} saved frames. Appending new data to the bottom!")

# --- 2. INITIAL STATE ---
cap = cv2.VideoCapture(0)
is_recording = False
gesture_name = "None"
expression_name = "None"
frame_count = 0

print("========================================")
print(" ADVANCED DATA COLLECTOR STARTED")
print("========================================")
print("HOW TO USE (Click on the Video Window First!):")
print(" -> Press 'n' to enter NEW labels")
print(" -> Press 'r' to start/pause RECORDING")
print(" -> Press 'q' to QUIT")
print("========================================")

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw Landmarks
        if results.face_landmarks:
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                     mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1))
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # --- 3. KEYBOARD CONTROLS ---
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Prevent recording if no labels are set
            if gesture_name == "None" and expression_name == "None":
                print("WARNING: Press 'n' to set your labels before recording!")
            else:
                is_recording = not is_recording
        elif key == ord('n'):
            is_recording = False # Always pause recording when typing
            print("\n--- ENTER NEW LABELS ---")
            # We use the terminal to get input. 
            gesture_name = input("Enter Hand Gesture (e.g., Peace, Fist, None): ")
            expression_name = input("Enter Face Expression (e.g., Smile, Angry, Neutral): ")
            frame_count = 0 # Reset frame counter for the new batch
            print(f"\n=> Ready! Click back on the video window and press 'r' to record.")

        # --- 4. DATA LOGGING ---
        if is_recording:
            try:
                pose = list(np.array([[p.x, p.y, p.z, p.visibility] for p in results.pose_landmarks.landmark]).flatten()) if results.pose_landmarks else list(np.zeros(33*4))
                face = list(np.array([[p.x, p.y, p.z, p.visibility] for p in results.face_landmarks.landmark]).flatten()) if results.face_landmarks else list(np.zeros(468*4))
                lh = list(np.array([[p.x, p.y, p.z, p.visibility] for p in results.left_hand_landmarks.landmark]).flatten()) if results.left_hand_landmarks else list(np.zeros(21*4))
                rh = list(np.array([[p.x, p.y, p.z, p.visibility] for p in results.right_hand_landmarks.landmark]).flatten()) if results.right_hand_landmarks else list(np.zeros(21*4))

                # Combine all rows
                row = pose + face + lh + rh
                
                # Insert BOTH labels at the start of the row
                row.insert(0, expression_name)
                row.insert(0, gesture_name)

                with open(CSV_FILE, mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row)
                
                frame_count += 1
                
                # Visual feedback for recording
                cv2.rectangle(image, (0,0), (640, 40), (0, 0, 255), -1)
                cv2.putText(image, f"REC | Gest: {gesture_name} | Expr: {expression_name} | Frames: {frame_count}", 
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            except Exception as e:
                pass 
        else:
            # Visual feedback for paused state
            cv2.rectangle(image, (0,0), (640, 40), (255, 0, 0), -1)
            cv2.putText(image, f"PAUSED | Gest: {gesture_name} | Expr: {expression_name} | Press 'n' to change", 
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Advanced Data Collector', image)

cap.release()
cv2.destroyAllWindows()