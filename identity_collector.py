import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# --- CONFIGURATION ---
CSV_FILE = 'identity_data.csv'

# --- 1. SETUP SINGLE-HEADER CSV (Face Only) ---
if not os.path.exists(CSV_FILE):
    print(f"[*] Creating new identity dataset file: {CSV_FILE}")
    landmarks = ['person_name']
    for val in range(1, 468 + 1): # 468 face landmarks
        landmarks += [f'x{val}', f'y{val}', f'z{val}', f'v{val}']
    
    with open(CSV_FILE, mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(landmarks)
else:
    with open(CSV_FILE, 'r') as f:
        row_count = sum(1 for row in f) - 1 
    print(f"[*] Found existing dataset '{CSV_FILE}'.")
    print(f"[*] Currently contains {row_count} saved faces. Appending new data to the bottom!")

# --- 2. INITIAL STATE ---
cap = cv2.VideoCapture(0)
is_recording = False
person_name = "unknown"
frame_count = 0

print("========================================")
print(" ADVANCED IDENTITY COLLECTOR STARTED")
print("========================================")
print("HOW TO USE (Click on the Video Window First!):")
print(" -> Press 'n' to enter NEW PERSON NAME")
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

        # Draw only Face Landmarks (since this is the Identity Collector)
        if results.face_landmarks:
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                     mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1))

        # --- 3. KEYBOARD CONTROLS ---
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            if person_name == "unknown":
                print("WARNING: You are recording baseline data (unknown)!")
            is_recording = not is_recording
        
        # --- THE NAME LOGIC ---
        elif key == ord('n'):
            is_recording = False 
            print("\n--- ENTER NEW IDENTITY ---")
            
            raw_name = input("Enter Person Name (e.g., Adith) or press Enter for 'unknown': ")
            
            # Apply default if input is blank
            person_name = raw_name.strip() if raw_name.strip() != "" else "unknown"
            
            frame_count = 0 
            print(f"\n=> Ready! Identity set to: [{person_name}]. Click back on video window and press 'r' to record.")

        # --- 4. DATA LOGGING ---
        if is_recording:
            try:
                # We only care about the face for identity!
                if results.face_landmarks:
                    face = list(np.array([[p.x, p.y, p.z, p.visibility] for p in results.face_landmarks.landmark]).flatten())
                    row = [person_name] + face

                    with open(CSV_FILE, mode='a', newline='') as f:
                        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(row)
                    
                    frame_count += 1
                    
                    cv2.rectangle(image, (0,0), (640, 40), (0, 0, 255), -1)
                    cv2.putText(image, f"REC | Name: {person_name} | Frames: {frame_count}", 
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    # Visual warning if you step out of frame while recording
                    cv2.rectangle(image, (0,0), (640, 40), (0, 165, 255), -1)
                    cv2.putText(image, "REC | WAITING FOR FACE...", 
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            except Exception as e:
                pass 
        else:
            cv2.rectangle(image, (0,0), (640, 40), (255, 0, 0), -1)
            cv2.putText(image, f"PAUSED | Name: {person_name} | Press 'n' to change", 
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Advanced Identity Collector', image)

cap.release()
cv2.destroyAllWindows()