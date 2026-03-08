import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import warnings
import csv
import os
from collections import Counter, deque

warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURATION ---
CSV_FILE = 'multi_target_data.csv'

# --- 1. LOAD BOTH AI BRAINS ---
print("Loading independent AI Brains...")
try:
    with open('gesture_brain.pkl', 'rb') as f:
        gesture_model = pickle.load(f)
    with open('expression_brain.pkl', 'rb') as f:
        expression_model = pickle.load(f)
except FileNotFoundError:
    print("Error: Could not find the .pkl files. Run train_decoupled.py first!")
    exit()

face_cols = [f'{axis}{i}' for i in range(34, 502) for axis in ('x', 'y', 'z', 'v')]
gesture_cols = [f'{axis}{i}' for i in range(1, 34) for axis in ('x', 'y', 'z', 'v')] + \
               [f'{axis}{i}' for i in range(502, 544) for axis in ('x', 'y', 'z', 'v')]

# --- 2. INITIALIZE MEDIAPIPE & CAMERA ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

cam_index = 0
cap = cv2.VideoCapture(cam_index)

# State Variables
show_landmarks = True
full_row = [] 

# Smoothing Buffers (Remembers the last 5 frames)
gest_buffer = deque(maxlen=5)
expr_buffer = deque(maxlen=5)
smooth_gest = "none"
smooth_expr = "neutral"
prob_gest = 0.0
prob_expr = 0.0

print("\n========================================")
print(" LIVE INFERENCE WITH FEEDBACK & SMOOTHING")
print("========================================")
print(" CONTROLS (Click the video window first):")
print("  [t] - Toggle Skeletons On/Off")
print("  [c] - Switch Camera")
print("  [y] - REWARD (Correct! Save 10 frames)")
print("  [w] - PENALIZE (Wrong! Override with 20 frames)")
print("  [q] - Quit")
print("========================================\n")

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: 
            print("Failed to grab frame. Exiting...")
            break

        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # --- 3. KEYBOARD CONTROLS ---
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('t'):
            show_landmarks = not show_landmarks
            
        # CAMERA SWITCH LOGIC
        elif key == ord('c'):
            cap.release()
            cam_index += 1
            cap = cv2.VideoCapture(cam_index)
            # If the next camera index doesn't exist, loop back to 0
            if not cap.isOpened():
                print(f"Camera index {cam_index} not found. Looping back to default camera (0).")
                cam_index = 0
                cap = cv2.VideoCapture(cam_index)
            else:
                print(f"Switched to Camera Index: {cam_index}")
        
        # REWARD LOGIC (Correct Prediction - Saves 10 frames)
        elif key == ord('y') and len(full_row) > 0:
            row_to_save = [smooth_gest, smooth_expr] + full_row
            with open(CSV_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                for _ in range(10):
                    writer.writerow(row_to_save)
            print(f"[REWARD] Saved 10 frames as: {smooth_gest} / {smooth_expr}")

        # PENALIZE LOGIC (Wrong Prediction - Saves 20 frames)
        elif key == ord('w') and len(full_row) > 0:
            print("\n[PENALIZE] Paused. Look at the terminal!")
            correct_gest = input(f"AI guessed Gesture '{smooth_gest}'. Correct it (or press Enter for 'none'): ")
            correct_expr = input(f"AI guessed Face '{smooth_expr}'. Correct it (or press Enter for 'neutral'): ")
            
            final_gest = correct_gest.strip() if correct_gest.strip() != "" else "none"
            final_expr = correct_expr.strip() if correct_expr.strip() != "" else "neutral"
            
            row_to_save = [final_gest, final_expr] + full_row
            
            with open(CSV_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                for _ in range(20):
                    writer.writerow(row_to_save)
                    
            print(f"[CORRECTED] BOOSTER APPLIED! Saved 20 frames as: {final_gest} / {final_expr}.")
            print("Click back on video window to resume!")

        # --- 4. DRAW LANDMARKS ---
        if show_landmarks:
            if results.face_landmarks:
                mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                         mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1))
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # --- 5. DECOUPLED PREDICTION LOGIC ---
        try:
            pose = list(np.array([[p.x, p.y, p.z, p.visibility] for p in results.pose_landmarks.landmark]).flatten()) if results.pose_landmarks else list(np.zeros(33*4))
            face = list(np.array([[p.x, p.y, p.z, p.visibility] for p in results.face_landmarks.landmark]).flatten()) if results.face_landmarks else list(np.zeros(468*4))
            lh = list(np.array([[p.x, p.y, p.z, p.visibility] for p in results.left_hand_landmarks.landmark]).flatten()) if results.left_hand_landmarks else list(np.zeros(21*4))
            rh = list(np.array([[p.x, p.y, p.z, p.visibility] for p in results.right_hand_landmarks.landmark]).flatten()) if results.right_hand_landmarks else list(np.zeros(21*4))

            full_row = pose + face + lh + rh
            gesture_row = pose + lh + rh
            face_row = face
            
            X_gesture = pd.DataFrame([gesture_row], columns=gesture_cols)
            X_face = pd.DataFrame([face_row], columns=face_cols)

            # Raw Predictions
            raw_gest = gesture_model.predict(X_gesture)[0]
            prob_gest = round(np.max(gesture_model.predict_proba(X_gesture)) * 100, 1)
            
            raw_expr = expression_model.predict(X_face)[0]
            prob_expr = round(np.max(expression_model.predict_proba(X_face)) * 100, 1)

            # Apply Smoothing Filter (Majority Vote)
            gest_buffer.append(raw_gest)
            expr_buffer.append(raw_expr)
            
            smooth_gest = Counter(gest_buffer).most_common(1)[0][0]
            smooth_expr = Counter(expr_buffer).most_common(1)[0][0]

            # --- 6. DISPLAY RESULTS ---
            cv2.rectangle(image, (0, 0), (640, 80), (30, 30, 30), -1)
            
            # Display the SMOOTHED predictions
            cv2.putText(image, f"GESTURE: {smooth_gest.upper()} ({prob_gest}%)", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(image, f"FACE:    {smooth_expr.upper()} ({prob_expr}%)", 
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        except Exception as e:
            # If no one is in frame, clear the buffers
            full_row = []
            gest_buffer.clear()
            expr_buffer.clear()
            pass
            
        # Show Toggle & Camera Info
        toggle_text = "Skeletons: ON ('t' to hide) | 'c' to switch camera" if show_landmarks else "Skeletons: OFF ('t' to show) | 'c' to switch camera"
        cv2.putText(image, toggle_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Decoupled AI with Feedback Loop', image)

cap.release()
cv2.destroyAllWindows()