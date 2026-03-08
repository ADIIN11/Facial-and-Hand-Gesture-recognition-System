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
ID_CSV_FILE = 'identity_data.csv'

# --- 1. LOAD 3 INDEPENDENT AI BRAINS ---
print("Loading 3 independent AI Brains...")
try:
    with open('gesture_brain.pkl', 'rb') as f:
        gesture_model = pickle.load(f)
    with open('expression_brain.pkl', 'rb') as f:
        expression_model = pickle.load(f)
    with open('identity_brain.pkl', 'rb') as f:
        identity_model = pickle.load(f)
except FileNotFoundError:
    print("Error: Could not find all .pkl files. Ensure you trained all 3 brains!")
    exit()

# Setup column structures to match the trainers
face_cols = [f'{axis}{i}' for i in range(34, 502) for axis in ('x', 'y', 'z', 'v')]
gesture_cols = [f'{axis}{i}' for i in range(1, 34) for axis in ('x', 'y', 'z', 'v')] + \
               [f'{axis}{i}' for i in range(502, 544) for axis in ('x', 'y', 'z', 'v')]
id_cols = [f'{axis}{i}' for i in range(1, 469) for axis in ('x', 'y', 'z', 'v')]

# --- 2. INITIALIZE MEDIAPIPE & CAMERA ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

cam_index = 0
cap = cv2.VideoCapture(cam_index)

# State Variables
show_landmarks = True
show_face_box = True  
full_row = [] 
current_face_row = [] 

# Smoothing & Locking Buffers
gest_buffer = deque(maxlen=5)
expr_buffer = deque(maxlen=5)
id_buffer = deque(maxlen=5)

smooth_gest = "none"
smooth_expr = "neutral"
smooth_id = "unknown"
locked_id = None 
prob_gest = 0.0
prob_expr = 0.0

# --- MOUSE CALLBACK VARIABLES & FUNCTION ---
trigger_gest_reward = False
trigger_gest_penalize = False

def mouse_click(event, x, y, flags, param):
    global trigger_gest_reward, trigger_gest_penalize
    if event ==  cv2.EVENT_RBUTTONDOWN:
        trigger_gest_penalize = True

# Setup window and attach mouse callback
cv2.namedWindow('Tri-Core AI Engine')
cv2.setMouseCallback('Tri-Core AI Engine', mouse_click)

print("\n========================================")
print(" TRI-CORE INFERENCE WITH FEEDBACK LOOP")
print("========================================")
print(" CONTROLS (Click the video window first):")
print("  [t] - Toggle Skeletons On/Off")
print("  [b] - Toggle Face Box On/Off")
print("  [c] - Switch Camera")
print("  [y] or [L-Click] - REWARD GEST/EXPR (Save 20 frames)")
print("  [w] or [R-Click] - PENALIZE GEST/EXPR (Override with 50 frames)")
print("  [u] - REWARD IDENTITY (Save 40 frames)")
print("  [i] - PENALIZE IDENTITY (Override with 100 frames)")
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
        h, w, _ = image.shape

        # --- 3. KEYBOARD & MOUSE CONTROLS ---
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('t'):
            show_landmarks = not show_landmarks
        elif key == ord('b'):
            show_face_box = not show_face_box 
            
        # CAMERA SWITCH LOGIC
        elif key == ord('c'):
            cap.release()
            cam_index += 1
            cap = cv2.VideoCapture(cam_index)
            if not cap.isOpened():
                print(f"Camera index {cam_index} not found. Looping back to default camera (0).")
                cam_index = 0
                cap = cv2.VideoCapture(cam_index)
            else:
                print(f"Switched to Camera Index: {cam_index}")
        
        # --- REWARD/PENALIZE (Gesture & Expression) ---
        elif (key == ord('y') or trigger_gest_reward) and len(full_row) > 0:
            row_to_save = [smooth_gest, smooth_expr] + full_row
            with open(CSV_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                for _ in range(20): # CHANGED TO 20
                    writer.writerow(row_to_save)
            print(f"[REWARD] Saved 20 frames as: {smooth_gest} / {smooth_expr}")
            trigger_gest_reward = False # Reset flag

        elif (key == ord('w') or trigger_gest_penalize) and len(full_row) > 0:
            print("\n[PENALIZE] Paused. Look at the terminal!")
            correct_gest = input(f"AI guessed Gesture '{smooth_gest}'. Correct it (or press Enter for 'none'): ")
            correct_expr = input(f"AI guessed Face '{smooth_expr}'. Correct it (or press Enter for 'neutral'): ")
            
            final_gest = correct_gest.strip() if correct_gest.strip() != "" else "none"
            final_expr = correct_expr.strip() if correct_expr.strip() != "" else "neutral"
            
            row_to_save = [final_gest, final_expr] + full_row
            
            with open(CSV_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                for _ in range(50): # CHANGED TO 50
                    writer.writerow(row_to_save)
                    
            print(f"[CORRECTED] BOOSTER APPLIED! Saved 50 frames as: {final_gest} / {final_expr}.")
            print("Click back on video window to resume!")
            trigger_gest_penalize = False # Reset flag

        # Reset flags if triggered while tracking was lost (prevents delayed triggers)
        trigger_gest_reward = False
        trigger_gest_penalize = False

        # --- REWARD/PENALIZE (Identity) ---
        if key == ord('u') and len(current_face_row) > 0:
            save_id = locked_id if locked_id else smooth_id
            row_to_save = [save_id] + current_face_row
            with open(ID_CSV_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                for _ in range(40):
                    writer.writerow(row_to_save)
            print(f"[ID REWARD] Saved 40 frames for Identity: {save_id}")

        elif key == ord('i') and len(current_face_row) > 0:
            print("\n[ID PENALIZE] Paused. Look at the terminal!")
            current_guess = locked_id if locked_id else smooth_id
            correct_id = input(f"AI guessed Identity '{current_guess}'. Correct it: ")
            
            final_id = correct_id.strip() if correct_id.strip() != "" else "unknown"
            
            row_to_save = [final_id] + current_face_row
            with open(ID_CSV_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                for _ in range(100):
                    writer.writerow(row_to_save)
                    
            print(f"[ID CORRECTED] BOOSTER APPLIED! Saved 100 frames as: {final_id}.")
            locked_id = final_id 
            print("Click back on video window to resume!")

        # --- 4. DRAW SKELETONS ---
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

        # --- 5. TRI-CORE PREDICTION LOGIC ---
        try:
            if not results.face_landmarks:
                locked_id = None
                id_buffer.clear()
            
            pose = list(np.array([[p.x, p.y, p.z, p.visibility] for p in results.pose_landmarks.landmark]).flatten()) if results.pose_landmarks else list(np.zeros(33*4))
            face = list(np.array([[p.x, p.y, p.z, p.visibility] for p in results.face_landmarks.landmark]).flatten()) if results.face_landmarks else list(np.zeros(468*4))
            lh = list(np.array([[p.x, p.y, p.z, p.visibility] for p in results.left_hand_landmarks.landmark]).flatten()) if results.left_hand_landmarks else list(np.zeros(21*4))
            rh = list(np.array([[p.x, p.y, p.z, p.visibility] for p in results.right_hand_landmarks.landmark]).flatten()) if results.right_hand_landmarks else list(np.zeros(21*4))

            full_row = pose + face + lh + rh
            gesture_row = pose + lh + rh
            current_face_row = face 
            
            X_gesture = pd.DataFrame([gesture_row], columns=gesture_cols)
            X_face = pd.DataFrame([face], columns=face_cols)
            X_id = pd.DataFrame([face], columns=id_cols) 

            # Gesture & Expression Predictions
            raw_gest = gesture_model.predict(X_gesture)[0]
            prob_gest = round(np.max(gesture_model.predict_proba(X_gesture)) * 100, 1)
            
            raw_expr = expression_model.predict(X_face)[0]
            prob_expr = round(np.max(expression_model.predict_proba(X_face)) * 100, 1)

            gest_buffer.append(raw_gest)
            expr_buffer.append(raw_expr)
            smooth_gest = Counter(gest_buffer).most_common(1)[0][0]
            smooth_expr = Counter(expr_buffer).most_common(1)[0][0]

            # Identity Prediction Logic
            if results.face_landmarks:
                if locked_id is None:
                    id_probs = identity_model.predict_proba(X_id)[0]
                    max_id_prob = np.max(id_probs)
                    
                    if max_id_prob < 0.70:
                        raw_id = "Unknown"
                    else:
                        raw_id = identity_model.predict(X_id)[0]
                    
                    id_buffer.append(raw_id)
                    smooth_id = Counter(id_buffer).most_common(1)[0][0]
                    
                    if smooth_id != "Unknown" and len(id_buffer) == 5:
                        locked_id = smooth_id
                else:
                    smooth_id = locked_id
            else:
                smooth_id = "Unknown"

            # --- 6. DRAW FACE BOUNDING BOX ---
            if show_face_box and results.face_landmarks:
                x_max, y_max = 0, 0
                x_min, y_min = w, h
                
                for lm in results.face_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max: x_max = x
                    if x < x_min: x_min = x
                    if y > y_max: y_max = y
                    if y < y_min: y_min = y
                
                cv2.rectangle(image, (x_min - 15, y_min - 40), (x_max + 15, y_max + 15), (255, 100, 100), 2)
                cv2.putText(image, smooth_id.upper(), (x_min - 10, y_min - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)

            # --- 7. DISPLAY TOP PANEL RESULTS ---
            cv2.rectangle(image, (0, 0), (640, 115), (30, 30, 30), -1)
            
            cv2.putText(image, f"PERSON:  {smooth_id.upper()}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
            cv2.putText(image, f"GESTURE: {smooth_gest.upper()} ({prob_gest}%)", 
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(image, f"FACE:    {smooth_expr.upper()} ({prob_expr}%)", 
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        except Exception as e:
            full_row = []
            current_face_row = []
            gest_buffer.clear()
            expr_buffer.clear()
            id_buffer.clear()
            locked_id = None
            pass
            
        # Show Toggle Controls Info at the bottom
        controls_text = "'r'/R-Click: Gest. Feedbk | 'u'/'i': ID Feedbk"
        cv2.putText(image, controls_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Tri-Core AI Engine', image)

cap.release()
cv2.destroyAllWindows()