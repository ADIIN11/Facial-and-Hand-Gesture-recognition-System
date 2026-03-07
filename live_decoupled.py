import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import warnings
import csv
import os

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

# --- 2. INITIALIZE MEDIAPIPE ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# State Variables
show_landmarks = True
pred_gest = "None"
pred_expr = "None"
full_row = [] # Will hold the 543 coordinates for the feedback loop

print("\n========================================")
print(" LIVE INFERENCE WITH FEEDBACK LOOP")
print("========================================")
print(" CONTROLS (Click the video window first):")
print("  [t] - Toggle Skeletons On/Off")
print("  [y] - REWARD (Correct! Save to dataset)")
print("  [w] - PENALIZE (Wrong! Enter correction)")
print("  [q] - Quit")
print("========================================\n")

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # --- 3. KEYBOARD CONTROLS ---
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('t'):
            show_landmarks = not show_landmarks # Toggle boolean
        
        # REWARD LOGIC (Correct Prediction)
        elif key == ord('y') and len(full_row) > 0:
            row_to_save = [pred_gest, pred_expr] + full_row
            with open(CSV_FILE, mode='a', newline='') as f:
                csv.writer(f).writerow(row_to_save)
            print(f"[REWARD] Saved frame as: {pred_gest} / {pred_expr}")

        # PENALIZE LOGIC (Wrong Prediction)
        elif key == ord('w') and len(full_row) > 0:
            print("\n[PENALIZE] Paused. Look at the terminal!")
            correct_gest = input(f"AI guessed Gesture '{pred_gest}'. What is it REALLY? (or press Enter to keep): ")
            correct_expr = input(f"AI guessed Face '{pred_expr}'. What is it REALLY? (or press Enter to keep): ")
            
            # If you just hit Enter, it keeps the original prediction
            final_gest = correct_gest if correct_gest != "" else pred_gest
            final_expr = correct_expr if correct_expr != "" else pred_expr
            
            row_to_save = [final_gest, final_expr] + full_row
            with open(CSV_FILE, mode='a', newline='') as f:
                csv.writer(f).writerow(row_to_save)
            print(f"[CORRECTED] Saved frame as: {final_gest} / {final_expr}. Click back on video to resume!")

        # --- 4. DRAW LANDMARKS (Based on Toggle) ---
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

            # Save the full row so it can be used by the Reward/Penalize keys on the next loop
            full_row = pose + face + lh + rh

            # Split for prediction
            gesture_row = pose + lh + rh
            face_row = face
            
            X_gesture = pd.DataFrame([gesture_row], columns=gesture_cols)
            X_face = pd.DataFrame([face_row], columns=face_cols)

            # Predict
            pred_gest = gesture_model.predict(X_gesture)[0]
            prob_gest = round(np.max(gesture_model.predict_proba(X_gesture)) * 100, 1)
            
            pred_expr = expression_model.predict(X_face)[0]
            prob_expr = round(np.max(expression_model.predict_proba(X_face)) * 100, 1)

            # --- 6. DISPLAY RESULTS ---
            cv2.rectangle(image, (0, 0), (640, 80), (30, 30, 30), -1)
            cv2.putText(image, f"GESTURE: {pred_gest.upper()} ({prob_gest}%)", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(image, f"FACE:    {pred_expr.upper()} ({prob_expr}%)", 
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Show Toggle Status
            toggle_text = "Skeletons: ON (Press 't' to hide)" if show_landmarks else "Skeletons: OFF (Press 't' to show)"
            cv2.putText(image, toggle_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        except Exception as e:
            full_row = [] # Clear buffer if nobody is in frame
            pass

        cv2.imshow('Decoupled AI with Feedback Loop', image)

cap.release()
cv2.destroyAllWindows()