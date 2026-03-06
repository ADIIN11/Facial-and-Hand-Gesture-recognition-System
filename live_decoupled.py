import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. LOAD BOTH AI BRAINS ---
print("Loading independent AI Brains...")
with open('gesture_brain.pkl', 'rb') as f:
    gesture_model = pickle.load(f)
with open('expression_brain.pkl', 'rb') as f:
    expression_model = pickle.load(f)

# Recreate the exact column names to prevent Scikit-Learn warnings
face_cols = [f'{axis}{i}' for i in range(34, 502) for axis in ('x', 'y', 'z', 'v')]
gesture_cols = [f'{axis}{i}' for i in range(1, 34) for axis in ('x', 'y', 'z', 'v')] + \
               [f'{axis}{i}' for i in range(502, 544) for axis in ('x', 'y', 'z', 'v')]

# --- 2. INITIALIZE MEDIAPIPE ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

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

        # --- 3. DECOUPLED PREDICTION LOGIC ---
        try:
            pose = list(np.array([[p.x, p.y, p.z, p.visibility] for p in results.pose_landmarks.landmark]).flatten()) if results.pose_landmarks else list(np.zeros(33*4))
            face = list(np.array([[p.x, p.y, p.z, p.visibility] for p in results.face_landmarks.landmark]).flatten()) if results.face_landmarks else list(np.zeros(468*4))
            lh = list(np.array([[p.x, p.y, p.z, p.visibility] for p in results.left_hand_landmarks.landmark]).flatten()) if results.left_hand_landmarks else list(np.zeros(21*4))
            rh = list(np.array([[p.x, p.y, p.z, p.visibility] for p in results.right_hand_landmarks.landmark]).flatten()) if results.right_hand_landmarks else list(np.zeros(21*4))

            # Split the live coordinates to match the independent brains
            gesture_row = pose + lh + rh
            face_row = face
            
            X_gesture = pd.DataFrame([gesture_row], columns=gesture_cols)
            X_face = pd.DataFrame([face_row], columns=face_cols)

            # Predict Gestures (Brain 1)
            pred_gest = gesture_model.predict(X_gesture)[0]
            prob_gest = round(np.max(gesture_model.predict_proba(X_gesture)) * 100, 1)
            
            # Predict Expressions (Brain 2)
            pred_expr = expression_model.predict(X_face)[0]
            prob_expr = round(np.max(expression_model.predict_proba(X_face)) * 100, 1)

            # --- 4. DISPLAY THE RESULTS ON SCREEN ---
            cv2.rectangle(image, (0, 0), (640, 80), (30, 30, 30), -1)
            cv2.putText(image, f"GESTURE: {pred_gest.upper()} ({prob_gest}%)", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(image, f"FACE:    {pred_expr.upper()} ({prob_expr}%)", 
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        except Exception as e:
            pass

        cv2.imshow('Decoupled AI Action Recognition', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()