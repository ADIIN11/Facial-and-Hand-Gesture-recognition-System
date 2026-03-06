import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import warnings

# Suppress annoying Scikit-Learn warnings about feature names
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. LOAD THE TRAINED AI BRAIN ---
model_path = 'advanced_brain.pkl'
print(f"Loading AI Brain from '{model_path}'...")
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Brain loaded successfully! Starting camera...")
except FileNotFoundError:
    print(f"Error: Could not find {model_path}. Did you run the training script?")
    exit()

# --- 2. INITIALIZE MEDIAPIPE ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # Flip and process the frame
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw Landmarks so you can see what the AI sees
        if results.face_landmarks:
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                     mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1))
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # --- 3. REAL-TIME PREDICTION LOGIC ---
        try:
            # Extract coordinates EXACTLY how we did during training
            pose = list(np.array([[p.x, p.y, p.z, p.visibility] for p in results.pose_landmarks.landmark]).flatten()) if results.pose_landmarks else list(np.zeros(33*4))
            face = list(np.array([[p.x, p.y, p.z, p.visibility] for p in results.face_landmarks.landmark]).flatten()) if results.face_landmarks else list(np.zeros(468*4))
            lh = list(np.array([[p.x, p.y, p.z, p.visibility] for p in results.left_hand_landmarks.landmark]).flatten()) if results.left_hand_landmarks else list(np.zeros(21*4))
            rh = list(np.array([[p.x, p.y, p.z, p.visibility] for p in results.right_hand_landmarks.landmark]).flatten()) if results.right_hand_landmarks else list(np.zeros(21*4))

            # Combine into a single row
            row = pose + face + lh + rh
            
            # Convert to a DataFrame so Scikit-Learn doesn't complain about missing column names
            X = pd.DataFrame([row])

            # PREDICT!
            prediction = model.predict(X)[0] # Returns something like ['Absolute Cinema', 'smiling']
            probabilities = model.predict_proba(X) # Returns the confidence percentages
            
            # Extract individual predictions
            predicted_gesture = prediction[0]
            predicted_expression = prediction[1]
            
            # Extract confidence (predict_proba returns a list of arrays for multi-output)
            gesture_confidence = round(np.max(probabilities[0]) * 100, 1)
            expression_confidence = round(np.max(probabilities[1]) * 100, 1)

            # --- 4. DISPLAY THE RESULTS ON SCREEN ---
            # Create a nice dark background box at the top
            cv2.rectangle(image, (0, 0), (640, 80), (30, 30, 30), -1)
            
            # Display Gesture
            cv2.putText(image, f"GESTURE: {predicted_gesture.upper()} ({gesture_confidence}%)", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Display Expression
            cv2.putText(image, f"FACE:    {predicted_expression.upper()} ({expression_confidence}%)", 
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        except Exception as e:
            # If no one is in the frame, do nothing
            pass

        cv2.imshow('Live AI Action Recognition', image)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()