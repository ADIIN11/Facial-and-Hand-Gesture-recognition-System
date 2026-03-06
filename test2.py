import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_dist(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # Convert to RGB for MediaPipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # --- 1. DRAWING SKELETONS ---
    # Face Mesh
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1))
    # Hands
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    # Pose
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    current_action = "None"
    expression = "Neutral"

    # --- 2. GESTURE LOGIC ---
    if results.left_hand_landmarks and results.right_hand_landmarks:
        # Check for Joining Hands (Namaste)
        dist_palms = calculate_dist(results.left_hand_landmarks.landmark[0], 
                                    results.right_hand_landmarks.landmark[0])
        if dist_palms < 0.1:
            current_action = "Joining Hands"
            
        # Check for Heart (Simplified: Thumb tips and Index tips touching)
        dist_thumbs = calculate_dist(results.left_hand_landmarks.landmark[4], results.right_hand_landmarks.landmark[4])
        dist_indices = calculate_dist(results.left_hand_landmarks.landmark[8], results.right_hand_landmarks.landmark[8])
        if dist_thumbs < 0.05 and dist_indices < 0.05:
            current_action = "Heart Shape"

    elif results.right_hand_landmarks:
        # Check for Pointing (Index up, others down)
        index_tip = results.right_hand_landmarks.landmark[8].y
        middle_tip = results.right_hand_landmarks.landmark[12].y
        if index_tip < middle_tip - 0.1:
            current_action = "Pointing"

    # --- 3. FACE EXPRESSION LOGIC ---
    if results.face_landmarks:
        # Simple mouth ratio for Smiling/Surprise
        top_lip = results.face_landmarks.landmark[13]
        bottom_lip = results.face_landmarks.landmark[14]
        mouth_open = calculate_dist(top_lip, bottom_lip)
        
        if mouth_open > 0.05:
            expression = "Surprised/Open"
        elif mouth_open > 0.02:
            expression = "Smiling"

    # --- 4. DISPLAY OVERLAY ---
    # Corner Expression Box
    cv2.rectangle(image, (0,0), (250, 80), (245, 117, 16), -1)
    cv2.putText(image, f"FACE: {expression}", (10, 30), 1, 2, (255, 255, 255), 2)
    cv2.putText(image, f"ACT: {current_action}", (10, 65), 1, 2, (255, 255, 255), 2)

    cv2.imshow('Action & Expression Recognition', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()