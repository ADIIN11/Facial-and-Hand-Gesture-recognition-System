import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Holistic (Legacy Version)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_dist(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # Flip the frame horizontally for a more natural mirror-like experience
    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # --- 1. DRAWING ---
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1))
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    current_action = "None"
    expression = "Neutral"

    # --- 2. GESTURE LOGIC (Right Hand) ---
    if results.right_hand_landmarks:
        hand = results.right_hand_landmarks.landmark
        
        # Check which fingers are "UP" (tip is higher than the lower knuckle)
        # Note: In OpenCV, lower Y values are HIGHER on the screen.
        fingers = []
        fingers.append(1 if hand[4].y < hand[3].y else 0)  # Thumb
        fingers.append(1 if hand[8].y < hand[6].y else 0)  # Index
        fingers.append(1 if hand[12].y < hand[10].y else 0) # Middle
        fingers.append(1 if hand[16].y < hand[14].y else 0) # Ring
        fingers.append(1 if hand[20].y < hand[18].y else 0) # Pinky

        # Decode the gesture based on the finger array [Thumb, Index, Middle, Ring, Pinky]
        if fingers == [0, 1, 1, 0, 0]:
            current_action = "Victory / Peace"
        elif fingers == [0, 0, 1, 0, 0]:
            current_action = "Middle Finger"
        elif fingers == [1, 0, 0, 0, 0]:
            current_action = "Thumbs Up"
        elif fingers == [0, 1, 0, 0, 0]:
            current_action = "Pointing"
        elif sum(fingers) == 5:
            current_action = "Open Hand / Stop"

    # --- 3. TWO-HANDED GESTURES ---
    if results.left_hand_landmarks and results.right_hand_landmarks:
        dist_palms = calculate_dist(results.left_hand_landmarks.landmark[0], 
                                    results.right_hand_landmarks.landmark[0])
        if dist_palms < 0.1:
            current_action = "Joining Hands"

    # --- 4. FACE EXPRESSION LOGIC ---
    if results.face_landmarks:
        face = results.face_landmarks.landmark
        
        # Landmarks: 13 (Top Lip), 14 (Bottom Lip), 61 & 291 (Mouth Corners)
        # 107 & 336 (Inner Eyebrows)
        
        mouth_open_dist = calculate_dist(face[13], face[14])
        eyebrow_dist = calculate_dist(face[107], face[336])
        
        # Frown calculation: Are the mouth corners lower than the bottom lip?
        # (Remember: higher Y value means lower on the screen)
        frown = face[61].y > face[14].y and face[291].y > face[14].y

        if eyebrow_dist < 0.035: # Eyebrows pinched together
            expression = "Anger"
        elif frown and mouth_open_dist < 0.02:
            expression = "Sadness"
        elif mouth_open_dist > 0.06:
            expression = "Laughter"
        elif mouth_open_dist > 0.02 and mouth_open_dist <= 0.06:
            expression = "Speaking"
        elif face[61].y < face[14].y and face[291].y < face[14].y: 
            # Corners higher than bottom lip
            expression = "Smiling"

    # --- 5. DISPLAY OVERLAY ---
    cv2.rectangle(image, (0,0), (400, 100), (40, 40, 40), -1)
    cv2.putText(image, f"FACE: {expression}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(image, f"ACT: {current_action}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Action & Expression Recognition', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()