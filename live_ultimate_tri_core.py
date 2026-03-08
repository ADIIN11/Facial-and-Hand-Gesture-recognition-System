import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import warnings
import csv
import os
import math
from collections import Counter, deque
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine

warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURATION ---
CSV_FILE = 'multi_target_data.csv'
FACE_DB_FILE = 'facenet_database.pkl'

# --- 1. LOAD AI BRAINS ---
print("Loading Gesture & Expression Brains...")
try:
    with open('gesture_brain.pkl', 'rb') as f:
        gesture_model = pickle.load(f)
    with open('expression_brain.pkl', 'rb') as f:
        expression_model = pickle.load(f)
except FileNotFoundError:
    print("Error: Could not find gesture/expression .pkl files!")
    exit()

print("Loading FaceNet Deep Learning Model (This takes a few seconds)...")
facenet_embedder = FaceNet()

if os.path.exists(FACE_DB_FILE):
    with open(FACE_DB_FILE, 'rb') as f:
        known_faces = pickle.load(f)
    print(f"Loaded {len(known_faces)} known faces from database.")
else:
    known_faces = {}

# Setup columns
face_cols = [f'{axis}{i}' for i in range(34, 502) for axis in ('x', 'y', 'z', 'v')]
gesture_cols = [f'{axis}{i}' for i in range(1, 34) for axis in ('x', 'y', 'z', 'v')] + \
               [f'{axis}{i}' for i in range(502, 544) for axis in ('x', 'y', 'z', 'v')]

# --- 2. INITIALIZE MODULAR MEDIAPIPE ---
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cam_index = 0
cap = cv2.VideoCapture(cam_index)

# State Variables
show_landmarks = True
show_face_box = True  
show_info_panel = True 

# Primary Tracking & Persistent ID Buffers
gest_buffer = deque(maxlen=5)
tracked_faces = [] 
next_person_id = 1
id_to_num_map = {} 
frame_count = 0
FACENET_SKIP_FRAMES = 10 

# Feedback Loop Variables
trigger_gest_reward = False

def mouse_click(event, x, y, flags, param):
    global trigger_gest_reward
    # Removed Right-Click. Only Left-Click triggers the reward now.
    if event == cv2.EVENT_LBUTTONDOWN: 
        trigger_gest_reward = True

cv2.namedWindow('Tri-Core AI Engine')
cv2.setMouseCallback('Tri-Core AI Engine', mouse_click)

print("\n========================================")
print(" MULTI-TARGET CROWD TRACKING SYSTEM")
print("========================================")
print("  [t] - Toggle Skeletons On/Off")
print("  [b] - Toggle Face ID Boxes On/Off")
print("  [i] - TOGGLE SIDE INFO PANEL")
print("  [u] - Add/Override Primary Face to DB")
print("  [y] / [L-Click] - Reward Primary Target (20 Frames)")
print("  [w] - Penalize Primary Target (50 Frames)")
print("  [q] - Quit")
print("========================================\n")

# Allow up to 5 faces and 4 hands
with mp_face_mesh.FaceMesh(max_num_faces=5, min_detection_confidence=0.5) as face_mesh, \
     mp_pose.Pose(min_detection_confidence=0.5) as pose_tracker, \
     mp_hands.Hands(max_num_hands=4, min_detection_confidence=0.5) as hands_tracker:

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame_count += 1
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_results = face_mesh.process(image_rgb)
        pose_results = pose_tracker.process(image_rgb)
        hands_results = hands_tracker.process(image_rgb)
        
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        h, w, _ = image_bgr.shape

        # --- KEYBOARD CONTROLS ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('t'): show_landmarks = not show_landmarks
        elif key == ord('b'): show_face_box = not show_face_box 
        elif key == ord('i'): show_info_panel = not show_info_panel 
        
        # --- POSE & HANDS EXTRACTION ---
        pose_coords = list(np.zeros(33*4))
        pose_nose_x, pose_nose_y = -1, -1
        if pose_results.pose_landmarks:
            pose_coords = list(np.array([[p.x, p.y, p.z, getattr(p, 'visibility', 0.0)] for p in pose_results.pose_landmarks.landmark]).flatten())
            pose_nose_x = int(pose_results.pose_landmarks.landmark[0].x * w)
            pose_nose_y = int(pose_results.pose_landmarks.landmark[0].y * h)
            if show_landmarks:
                mp_drawing.draw_landmarks(image_bgr, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        lh_coords = list(np.zeros(21*4))
        rh_coords = list(np.zeros(21*4))
        if hands_results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                hand_label = hands_results.multi_handedness[idx].classification[0].label
                coords = list(np.array([[p.x, p.y, p.z, getattr(p, 'visibility', 0.0)] for p in hand_landmarks.landmark]).flatten())
                if hand_label == 'Left': lh_coords = coords
                elif hand_label == 'Right': rh_coords = coords
                if show_landmarks:
                    mp_drawing.draw_landmarks(image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # --- MULTI-FACE PROCESSING ---
        current_faces = [] 
        
        # Variables to hold the Primary Target's data for the Reward/Penalize system
        primary_full_row = []
        primary_gest_label = "none"
        primary_expr_label = "neutral"
        
        if face_results.multi_face_landmarks:
            for face_idx, face_landmarks in enumerate(face_results.multi_face_landmarks):
                x_max, y_max, x_min, y_min = 0, 0, w, h
                for lm in face_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max: x_max = x
                    if x < x_min: x_min = x
                    if y > y_max: y_max = y
                    if y < y_min: y_min = y
                
                cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2 
                pad = 20
                y1, y2 = max(0, y_min - pad), min(h, y_max + pad)
                x1, x2 = max(0, x_min - pad), min(w, x_max + pad)

                is_primary = False
                if pose_nose_x != -1:
                    dist_to_pose = math.hypot(cx - pose_nose_x, cy - pose_nose_y)
                    if dist_to_pose < 100:
                        is_primary = True

                face_coords = list(np.array([[p.x, p.y, p.z, getattr(p, 'visibility', 0.0)] for p in face_landmarks.landmark]).flatten())
                X_face = pd.DataFrame([face_coords], columns=face_cols)
                raw_expr = expression_model.predict(X_face)[0]
                prob_expr = round(np.max(expression_model.predict_proba(X_face)) * 100, 1)

                raw_gest = "N/A (No Pose)"
                if is_primary:
                    gesture_row = pose_coords + lh_coords + rh_coords
                    X_gesture = pd.DataFrame([gesture_row], columns=gesture_cols)
                    raw_gest = gesture_model.predict(X_gesture)[0]
                    gest_buffer.append(raw_gest)
                    raw_gest = Counter(gest_buffer).most_common(1)[0][0]
                    
                    # Capture the primary target's full data row for the feedback loop
                    primary_full_row = pose_coords + face_coords + lh_coords + rh_coords
                    primary_gest_label = raw_gest
                    primary_expr_label = raw_expr

                matched_id = "unknown"
                matched_num = -1
                for tf in tracked_faces:
                    dist = math.hypot(cx - tf['cx'], cy - tf['cy'])
                    if dist < 80: 
                        matched_id = tf['id']
                        matched_num = tf['num']
                        break
                
                if matched_num == -1:
                    matched_num = next_person_id
                    next_person_id += 1
                
                if matched_id == "unknown" and frame_count % FACENET_SKIP_FRAMES == 0:
                    face_crop = image_rgb[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        face_crop_resized = cv2.resize(face_crop, (160, 160))
                        current_embedding = facenet_embedder.embeddings([face_crop_resized])[0]
                        
                        if key == ord('u') and is_primary: 
                            print(f"\n[FACENET LEARNING: Person {matched_num}]")
                            new_name = input("Enter name for the PRIMARY face: ").strip().lower()
                            if new_name:
                                known_faces[new_name] = current_embedding
                                with open(FACE_DB_FILE, 'wb') as f:
                                    pickle.dump(known_faces, f)
                                matched_id = new_name
                                
                                if matched_id not in id_to_num_map:
                                    id_to_num_map[matched_id] = matched_num
                                else:
                                    matched_num = id_to_num_map[matched_id]
                            print("Click back on video window to resume!")
                        else:
                            lowest_distance = 0.50 
                            for name, db_embedding in known_faces.items():
                                dist = cosine(current_embedding, db_embedding)
                                if dist < lowest_distance:
                                    lowest_distance = dist
                                    matched_id = name
                            
                            if matched_id != "unknown":
                                if matched_id in id_to_num_map:
                                    matched_num = id_to_num_map[matched_id]
                                else:
                                    id_to_num_map[matched_id] = matched_num

                current_faces.append({
                    'num': matched_num, 'id': matched_id, 
                    'cx': cx, 'cy': cy, 'box': (x1, y1, x2, y2),
                    'expr': raw_expr, 'expr_prob': prob_expr,
                    'gest': raw_gest, 'is_primary': is_primary
                })

                if show_landmarks:
                    mp_drawing.draw_landmarks(image_bgr, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, 
                                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1))

        tracked_faces = current_faces 

        # --- 6. FEEDBACK LOOP LOGIC (Rewards & Penalties) ---
        if (key == ord('y') or trigger_gest_reward) and len(primary_full_row) > 0:
            row_to_save = [primary_gest_label, primary_expr_label] + primary_full_row
            with open(CSV_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                for _ in range(20): writer.writerow(row_to_save)
            print(f"[REWARD] Saved 20 frames for Primary Target: {primary_gest_label} / {primary_expr_label}")
            trigger_gest_reward = False

        elif key == ord('w') and len(primary_full_row) > 0:
            print("\n[PENALIZE] Paused. Look at the terminal!")
            correct_gest = input(f"AI guessed Gesture '{primary_gest_label}'. Correct it (Enter for 'none'): ").strip() or "none"
            correct_expr = input(f"AI guessed Face '{primary_expr_label}'. Correct it (Enter for 'neutral'): ").strip() or "neutral"
            
            row_to_save = [correct_gest, correct_expr] + primary_full_row
            with open(CSV_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                for _ in range(50): writer.writerow(row_to_save)
                    
            print(f"[CORRECTED] Saved 50 frames for Primary Target as: {correct_gest} / {correct_expr}.")
            print("Click back on video window to resume!")

        # Reset delayed triggers
        trigger_gest_reward = False

        # --- 7. OVERLAYS & TRUE SIDE PANEL ---
        if show_face_box:
            for f in tracked_faces:
                x1, y1, x2, y2 = f['box']
                color = (0, 255, 0) if f['is_primary'] else (255, 100, 100)
                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
                
                box_title = f"P{f['num']}: {f['id'].upper()}"
                cv2.putText(image_bgr, box_title, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                box_details = f"Expr: {f['expr'].capitalize()} | Gest: {f['gest'].capitalize()}"
                cv2.putText(image_bgr, box_details, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

        panel_width = 350
        
        if show_info_panel:
            display_img = np.zeros((h, w + panel_width, 3), dtype=np.uint8)
            display_img[:, :w] = image_bgr 
            
            cv2.rectangle(display_img, (w, 0), (w + panel_width, h), (25, 25, 25), -1)
            
            cv2.putText(display_img, "TARGET DETAILS", (w + 15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.line(display_img, (w + 15, 45), (w + panel_width - 15, 45), (255, 255, 255), 1)

            y_offset = 80
            for f in sorted(tracked_faces, key=lambda x: x['num']):
                color = (0, 255, 0) if f['is_primary'] else (255, 200, 100)
                
                cv2.putText(display_img, f"Person {f['num']}: {f['id'].upper()}", (w + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                cv2.putText(display_img, f"Expr: {f['expr'].capitalize()} ({f['expr_prob']}%)", (w + 15, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                cv2.putText(display_img, f"Gest: {f['gest'].capitalize()}", (w + 15, y_offset + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                
                y_offset += 85 

            controls_text = "'b': Boxes | 'i': Info Panel | 't': Skeletons | L-Click: Reward"
            cv2.putText(display_img, controls_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow('Tri-Core AI Engine', display_img)
        else:
            controls_text = "'b': Boxes | 'i': Info Panel | 't': Skeletons | L-Click: Reward"
            cv2.putText(image_bgr, controls_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Tri-Core AI Engine', image_bgr)

cap.release()
cv2.destroyAllWindows()