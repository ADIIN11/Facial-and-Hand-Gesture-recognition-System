import cv2
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from ultralytics import YOLO

# --- 1. DEFINE THE LSTM MODEL ARCHITECTURE ---
class ActionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ActionClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x shape: (batch, sequence_length, input_size)
        out, _ = self.lstm(x)
        # We only care about the output of the last time step
        out = self.fc(out[:, -1, :])
        return out

# --- 2. CONFIGURATION & INITIALIZATION ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
POSE_MODEL = YOLO('yolov8n-pose.pt') # Lightweight version for speed
SEQUENCE_LENGTH = 30  # Number of frames to analyze (approx 1 second)
INPUT_SIZE = 34       # 17 keypoints * (x, y) coordinates
HIDDEN_SIZE = 64
NUM_CLASSES = 3       # Example: 0: Walking, 1: Falling, 2: Waving
ACTION_LABELS = {0: "Walking", 1: "Falling", 2: "Waving"}

# Initialize Model (In a real scenario, you'd load .pth weights here)
model = ActionClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(DEVICE)
model.eval()

# Buffer to store sequences for one person
pose_history = deque(maxlen=SEQUENCE_LENGTH)

# --- 3. HELPER: DATA NORMALIZATION ---
def normalize_keypoints(keypoints, boxes):
    """
    Translates keypoints to 0,0 relative to the bounding box 
    and scales them between 0 and 1.
    """
    x_min, y_min, x_max, y_max = boxes[0] # Get person bounding box
    width = x_max - x_min
    height = y_max - y_min
    
    # Flatten (17, 2) to (34,) and normalize
    normalized = []
    for kp in keypoints:
        norm_x = (kp[0] - x_min) / width
        norm_y = (kp[1] - y_min) / height
        normalized.extend([norm_x, norm_y])
    return np.array(normalized, dtype=np.float32)

# --- 4. MAIN INFERENCE LOOP ---
cap = cv2.VideoCapture(0) # Open Webcam

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # Step A: Run YOLO-Pose
    results = POSE_MODEL(frame, verbose=False, device=DEVICE)
    
    for r in results:
        if r.keypoints is not None and len(r.keypoints.xy) > 0:
            # We'll track the first person detected for this test
            kps = r.keypoints.xy[0].cpu().numpy()
            bbox = r.boxes.xyxy[0].cpu().numpy()
            
            # Step B: Normalize and add to buffer
            norm_kps = normalize_keypoints(kps, [bbox])
            pose_history.append(norm_kps)
            
            # Step C: If buffer is full, classify!
            if len(pose_history) == SEQUENCE_LENGTH:
                # Convert buffer to tensor: (1, 30, 34)
                sequence_tensor = torch.tensor(list(pose_history)).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    output = model(sequence_tensor)
                    prediction = torch.argmax(output, dim=1).item()
                    action_name = ACTION_LABELS[prediction]
                    
                # Display the Action ID
                cv2.putText(frame, f"Action: {action_name}", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Security Feed - Action Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()