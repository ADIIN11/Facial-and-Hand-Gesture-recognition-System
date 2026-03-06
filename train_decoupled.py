import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

csv_file = 'multi_target_data.csv'
print("Loading and cleaning data...")
df = pd.read_csv(csv_file, low_memory=False)
df.dropna(inplace=True)

# Clean typos
df['expression_class'] = df['expression_class'].str.lower().str.strip()
df['expression_class'] = df['expression_class'].replace({'netral': 'neutral'})
df['gesture_class'] = df['gesture_class'].str.lower().str.strip()

# --- 1. THE DECOUPLING LOGIC ---
print("Slicing the coordinates into Face vs. Body/Hands...")

# Face Landmarks are points 34 through 501 in the MediaPipe array
face_cols = []
for i in range(34, 502):
    face_cols.extend([f'x{i}', f'y{i}', f'z{i}', f'v{i}'])

# Pose/Body is points 1-33. Hands are points 502-543.
gesture_cols = []
for i in range(1, 34):   # Pose
    gesture_cols.extend([f'x{i}', f'y{i}', f'z{i}', f'v{i}'])
for i in range(502, 544): # Left & Right Hands
    gesture_cols.extend([f'x{i}', f'y{i}', f'z{i}', f'v{i}'])

# --- 2. PREPARE THE TWO DATASETS ---
X_gesture = df[gesture_cols]
y_gesture = df['gesture_class']

X_face = df[face_cols]
y_face = df['expression_class']

# --- 3. TRAIN BRAIN 1: GESTURES ---
print("\nTraining GESTURE Brain (Blind to Face)...")
X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X_gesture, y_gesture, test_size=0.2, random_state=42)
model_gesture = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model_gesture.fit(X_train_g, y_train_g)
gesture_acc = accuracy_score(y_test_g, model_gesture.predict(X_test_g))
print(f"Gesture Model Accuracy: {gesture_acc * 100:.2f}%")

with open('gesture_brain.pkl', 'wb') as f:
    pickle.dump(model_gesture, f)

# --- 4. TRAIN BRAIN 2: EXPRESSIONS ---
print("\nTraining EXPRESSION Brain (Blind to Hands)...")
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_face, y_face, test_size=0.2, random_state=42)
model_expression = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model_expression.fit(X_train_f, y_train_f)
expr_acc = accuracy_score(y_test_f, model_expression.predict(X_test_f))
print(f"Expression Model Accuracy: {expr_acc * 100:.2f}%")

with open('expression_brain.pkl', 'wb') as f:
    pickle.dump(model_expression, f)

print("\nSuccess! Both independent brains have been saved.")