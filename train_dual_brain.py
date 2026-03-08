import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# AI RETRAINING: GESTURE AND EXPRESSION
# ==============================================================================
csv_file = 'multi_target_data.csv'
print("========================================")
print(" AI RETRAINING & ANALYTICS ")
print("========================================")
print(f"Loading data from '{csv_file}'...")

try:
    df = pd.read_csv(csv_file, low_memory=False)
    original_len = len(df)
    df.dropna(inplace=True)

    # Clean typos and standardization
    # Using .astype(str) to prevent errors if numbers accidentally slip into the labels
    df['expression_class'] = df['expression_class'].astype(str).str.lower().str.strip()
    df['expression_class'] = df['expression_class'].replace({'netral': 'neutral'})
    df['gesture_class'] = df['gesture_class'].astype(str).str.lower().str.strip()

    print(f"Total clean frames ready for training: {len(df)} (Removed {original_len - len(df)} corrupted frames)")

    # --- DATASET DISTRIBUTION ANALYTICS ---
    print("\n--- GESTURE DATASET BALANCE ---")
    print(df['gesture_class'].value_counts())

    print("\n--- EXPRESSION DATASET BALANCE ---")
    print(df['expression_class'].value_counts())
    print("----------------------------------------\n")

    # --- THE DECOUPLING LOGIC ---
    print("Slicing the coordinates into Face vs. Body/Hands...")
    
    # MediaPipe coordinate mapping
    face_cols = [f'{axis}{i}' for i in range(34, 502) for axis in ('x', 'y', 'z', 'v')]
    gesture_cols = [f'{axis}{i}' for i in range(1, 34) for axis in ('x', 'y', 'z', 'v')] + \
                   [f'{axis}{i}' for i in range(502, 544) for axis in ('x', 'y', 'z', 'v')]

    X_gesture = df[gesture_cols]
    y_gesture = df['gesture_class']

    X_face = df[face_cols]
    y_face = df['expression_class']

    # --- TRAIN BRAIN 1: GESTURES ---
    print("Training GESTURE Brain (Blind to Face)...")
    X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X_gesture, y_gesture, test_size=0.2, random_state=42)
    model_gesture = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model_gesture.fit(X_train_g, y_train_g)

    gesture_acc = accuracy_score(y_test_g, model_gesture.predict(X_test_g))
    print(f"-> Gesture Model Accuracy: {gesture_acc * 100:.2f}%")

    with open('gesture_brain.pkl', 'wb') as f:
        pickle.dump(model_gesture, f)

    # --- TRAIN BRAIN 2: EXPRESSIONS ---
    print("\nTraining EXPRESSION Brain (Blind to Hands)...")
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_face, y_face, test_size=0.2, random_state=42)
    model_expression = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model_expression.fit(X_train_f, y_train_f)

    expr_acc = accuracy_score(y_test_f, model_expression.predict(X_test_f))
    print(f"-> Expression Model Accuracy: {expr_acc * 100:.2f}%")

    with open('expression_brain.pkl', 'wb') as f:
        pickle.dump(model_expression, f)

    print("\n========================================")
    print("SUCCESS: Dual AI Engine Upgrades Complete!")
    print("========================================")

except FileNotFoundError:
    print(f"\nError: Could not find '{csv_file}'. Make sure you run your data collection first!")
except KeyError as e:
    print(f"\nError: Missing expected columns in your CSV. Are you sure 'gesture_class' and 'expression_class' exist? Details: {e}")