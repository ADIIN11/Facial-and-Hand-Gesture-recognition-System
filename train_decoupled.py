import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# ==============================================================================
# PART 1: GESTURE AND EXPRESSION TRAINING
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
    df['expression_class'] = df['expression_class'].str.lower().str.strip()
    df['expression_class'] = df['expression_class'].replace({'netral': 'neutral'})
    df['gesture_class'] = df['gesture_class'].str.lower().str.strip()

    print(f"Total clean frames ready for training: {len(df)} (Removed {original_len - len(df)} corrupted frames)")

    # --- DATASET DISTRIBUTION ANALYTICS ---
    print("\n--- GESTURE DATASET BALANCE ---")
    print(df['gesture_class'].value_counts())

    print("\n--- EXPRESSION DATASET BALANCE ---")
    print(df['expression_class'].value_counts())
    print("----------------------------------------\n")

    # --- THE DECOUPLING LOGIC ---
    print("Slicing the coordinates into Face vs. Body/Hands...")
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

except FileNotFoundError:
    print(f"Warning: Could not find '{csv_file}'. Skipping Gesture/Expression training.")

# ==============================================================================
# PART 2: IDENTITY TRAINING
# ==============================================================================
id_csv = 'identity_data.csv'
print("\n========================================")
print(" TRAINING BRAIN 3: IDENTITY ")
print("========================================")

try:
    print(f"Loading data from '{id_csv}'...")
    df_id = pd.read_csv(id_csv, low_memory=False)
    original_id_len = len(df_id)
    df_id.dropna(inplace=True)
    
    # Standardize names
    df_id['person_name'] = df_id['person_name'].str.lower().str.strip()
    
    print(f"Total clean identity frames: {len(df_id)} (Removed {original_id_len - len(df_id)} corrupted frames)")
    
    print("\n--- IDENTITY DATASET BALANCE ---")
    print(df_id['person_name'].value_counts())
    print("----------------------------------------\n")

    # The Identity collector only saved the 468 face landmarks
    id_cols = [f'{axis}{i}' for i in range(1, 469) for axis in ('x', 'y', 'z', 'v')]
    
    X_id = df_id[id_cols]
    y_id = df_id['person_name']

    print("Training IDENTITY Brain (Facial Biometrics)...")
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X_id, y_id, test_size=0.2, random_state=42)
    model_identity = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model_identity.fit(X_train_i, y_train_i)

    id_acc = accuracy_score(y_test_i, model_identity.predict(X_test_i))
    print(f"-> Identity Model Accuracy: {id_acc * 100:.2f}%")

    with open('identity_brain.pkl', 'wb') as f:
        pickle.dump(model_identity, f)

except FileNotFoundError:
    print(f"Warning: '{id_csv}' not found. Skipping Identity Brain training.")
    print("If you want facial recognition, run 'identity_collector.py' first!")

print("\n========================================")
print("SUCCESS: AI Engine Upgrades Complete!")
print("========================================")