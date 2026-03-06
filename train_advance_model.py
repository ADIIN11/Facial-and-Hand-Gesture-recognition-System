import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# --- 1. LOAD THE DATA ---
csv_file = 'multi_target_data.csv'
print(f"Loading data from '{csv_file}'...")

# low_memory=False handles the 'mixed types' warning you saw
df = pd.read_csv(csv_file, low_memory=False)

# --- 2. DATA CLEANING (The Fix) ---
print(f"Original frames: {len(df)}")

# Remove any rows that have missing values (NaN)
df.dropna(inplace=True)
print(f"Frames after removing NaNs: {len(df)}")

# Fix Typos and Case Sensitivity
# This merges 'Netral', 'neutral', and 'Neutral' into one single category
df['expression_class'] = df['expression_class'].str.lower().str.strip()
df['expression_class'] = df['expression_class'].replace({'netral': 'neutral'})

df['gesture_class'] = df['gesture_class'].str.lower().str.strip()

print(f"Cleaned Gestures: {df['gesture_class'].unique()}")
print(f"Cleaned Expressions: {df['expression_class'].unique()}")

# --- 3. SEPARATE FEATURES AND LABELS ---
X = df.drop(['gesture_class', 'expression_class'], axis=1) 
y = df[['gesture_class', 'expression_class']]

# --- 4. SPLIT AND TRAIN ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining the AI... this may take a moment for 12,000 frames...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 uses all CPU cores
model.fit(X_train, y_train)

# --- 5. EVALUATE ---
y_pred = model.predict(X_test)
gesture_acc = accuracy_score(y_test['gesture_class'], y_pred[:, 0])
expression_acc = accuracy_score(y_test['expression_class'], y_pred[:, 1])

print(f"\n========================================")
print(f"  GESTURE Accuracy:    {gesture_acc * 100:.2f}%")
print(f"  EXPRESSION Accuracy: {expression_acc * 100:.2f}%")
print(f"========================================")

# --- 6. SAVE ---
with open('advanced_brain.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Success! 'advanced_brain.pkl' is ready for live use.")