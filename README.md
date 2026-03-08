# 🚀 Tri-Core RSAR System
**Real-Time Spatial Action Recognition**

The **Tri-Core RSAR System** is a high-performance, parallel AI engine designed for multi-target facial recognition, expression detection, and gesture analysis. 

Built with a **"Human-in-the-Loop"** philosophy, this system allows you to act as the supervisor. You can dynamically reward or penalize the AI in real-time, instantly correcting its predictions and saving high-quality data to continuously upgrade the underlying models.

---

## ✨ Key Features
* **Multi-Target Crowd Tracking:** Tracks up to 5 people simultaneously with persistent IDs and dedicated bounding boxes.
* **Zero-Shot Identity Learning:** Uses **FaceNet** (Deep Learning) to extract 128D mathematical "face prints." Register new faces instantly with the `[u]` key without retraining.
* **Modular Tabular AI:** Leverages **Google MediaPipe** to extract 543 skeletal/facial landmarks. These are processed by fast **Random Forest Classifiers** for real-time Gesture and Expression tracking.
* **Dynamic Analytics Panel:** Features an interactive side panel providing a live roster of every person in frame, including their expression probabilities and current gestures.

---

## 🛠️ Prerequisites & Installation

> **⚠️ IMPORTANT:** This project requires specific versions of TensorFlow and OpenCV to remain compatible with MediaPipe and NumPy. Please follow these steps exactly.

### 1. Set up a Virtual Environment
It is highly recommended to use an isolated environment.
```bash
python -m venv myenv

# Activate on Windows:
myenv\Scripts\activate
# Activate on Mac/Linux:
source myenv/bin/activate
2. Install Dependencies
Create a requirements.txt file in your directory and paste the following:

Plaintext
pandas
scikit-learn
scipy
mediapipe
opencv-python<4.10
opencv-contrib-python<4.10
tensorflow==2.15.0
keras-facenet
Then, run:

Bash
pip install -r requirements.txt
🚀 The 3-Step Workflow
To get the system running, follow these steps in order:

Step 1: Record Data
Bash
python advanced_collector.py
Map your skeletal coordinates to specific labels. This generates your multi_target_data.csv.

Step 2: Train the Brains
Bash
python train_dual_brain.py
This script cleans the data and trains your Random Forest models, outputting gesture_brain.pkl and expression_brain.pkl.

Step 3: Launch Live Engine
Bash
python live_ultimate_tri_core.py
This starts the production feed with the UI overlays and analytics panel.

🎮 Live Engine Controls
General UI:

[t] - Toggle Skeletons (Landmarks)

[b] - Toggle Face ID Bounding Boxes

[i] - Toggle Side Analytics Info Panel

[q] - Quit

Active Learning Loop:

[u] - Register Face: Type the name of the "Primary Target" (Green Box) in the terminal to save them to the database.

[L-Click] or [y] - Reward: Saves 20 frames of the current gesture/expression to the CSV to reinforce correct behavior.

[w] - Penalize: Pauses the feed. Correct the AI's mistake in the terminal to save 50 frames of "corrected" data for the next training session.

📁 Repository Structure
advanced_collector.py - Coordinate mapping tool.

train_dual_brain.py - ML training script.

live_ultimate_tri_core.py - Main inference engine.

multi_target_data.csv - The training dataset.

*_brain.pkl - Trained Random Forest models.

facenet_database.pkl - Persistent database for recognized identities.

Tri-Core RSAR System © 2026




Would you like me to also provide the `requirements.txt` as a separate, downloadable block, or are you a