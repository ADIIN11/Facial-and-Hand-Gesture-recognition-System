<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tri-Core RSAR System README</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #24292e;
            max-width: 900px;
            margin: 40px auto;
            padding: 0 20px;
            background-color: #ffffff;
        }
        h1 { border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; color: #0366d6; }
        h2 { border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; margin-top: 24px; }
        h3 { margin-top: 20px; }
        code {
            background-color: rgba(27, 31, 35, 0.05);
            padding: 0.2em 0.4em;
            border-radius: 3px;
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
            font-size: 85%;
        }
        pre {
            background-color: #f6f8fa;
            padding: 16px;
            border-radius: 6px;
            overflow: auto;
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
            font-size: 85%;
            line-height: 1.45;
        }
        ul { padding-left: 2em; }
        li { margin-bottom: 8px; }
        .warning {
            background-color: #fffbdd;
            border: 1px solid #d4a017;
            padding: 15px;
            border-radius: 6px;
            margin: 20px 0;
        }
        .footer {
            margin-top: 40px;
            font-size: 0.9em;
            color: #6a737d;
            text-align: center;
            border-top: 1px solid #eaecef;
            padding-top: 20px;
        }
    </style>
</head>
<body>

    <h1>🚀 Tri-Core RSAR System</h1>
    <p><strong>Real-Time Spatial Action Recognition</strong></p>

    <p>Welcome to the <strong>Tri-Core RSAR System</strong>, a custom-built, parallel AI engine designed for multi-target facial recognition, expression detection, and gesture analysis.</p>

    <p>Unlike standard static AI models, this system features a <strong>"Human-in-the-Loop"</strong> continuous learning architecture. You can dynamically reward or penalize the AI in real-time, instantly correcting its predictions and saving high-quality data to continuously upgrade its accuracy.</p>

    <h2>✨ Key Features</h2>
    <ul>
        <li><strong>Multi-Target Crowd Tracking:</strong> Simultaneously tracks up to 5 people in the camera frame, assigning them persistent IDs and individual bounding boxes.</li>
        <li><strong>Zero-Shot Identity Learning:</strong> Uses <strong>FaceNet</strong> (Deep Learning) to extract 128D mathematical "face prints." Press a single button to teach the AI a new face instantly—no model retraining required.</li>
        <li><strong>Modular Tabular AI:</strong> Uses <strong>Google MediaPipe</strong> to extract 543 skeletal/facial coordinates, feeding them into lightning-fast <strong>Random Forest Classifiers</strong> to predict Body Gestures and Facial Expressions with minimal CPU load.</li>
        <li><strong>Dynamic UI Panel:</strong> Features an interactive, expandable side panel detailing the Live Roster, Expression probabilities, and Gestures of everyone in the room.</li>
    </ul>

    <hr>

    <h2>🛠️ Prerequisites & Installation</h2>

    <div class="warning">
        <strong>⚠️ IMPORTANT:</strong> This project relies on highly specific versioning. Newer versions of TensorFlow and OpenCV will clash with MediaPipe over <code>protobuf</code> and <code>numpy 2.0</code> architecture. Please use the exact requirements file below to guarantee a stable environment.
    </div>

    <h3>1. Set up a Virtual Environment (Highly Recommended)</h3>
    <p>Ensure you are running in an isolated environment so this doesn't overwrite your global Python packages.</p>
    <pre><code>python -m venv myenv
myenv\Scripts\activate  # On Windows
# source myenv/bin/activate  # On Mac/Linux</code></pre>

    <h3>2. Create the <code>requirements.txt</code> File</h3>
    <p>Create a new file in your project folder named <code>requirements.txt</code> and paste the following code exactly as is:</p>
    <pre><code>pandas
scikit-learn
scipy
mediapipe
opencv-python&lt;4.10
opencv-contrib-python&lt;4.10
tensorflow==2.15.0
keras-facenet</code></pre>

    <h3>3. Install the Packages</h3>
    <p>With your virtual environment activated, run this single command to install the perfectly balanced tech stack:</p>
    <pre><code>pip install -r requirements.txt</code></pre>

    <hr>

    <h2>🚀 Quick Start Guide (The 3-Step Workflow)</h2>
    <p>To get the engine running from scratch, follow this pipeline based on the repository structure:</p>

    <h3>Step 1: Collect Training Data</h3>
    <pre><code>python advanced_collector.py</code></pre>
    <ul>
        <li>Follow the on-screen prompts to record various gestures and facial expressions.</li>
        <li>This script will map your skeletal coordinates and generate the <code>multi_target_data.csv</code> file.</li>
    </ul>

    <h3>Step 2: Train the Dual Brains</h3>
    <pre><code>python train_dual_brain.py</code></pre>
    <ul>
        <li>This script reads your newly created CSV file, cleans the data, and trains two separate Random Forest algorithms.</li>
        <li>It will output your two core models: <code>gesture_brain.pkl</code> and <code>expression_brain.pkl</code>.</li>
    </ul>

    <h3>Step 3: Launch the Tri-Core Live Engine</h3>
    <pre><code>python live_ultimate_tri_core.py</code></pre>
    <ul>
        <li>This launches the live webcam feed, overlaying skeletons, FaceNet bounding boxes, and the live analytics panel.</li>
    </ul>

    <hr>

    <h2>🎮 Live Engine Controls</h2>
    <p>Click on the video window to ensure it has focus, then use these hotkeys to interact with the system:</p>

    <p><strong>General UI Controls:</strong></p>
    <ul>
        <li><code>[t]</code> - Toggle MediaPipe Skeletons On/Off</li>
        <li><code>[b]</code> - Toggle Face ID Bounding Boxes On/Off</li>
        <li><code>[i]</code> - Toggle the Side Analytics Info Panel</li>
        <li><code>[q]</code> - Quit Application</li>
    </ul>

    <p><strong>AI Feedback & Learning Loop:</strong></p>
    <ul>
        <li><code>[u]</code> - <strong>FaceNet One-Shot Learning:</strong> Registers the "Primary Target" (the person with the green box). Type their name in the terminal, and the AI will remember them forever.</li>
        <li><code>[L-Click]</code> or <code>[y]</code> - <strong>Reward:</strong> Validates the AI's current Gesture/Expression guess and saves 20 frames of heavily weighted coordinate data to the CSV.</li>
        <li><code>[w]</code> - <strong>Penalize:</strong> Pauses the feed. Tells the AI it made a mistake. Enter the <em>correct</em> gesture/expression in the terminal, and it will save 50 frames of corrected data to the CSV for the next training cycle.</li>
    </ul>

    <hr>

    <h2>📁 Repository Structure Overview</h2>
    <ul>
        <li><code>advanced_collector.py</code> - Script for mapping MediaPipe coordinates to a dataset.</li>
        <li><code>train_dual_brain.py</code> - Scikit-learn training script for Gestures & Expressions.</li>
        <li><code>live_ultimate_tri_core.py</code> - The main production inference script combining all models.</li>
        <li><code>multi_target_data.csv</code> - The tabular dataset generated by the collector.</li>
        <li><code>*_brain.pkl</code> - Your frozen, trained Machine Learning models.</li>
        <li><code>facenet_database.pkl</code> - Your dynamic dictionary containing 128D FaceNet embeddings.</li>
    </ul>

    <div class="footer">
        Tri-Core RSAR System &copy; 2026
    </div>

</body>
</html>