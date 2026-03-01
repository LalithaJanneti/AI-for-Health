AI for Health – Breathing Irregularity Detection
👩‍💻 Project Overview

This project detects abnormal breathing patterns during sleep using physiological signals.
The goal is to identify breathing problems such as:
Apnea
Hypopnea
Normal breathing

The model is trained using overnight sleep data collected from 5 participants.
This project follows a complete pipeline:
Data understanding
Signal processing
Dataset creation
Deep learning model training
Proper evaluation

📊 Data Description

For each participant, the dataset contains:
1️⃣ Nasal Airflow (32 Hz)
      Measures air flowing through the nose.
2️⃣ Thoracic Movement (32 Hz)
      Measures chest movement while breathing.
3️⃣ SpO₂ (4 Hz)
      Measures oxygen level in blood.
4️⃣ flowEvents.txt
      Contains start time and end time of abnormal breathing events.
5️⃣ Sleep Profile.txt
      Contains sleep stage information (not used in this model).
Each recording is around 8 hours long.

🔎 Step 1 – Data Visualization

To understand the signals clearly:
  All signals are plotted over time.
  Breathing events are highlighted on the graph.
  Visualization is saved as a PDF file.

⚙️ Step 2 – Signal Preprocessing

Normal breathing happens between:
  10 to 24 breaths per minute
  Which equals 0.17 Hz to 0.4 Hz
So we applied a bandpass filter (0.17–0.4 Hz) to:
  Remove noise
  Keep only breathing-related frequency
Filtering is applied to:
  Nasal airflow
  Thoracic movement

📦 Step 3 – Windowing and Labeling

Signals are divided into:
  30-second windows
  50% overlap
Why 30 seconds?
   Because sleep studies use 30-second intervals as standard.
Labeling rule:
  If a window overlaps more than 50% with an event → assign event label
  Otherwise → label as Normal
This creates a clean training dataset.

🤖 Step 4 – Deep Learning Model

We built a 1D Convolutional Neural Network (1D CNN).
Why 1D CNN?
  Because:
     Our data is time-series.
     CNN can learn breathing patterns automatically.
     It is efficient and works well for physiological signals.
Input to the model:
  Airflow
  Thoracic movement
  SpO₂ (upsampled to match sampling rate)

🧪 Step 5 – Evaluation Strategy

We used:
  Leave-One-Participant-Out Cross Validation
This means:
  Train on 4 participants
  Test on the remaining 1
Repeat for all 5 participants

Why?
  Because this tests whether the model works on a new unseen patient.
This avoids data leakage.

Metrics Reported
For each fold we report:
  Accuracy
  Precision
  Recall
  Confusion Matrix
These metrics show how well the model detects abnormal breathing.

💡 What This Project Demonstrates

This project shows:
  Time-series signal handling
  Biomedical signal filtering
  Window-based dataset creation
  Event-based labeling
  Deep learning using 1D CNN
  Proper cross-validation strategy

  