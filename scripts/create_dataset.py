

import os
import argparse
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt



def bandpass_filter(signal, fs, low=0.17, high=0.4, order=4):
    nyquist = 0.5 * fs
    low /= nyquist
    high /= nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)



def create_windows(signal, window_size, step_size):
    windows = []
    for start in range(0, len(signal) - window_size, step_size):
        windows.append(signal[start:start + window_size])
    return windows



def load_signal(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            numeric_parts = []
            for p in parts:
                try:
                    numeric_parts.append(float(p))
                except:
                    continue

            if len(numeric_parts) > 0:
                data.append(numeric_parts)

    arr = np.array(data)
    return arr.squeeze()   

def load_events(file_path):
    events = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            numeric_parts = []
            for p in parts:
                try:
                    numeric_parts.append(float(p))
                except:
                    continue

            if len(numeric_parts) >= 2:
                start = numeric_parts[0]
                end = numeric_parts[1]
                event_type = parts[-1]

                events.append((start, end, event_type))

    return events

def process_participant(participant_path, participant_name):


    airflow = load_signal(os.path.join(participant_path, "flow.txt"))
    thoracic = load_signal(os.path.join(participant_path, "Thorac.txt"))
    spo2 = load_signal(os.path.join(participant_path, "SPO2.txt"))

    
    events = load_events(os.path.join(participant_path, "flowEvents.txt"))


    airflow = bandpass_filter(airflow, fs=32)
    thoracic = bandpass_filter(thoracic, fs=32)

    
    window_size_32 = 32 * 30     
    step_size_32 = 32 * 15         

    window_size_4 = 4 * 30      
    step_size_4 = 4 * 15

    airflow_windows = create_windows(airflow, window_size_32, step_size_32)
    thoracic_windows = create_windows(thoracic, window_size_32, step_size_32)
    spo2_windows = create_windows(spo2, window_size_4, step_size_4)

    dataset = []

    for i in range(len(airflow_windows)):
        window_start=i*(step_size_32 / 32)
        window_end=window_start+30
        label="Normal"

        for event_start,event_end,event_type in events:
            overlap=max(
                0,
                min(window_end,event_end)-max(window_start,event_start)
            )
            if overlap>15:
                label=event_type
                break
        dataset.append(
            {
                "participant":participant_name,
                "airflow": airflow_windows[i],
                "thoracic": thoracic_windows[i],
                "spo2": spo2_windows[i],
                "label": label
            }
        )
    return dataset



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-in_dir", required=True)
    parser.add_argument("-out_dir", required=True)
    args = parser.parse_args()

    all_data = []

    for participant in os.listdir(args.in_dir):
        path = os.path.join(args.in_dir, participant)

        if os.path.isdir(path):
            print(f"Processing {participant}...")
            all_data.extend(process_participant(path, participant))

    df = pd.DataFrame(all_data)

    os.makedirs(args.out_dir, exist_ok=True)
    df.to_pickle(os.path.join(args.out_dir, "breathing_dataset.pkl"))

    print("Dataset saved successfully.")