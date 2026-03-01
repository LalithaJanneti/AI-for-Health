

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from models.cnn_model import CNN1D



def upsample_spo2(spo2_window):
    original_len = len(spo2_window)
    target_len = 960

    return np.interp(
        np.linspace(0, original_len - 1, target_len),
        np.arange(original_len),
        spo2_window
    )



def normalize(signal):
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)



def prepare_data(df):

    X = []
    y = []

    for _, row in df.iterrows():

        airflow = normalize(np.array(row["airflow"]))
        thoracic = normalize(np.array(row["thoracic"]))
        spo2 = normalize(upsample_spo2(np.array(row["spo2"])))

        combined = np.stack([airflow, thoracic, spo2], axis=0)

        X.append(combined)
        y.append(row["label"])

    return np.array(X), np.array(y)



def train_model(dataset_path):

    df = pd.read_pickle(dataset_path)

    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["label"])

    participants = df["participant"].unique()

    for test_participant in participants:

        print("\n" + "=" * 50)
        print(f"Testing on participant: {test_participant}")
        print("=" * 50)

        train_df = df[df["participant"] != test_participant]
        test_df = df[df["participant"] == test_participant]

        X_train, y_train = prepare_data(train_df)
        X_test, y_test = prepare_data(test_df)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)

        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

        model = CNN1D(
            input_channels=3,
            num_classes=len(label_encoder.classes_)
        )

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        epochs = 10

        for epoch in range(epochs):

            model.train()

            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)

            loss.backward()
            optimizer.step()

            if epoch % 2 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")

        
        model.eval()

        with torch.no_grad():
            predictions = model(X_test).argmax(dim=1)

        y_test_np = y_test.cpu().numpy()
        pred_np = predictions.cpu().numpy()

        print("\nResults:")
        print("Accuracy :", accuracy_score(y_test_np, pred_np))
        print("Precision:", precision_score(y_test_np, pred_np, average="macro"))
        print("Recall   :", recall_score(y_test_np, pred_np, average="macro"))
        print("Confusion Matrix:\n", confusion_matrix(y_test_np, pred_np))



if __name__ == "__main__":
    train_model("Dataset/breathing_dataset.pkl")