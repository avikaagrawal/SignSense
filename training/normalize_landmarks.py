import pandas as pd
import numpy as np

INPUT_PATH = "data/landmarks.csv"
OUTPUT_PATH = "data/normalized_landmarks.csv"

df = pd.read_csv(INPUT_PATH)

normalized_rows = []

for _, row in df.iterrows():

    label = row["label"]

    # Skip "nothing"
    if label == "nothing":
        continue

    # Convert 63 values → 21 x 3 landmarks
    values = row.drop("label").values.astype(float)
    landmarks = values.reshape(21, 3)

    # -------------------------
    # 1. Move wrist to origin
    # -------------------------
    wrist = landmarks[0].copy()

    landmarks = landmarks - wrist

    # -------------------------
    # 2. Normalize hand size
    # -------------------------
    distances = np.linalg.norm(landmarks, axis=1)
    max_distance = np.max(distances)

    if max_distance == 0:
        continue

    landmarks = landmarks / max_distance

    # Flatten back to 63 features
    normalized_features = landmarks.flatten()

    normalized_rows.append(
        [label] + normalized_features.tolist()
    )


# Create column names
columns = ["label"]

for i in range(21):
    columns.extend([
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ])

normalized_df = pd.DataFrame(
    normalized_rows,
    columns=columns
)

normalized_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("Normalization complete!")
print("Original rows:", len(df))
print("Normalized rows:", len(normalized_df))
print("Saved to:", OUTPUT_PATH)
