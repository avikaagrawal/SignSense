import pandas as pd
import numpy as np

INPUT_PATH = "data/normalized_landmarks.csv"
OUTPUT_PATH = "data/augmented_landmarks.csv"

df = pd.read_csv(INPUT_PATH)

original_rows = []
mirrored_rows = []

for _, row in df.iterrows():

    label = row["label"]

    # 63 features -> 21 x 3
    landmarks = (
        row.drop("label")
        .values
        .astype(float)
        .reshape(21, 3)
    )

    # Keep original
    original_rows.append(
        [label] + landmarks.flatten().tolist()
    )

    # Create mirrored version
    mirrored = landmarks.copy()

    # Flip X axis
    mirrored[:, 0] *= -1

    mirrored_rows.append(
        [label] + mirrored.flatten().tolist()
    )


# Column names
columns = ["label"]

for i in range(21):
    columns.extend([
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ])

augmented_df = pd.DataFrame(
    original_rows + mirrored_rows,
    columns=columns
)

augmented_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("Hand augmentation complete!")
print("Original samples:", len(original_rows))
print("Mirrored samples:", len(mirrored_rows))
print("Total samples:", len(augmented_df))
print("Saved to:", OUTPUT_PATH)