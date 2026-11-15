import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data/landmarks.csv")

letters = ["T", "V", "Z", "R"]

fig, axes = plt.subplots(4, 5, figsize=(15, 12))

for row, letter in enumerate(letters):

    samples = df[df["label"] == letter].sample(
        5, random_state=42
    )

    for col, (_, sample) in enumerate(samples.iterrows()):

        landmarks = sample.drop("label").values.astype(float)
        landmarks = landmarks.reshape(21, 3)

        axes[row, col].scatter(
            landmarks[:, 0],
            -landmarks[:, 1]
        )

        for i, (x, y, z) in enumerate(landmarks):
            axes[row, col].text(x, -y, str(i), fontsize=7)

        axes[row, col].set_title(letter)
        axes[row, col].axis("equal")
        axes[row, col].axis("off")

plt.tight_layout()
plt.savefig("data/gesture_comparison.png")
plt.show()