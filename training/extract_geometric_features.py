import os
import numpy as np
import pandas as pd

# ==========================================
# PATHS
# ==========================================

INPUT_PATH = "data/normalized_landmarks.csv"
OUTPUT_PATH = "data/geometric_features.csv"

# ==========================================
# MEDIAPIPE LANDMARK INDICES
# ==========================================

FINGER_LANDMARKS = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20]
}

FINGERTIPS = [4, 8, 12, 16, 20]


# ==========================================
# FEATURE EXTRACTION
# ==========================================

def calculate_angle(a, b, c):
    """
    Calculate angle ABC in degrees.
    """

    ba = a - b
    bc = c - b

    denominator = (
        np.linalg.norm(ba) *
        np.linalg.norm(bc)
    )

    if denominator < 1e-8:
        return 0.0

    cosine = np.dot(ba, bc) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(np.arccos(cosine))


def extract_geometric_features(landmarks):

    features = []

    # --------------------------------------
    # 1. Finger joint angles (15 features)
    # --------------------------------------

    for finger, indices in FINGER_LANDMARKS.items():

        if finger == "thumb":
            angle_triplets = [
                (0, 1, 2),
                (1, 2, 3),
                (2, 3, 4)
            ]

        else:
            angle_triplets = [
                (0, indices[0], indices[1]),
                (indices[0], indices[1], indices[2]),
                (indices[1], indices[2], indices[3])
            ]

        for a, b, c in angle_triplets:

            angle = calculate_angle(
                landmarks[a],
                landmarks[b],
                landmarks[c]
            )

            features.append(angle / 180.0)

    # --------------------------------------
    # 2. Fingertip-to-wrist distances (5)
    # --------------------------------------

    for tip in FINGERTIPS:

        distance = np.linalg.norm(
            landmarks[tip] - landmarks[0]
        )

        features.append(distance)

    # --------------------------------------
    # 3. Fingertip-to-fingertip distances (10)
    # --------------------------------------

    for i in range(len(FINGERTIPS)):

        for j in range(i + 1, len(FINGERTIPS)):

            distance = np.linalg.norm(
                landmarks[FINGERTIPS[i]] -
                landmarks[FINGERTIPS[j]]
            )

            features.append(distance)

    return np.array(features, dtype=np.float32)


# ==========================================
# MAIN
# ==========================================

def main():

    print("=" * 55)
    print("SIGNSENSE - GEOMETRIC FEATURE EXTRACTION")
    print("=" * 55)

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print("\nLoaded dataset:", df.shape)

    labels = df["label"].astype(str)

    X = df.drop(columns=["label"]).to_numpy(
        dtype=np.float32
    )

    if X.shape[1] != 63:
        raise ValueError(
            f"Expected 63 landmark features, got {X.shape[1]}"
        )

    output_rows = []

    for index, row in enumerate(X):

        landmarks = row.reshape(21, 3)

        geometric = extract_geometric_features(
            landmarks
        )

        combined = np.concatenate([
            row,
            geometric
        ])

        output_rows.append(
            [labels.iloc[index]] + combined.tolist()
        )

        if (index + 1) % 10000 == 0:
            print(f"Processed {index + 1} samples")

    # --------------------------------------
    # COLUMN NAMES
    # --------------------------------------

    columns = ["label"]

    for i in range(21):
        columns.extend([
            f"x{i}",
            f"y{i}",
            f"z{i}"
        ])

    for finger in FINGER_LANDMARKS:
        for joint in range(1, 4):
            columns.append(
                f"{finger}_angle_{joint}"
            )

    for tip in FINGERTIPS:
        columns.append(
            f"wrist_to_tip_{tip}"
        )

    for i in range(len(FINGERTIPS)):
        for j in range(i + 1, len(FINGERTIPS)):
            columns.append(
                f"tip_distance_{i}_{j}"
            )

    # --------------------------------------
    # SAVE
    # --------------------------------------

    result = pd.DataFrame(
        output_rows,
        columns=columns
    )

    os.makedirs("data", exist_ok=True)

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 55)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 55)

    print("Original features: 63")
    print("Geometric features: 30")
    print("Total features:", result.shape[1] - 1)

    print("Samples:", len(result))
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()