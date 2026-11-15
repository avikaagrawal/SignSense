import numpy as np
import pandas as pd

from pathlib import Path


# ==========================================
# PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data/wlasl/landmarks/wlasl_landmarks.npz"

OUTPUT_DIR = ROOT / "data/wlasl/landmarks/analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QUALITY_FILE = OUTPUT_DIR / "sample_quality.csv"
DISTRIBUTION_FILE = OUTPUT_DIR / "class_distribution.csv"
SUMMARY_FILE = OUTPUT_DIR / "analysis_summary.txt"


# ==========================================
# CONFIGURATION
# ==========================================

EXPECTED_FRAMES = 30
EXPECTED_FEATURES = 128

LEFT_HAND_START = 0
RIGHT_HAND_START = 63

LEFT_PRESENCE_INDEX = 126
RIGHT_PRESENCE_INDEX = 127


# ==========================================
# LOAD DATASET
# ==========================================

def load_dataset():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    data = np.load(DATA_FILE, allow_pickle=False)

    required_keys = [
        "X",
        "y",
        "splits",
        "video_ids",
        "signer_ids"
    ]

    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing dataset field: {key}")

    X = data["X"]
    y = data["y"]
    splits = data["splits"]
    video_ids = data["video_ids"]
    signer_ids = data["signer_ids"]

    return X, y, splits, video_ids, signer_ids


# ==========================================
# DATASET STRUCTURE
# ==========================================

def inspect_structure(X, y, splits):

    print("\n" + "=" * 55)
    print("1. DATASET STRUCTURE")
    print("=" * 55)

    print("X shape:", X.shape)
    print("Labels shape:", y.shape)
    print("Splits shape:", splits.shape)

    print("Data type:", X.dtype)
    print("Expected shape:", (len(y), EXPECTED_FRAMES, EXPECTED_FEATURES))

    print("Total samples:", len(X))
    print("Unique words:", len(np.unique(y)))

    print("\nAvailable words:")
    print(sorted(np.unique(y).tolist()))

    if X.ndim != 3:
        raise ValueError("X must be a 3-dimensional array.")

    if X.shape[1:] != (EXPECTED_FRAMES, EXPECTED_FEATURES):
        raise ValueError("Unexpected sequence or feature dimensions.")

    if not (len(X) == len(y) == len(splits)):
        raise ValueError("Dataset arrays have inconsistent lengths.")


# ==========================================
# CHECK INVALID VALUES
# ==========================================

def inspect_values(X):

    print("\n" + "=" * 55)
    print("2. INVALID VALUE CHECK")
    print("=" * 55)

    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()

    print("NaN values:", nan_count)
    print("Infinite values:", inf_count)

    print("Minimum value:", np.nanmin(X))
    print("Maximum value:", np.nanmax(X))
    print("Mean value:", np.nanmean(X))
    print("Standard deviation:", np.nanstd(X))

    return nan_count, inf_count


# ==========================================
# HAND DETECTION ANALYSIS
# ==========================================

def inspect_hands(X):

    print("\n" + "=" * 55)
    print("3. HAND DETECTION ANALYSIS")
    print("=" * 55)

    left_presence = X[:, :, LEFT_PRESENCE_INDEX]
    right_presence = X[:, :, RIGHT_PRESENCE_INDEX]

    left_frames = np.sum(left_presence > 0.5, axis=1)
    right_frames = np.sum(right_presence > 0.5, axis=1)

    both_hands = np.sum(
        (left_presence > 0.5) &
        (right_presence > 0.5),
        axis=1
    )

    any_hand = np.sum(
        (left_presence > 0.5) |
        (right_presence > 0.5),
        axis=1
    )

    print("Samples with left hand:", np.sum(left_frames > 0))
    print("Samples with right hand:", np.sum(right_frames > 0))
    print("Samples with both hands:", np.sum(both_hands > 0))
    print("Samples with no detected hands:", np.sum(any_hand == 0))

    print("\nAverage detected frames per sample:")

    print("Left hand:", round(left_frames.mean(), 2))
    print("Right hand:", round(right_frames.mean(), 2))
    print("Both hands:", round(both_hands.mean(), 2))
    print("At least one hand:", round(any_hand.mean(), 2))

    return left_frames, right_frames, both_hands, any_hand


# ==========================================
# MOVEMENT ANALYSIS
# ==========================================

def inspect_movement(X):

    print("\n" + "=" * 55)
    print("4. MOVEMENT ANALYSIS")
    print("=" * 55)

    # Calculate changes between consecutive frames.
    differences = np.diff(X[:, :, :126], axis=1)

    movement = np.mean(
        np.abs(differences),
        axis=(1, 2)
    )

    print("Average movement:", round(movement.mean(), 6))
    print("Minimum movement:", round(movement.min(), 6))
    print("Maximum movement:", round(movement.max(), 6))

    static_samples = np.sum(movement < 1e-6)

    print("Nearly static samples:", static_samples)

    return movement


# ==========================================
# CLASS DISTRIBUTION
# ==========================================

def inspect_distribution(y, splits):

    print("\n" + "=" * 55)
    print("5. CLASS DISTRIBUTION")
    print("=" * 55)

    df = pd.DataFrame({
        "word": y,
        "split": splits
    })

    distribution = pd.crosstab(
        df["word"],
        df["split"]
    )

    distribution["total"] = distribution.sum(axis=1)

    print(distribution.to_string())

    distribution.to_csv(DISTRIBUTION_FILE)

    print("\nTotal samples per split:")

    print(
        df["split"]
        .value_counts()
        .to_string()
    )

    return distribution


# ==========================================
# SIGNER ANALYSIS
# ==========================================

def inspect_signers(y, splits, signer_ids):

    print("\n" + "=" * 55)
    print("6. SIGNER ANALYSIS")
    print("=" * 55)

    df = pd.DataFrame({
        "word": y,
        "split": splits,
        "signer_id": signer_ids
    })

    print("Unique signers:", df["signer_id"].nunique())

    print("\nSigners per split:")

    print(
        df.groupby("split")["signer_id"]
        .nunique()
        .to_string()
    )

    # Check whether the same signer appears in multiple splits.
    signer_splits = df.groupby("signer_id")["split"].nunique()

    overlapping_signers = signer_splits[
        signer_splits > 1
    ]

    print("\nSigners appearing in multiple splits:",
          len(overlapping_signers))

    return overlapping_signers


# ==========================================
# SAMPLE QUALITY REPORT
# ==========================================

def create_quality_report(
    X,
    y,
    splits,
    video_ids,
    signer_ids,
    left_frames,
    right_frames,
    both_hands,
    any_hand,
    movement
):

    quality = pd.DataFrame({

        "video_id": video_ids,
        "word": y,
        "split": splits,
        "signer_id": signer_ids,

        "left_hand_frames": left_frames,
        "right_hand_frames": right_frames,
        "both_hand_frames": both_hands,
        "any_hand_frames": any_hand,

        "movement_score": movement,

        "has_nan": np.isnan(X).any(axis=(1, 2)),
        "has_infinity": np.isinf(X).any(axis=(1, 2)),

        "nearly_static": movement < 1e-6,
        "valid_hand_detection": any_hand > 0

    })

    quality.to_csv(
        QUALITY_FILE,
        index=False
    )

    print("\nQuality report saved:", QUALITY_FILE)

    return quality


# ==========================================
# MAIN
# ==========================================

def main():

    X, y, splits, video_ids, signer_ids = load_dataset()

    inspect_structure(X, y, splits)

    nan_count, inf_count = inspect_values(X)

    (
        left_frames,
        right_frames,
        both_hands,
        any_hand
    ) = inspect_hands(X)

    movement = inspect_movement(X)

    distribution = inspect_distribution(y, splits)

    overlapping_signers = inspect_signers(
        y,
        splits,
        signer_ids
    )

    quality = create_quality_report(
        X,
        y,
        splits,
        video_ids,
        signer_ids,
        left_frames,
        right_frames,
        both_hands,
        any_hand,
        movement
    )

    # ======================================
    # FINAL SUMMARY
    # ======================================

    summary = []

    summary.append("WLASL DATASET ANALYSIS")
    summary.append("=" * 45)

    summary.append(f"Total samples: {len(X)}")
    summary.append(f"Unique words: {len(np.unique(y))}")
    summary.append(f"Dataset shape: {X.shape}")

    summary.append(f"NaN values: {nan_count}")
    summary.append(f"Infinite values: {inf_count}")

    summary.append(
        f"Samples with hand detection: {np.sum(any_hand > 0)}"
    )

    summary.append(
        f"Samples with both hands: {np.sum(both_hands > 0)}"
    )

    summary.append(
        f"Nearly static samples: {np.sum(movement < 1e-6)}"
    )

    summary.append(
        f"Signers appearing in multiple splits: {len(overlapping_signers)}"
    )

    summary.append("\nSamples by split:")
    summary.append(
        pd.Series(splits).value_counts().to_string()
    )

    summary.append("\nSamples by word:")
    summary.append(
        pd.Series(y).value_counts().sort_index().to_string()
    )

    summary_text = "\n".join(summary)

    SUMMARY_FILE.write_text(
        summary_text,
        encoding="utf-8"
    )

    print("\n" + "=" * 55)
    print("ANALYSIS COMPLETE")
    print("=" * 55)

    print(summary_text)

    print("\nReports saved in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()