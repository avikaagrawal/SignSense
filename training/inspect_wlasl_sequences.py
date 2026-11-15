import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data/wlasl/landmarks/wlasl_landmarks.npz"

OUTPUT_DIR = ROOT / "data/wlasl/landmarks/visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# MediaPipe hand skeleton connections
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

LEFT_START = 0
RIGHT_START = 63

LEFT_PRESENCE = 126
RIGHT_PRESENCE = 127

FRAME_POINTS = [0, 14, 29]


# ==========================================
# LOAD DATA
# ==========================================

def load_data():

    data = np.load(DATA_FILE)

    return (
        data["X"],
        data["y"],
        data["splits"],
        data["video_ids"]
    )


# ==========================================
# DRAW ONE HAND
# ==========================================

def draw_hand(ax, frame, start, presence_index, title):

    detected = frame[presence_index] > 0.5

    ax.set_title(title)

    if not detected:
        ax.text(
            0.5, 0.5,
            "Hand not detected",
            ha="center",
            va="center",
            transform=ax.transAxes
        )

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.grid(True)
        return

    points = frame[start:start + 63].reshape(21, 3)

    x = points[:, 0]
    y = -points[:, 1]

    for a, b in CONNECTIONS:

        ax.plot(
            [x[a], x[b]],
            [y[a], y[b]]
        )

    ax.scatter(x, y, s=15)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)

    ax.set_aspect("equal")
    ax.grid(True)


# ==========================================
# VISUALIZE ONE VIDEO
# ==========================================

def visualize_sample(sequence, word, video_id):

    fig, axes = plt.subplots(
        3, 2,
        figsize=(10, 12)
    )

    fig.suptitle(
        f"Word: {word} | Video: {video_id}",
        fontsize=15
    )

    for row, frame_index in enumerate(FRAME_POINTS):

        frame = sequence[frame_index]

        draw_hand(
            axes[row, 0],
            frame,
            LEFT_START,
            LEFT_PRESENCE,
            f"Left hand - Frame {frame_index}"
        )

        draw_hand(
            axes[row, 1],
            frame,
            RIGHT_START,
            RIGHT_PRESENCE,
            f"Right hand - Frame {frame_index}"
        )

    plt.tight_layout()

    filename = OUTPUT_DIR / f"{word}_{video_id}.png"

    plt.savefig(
        filename,
        dpi=120,
        bbox_inches="tight"
    )

    plt.close(fig)


# ==========================================
# MAIN
# ==========================================

def main():

    X, y, splits, video_ids = load_data()

    print("=" * 55)
    print("WLASL LANDMARK VISUAL INSPECTION")
    print("=" * 55)

    print("Samples:", len(X))
    print("Words:", len(np.unique(y)))

    summary = []

    for word in sorted(np.unique(y)):

        indices = np.where(y == word)[0]

        # Select the sample with the most frames
        # containing both hands, for inspection.
        both_hand_counts = [
            np.sum(
                (X[i, :, LEFT_PRESENCE] > 0.5) &
                (X[i, :, RIGHT_PRESENCE] > 0.5)
            )
            for i in indices
        ]

        selected_position = int(np.argmax(both_hand_counts))
        selected_index = indices[selected_position]

        sequence = X[selected_index]

        left_frames = np.sum(
            sequence[:, LEFT_PRESENCE] > 0.5
        )

        right_frames = np.sum(
            sequence[:, RIGHT_PRESENCE] > 0.5
        )

        both_frames = np.sum(
            (sequence[:, LEFT_PRESENCE] > 0.5) &
            (sequence[:, RIGHT_PRESENCE] > 0.5)
        )

        visualize_sample(
            sequence,
            word,
            video_ids[selected_index]
        )

        summary.append({
            "word": word,
            "video_id": video_ids[selected_index],
            "split": splits[selected_index],
            "left_hand_frames": left_frames,
            "right_hand_frames": right_frames,
            "both_hand_frames": both_frames
        })

        print(f"Generated visualization: {word}")

    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(
        OUTPUT_DIR / "visualization_summary.csv",
        index=False
    )

    print("\nInspection completed.")

    print("Saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()