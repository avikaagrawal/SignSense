import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from pathlib import Path
from tqdm import tqdm


# ==========================================
# PATHS AND CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

CLIPS_CSV = ROOT / "data/wlasl/extracted_clips.csv"
MODEL_PATH = ROOT / "models/hand_landmarker.task"

OUTPUT_DIR = ROOT / "data/wlasl/landmarks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "wlasl_landmarks.npz"
REPORT_FILE = OUTPUT_DIR / "landmark_report.csv"

SEQUENCE_LENGTH = 30
LANDMARKS_PER_HAND = 21
COORDINATES = 3

FEATURES_PER_HAND = LANDMARKS_PER_HAND * COORDINATES

# Left hand: 63
# Right hand: 63
# Presence indicators: 2
FEATURES_PER_FRAME = 128

# Each video gets a separate timestamp range.
TIMESTAMP_GAP_MS = 60000


# ==========================================
# NORMALIZE HAND LANDMARKS
# ==========================================

def normalize_hand(landmarks):

    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks],
        dtype=np.float32
    )

    # Use wrist as origin.
    wrist = points[0].copy()
    points -= wrist

    # Scale using maximum distance from wrist.
    scale = np.max(
        np.linalg.norm(points, axis=1)
    )

    if scale > 0:
        points /= scale

    return points.flatten()


# ==========================================
# PROCESS ONE FRAME
# ==========================================

def process_frame(result):

    features = np.zeros(
        FEATURES_PER_FRAME,
        dtype=np.float32
    )

    for hand_landmarks, handedness in zip(
        result.hand_landmarks,
        result.handedness
    ):

        if not handedness:
            continue

        label = handedness[0].category_name

        if label == "Left":
            start = 0
            presence_index = 126

        elif label == "Right":
            start = 63
            presence_index = 127

        else:
            continue

        normalized = normalize_hand(hand_landmarks)

        features[start:start + 63] = normalized
        features[presence_index] = 1.0

    return features


# ==========================================
# EXTRACT VIDEO SEQUENCE
# ==========================================

def extract_sequence(
    video_path,
    landmarker,
    timestamp_offset_ms=0
):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None, 0

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames < 2 or not np.isfinite(fps) or fps <= 0:
        cap.release()
        return None, 0

    # Select 30 evenly distributed frames.
    # np.unique prevents duplicate frame indices in short videos.
    frame_indices = np.unique(
        np.linspace(
            0,
            total_frames - 1,
            SEQUENCE_LENGTH,
            dtype=int
        )
    )

    sequence = []
    detected_frames = 0

    # Timestamps must increase across every call to the same
    # VIDEO-mode landmarker, including between different videos.
    previous_timestamp_ms = timestamp_offset_ms - 1

    for frame_index in frame_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_index)
        )

        success, frame = cap.read()

        if not success:
            cap.release()
            return None, detected_frames

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Calculate a globally increasing timestamp.
        timestamp_ms = (
            timestamp_offset_ms
            + int(frame_index * 1000 / fps)
        )

        timestamp_ms = max(
            timestamp_ms,
            previous_timestamp_ms + 1
        )

        previous_timestamp_ms = timestamp_ms

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        features = process_frame(result)

        if np.any(features[126:128]):
            detected_frames += 1

        sequence.append(features)

    cap.release()

    if not sequence:
        return None, 0

    # Pad short videos to exactly 30 frames by repeating
    # their final sampled frame.
    while len(sequence) < SEQUENCE_LENGTH:
        sequence.append(sequence[-1].copy())

    sequence = np.asarray(
        sequence[:SEQUENCE_LENGTH],
        dtype=np.float32
    )

    if detected_frames == 0:
        return None, 0

    return sequence, detected_frames


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():

    if not CLIPS_CSV.exists():
        raise FileNotFoundError(CLIPS_CSV)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    df = pd.read_csv(CLIPS_CSV)

    df = df[
        df["status"] == "extracted"
    ].copy()

    print("=" * 55)
    print("W L A S L  LANDMARK EXTRACTION")
    print("=" * 55)

    print("Videos:", len(df))
    print("Sequence length:", SEQUENCE_LENGTH)
    print("Features per frame:", FEATURES_PER_FRAME)

    # MediaPipe configuration.
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    X = []
    labels = []
    splits = []
    video_ids = []
    signer_ids = []

    report = []

    # Keep timestamps increasing across all videos.
    global_timestamp_offset = 0

    with mp.tasks.vision.HandLandmarker.create_from_options(
        options
    ) as landmarker:

        for row in tqdm(
            df.itertuples(index=False),
            total=len(df),
            desc="Extracting landmarks"
        ):

            video_path = Path(row.clip_path)

            sequence = None
            detected_frames = 0

            if video_path.exists():

                sequence, detected_frames = extract_sequence(
                    video_path,
                    landmarker,
                    timestamp_offset_ms=global_timestamp_offset
                )

            # Move to a new timestamp range for the next video.
            global_timestamp_offset += TIMESTAMP_GAP_MS

            if sequence is None:

                status = "no_hands_or_invalid_video"

            else:

                status = "success"

                X.append(sequence)
                labels.append(row.gloss)
                splits.append(row.split)
                video_ids.append(str(row.video_id))
                signer_ids.append(str(row.signer_id))

            report.append({
                "video_id": row.video_id,
                "gloss": row.gloss,
                "split": row.split,
                "signer_id": row.signer_id,
                "detected_frames": detected_frames,
                "status": status
            })

    # ======================================
    # SAVE EXTRACTION REPORT
    # ======================================

    report_df = pd.DataFrame(report)

    report_df.to_csv(
        REPORT_FILE,
        index=False
    )

    if not X:

        print("\nNo valid landmark sequences found.")
        print("Check the clips and MediaPipe detection settings.")
        return

    X = np.asarray(
        X,
        dtype=np.float32
    )

    # ======================================
    # SAVE DATASET
    # ======================================

    np.savez_compressed(
        OUTPUT_FILE,
        X=X,
        y=np.array(labels),
        splits=np.array(splits),
        video_ids=np.array(video_ids),
        signer_ids=np.array(signer_ids)
    )

    # ======================================
    # SUMMARY
    # ======================================

    print("\n" + "=" * 55)
    print("LANDMARK EXTRACTION SUMMARY")
    print("=" * 55)

    print("Successful:", len(X))
    print("Failed:", len(report) - len(X))
    print("Dataset shape:", X.shape)

    print("\nSamples by word:")

    print(
        pd.Series(labels)
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nSamples by split:")

    print(
        pd.Series(splits)
        .value_counts()
        .to_string()
    )

    print("\nAverage detected frames:")

    print(
        round(
            report_df["detected_frames"].mean(),
            2
        )
    )

    print("\nDataset saved:", OUTPUT_FILE)
    print("Report saved:", REPORT_FILE)


if __name__ == "__main__":
    main()