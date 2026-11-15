import cv2
import mediapipe as mp
import numpy as np
import time
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
DATASET_PATH = ROOT / "word_recognition" / "dataset"

DATASET_PATH.mkdir(parents=True, exist_ok=True)

# Configuration
SEQUENCE_LENGTH = 30
NUM_HANDS = 2

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=NUM_HANDS
)

landmarker = HandLandmarker.create_from_options(options)

# Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

sequence = []
recording = False
sign_label = input("Enter ASL sign label: ").strip().lower()

if not sign_label.isalnum():
    raise ValueError("Use a simple label containing letters or numbers.")

print("\nControls:")
print("S - Start recording")
print("Q - Quit")

start_time = time.time()


def extract_landmarks(result):
    """
    Returns 126 values:
    21 landmarks x 3 coordinates x 2 hands.

    Missing hands are represented by zeros.
    """

    features = np.zeros((NUM_HANDS, 21, 3), dtype=np.float32)

    if not result.hand_landmarks:
        return features.flatten()

    for i, hand in enumerate(result.hand_landmarks[:NUM_HANDS]):
        for j, landmark in enumerate(hand):
            features[i, j] = [
                landmark.x,
                landmark.y,
                landmark.z
            ]

    return features.flatten()


try:
    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = int((time.time() - start_time) * 1000)

        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )

        cv2.putText(
            frame,
            f"Sign: {sign_label.upper()}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        if recording:

            landmarks = extract_landmarks(result)
            sequence.append(landmarks)

            cv2.putText(
                frame,
                f"Recording: {len(sequence)}/{SEQUENCE_LENGTH}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            if len(sequence) >= SEQUENCE_LENGTH:

                filename = (
                    DATASET_PATH /
                    f"{sign_label}_{int(time.time() * 1000)}.npy"
                )

                np.save(filename, np.array(sequence))

                print("Saved:", filename.name)

                sequence = []
                recording = False

        cv2.imshow("SignSense - Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s") and not recording:
            sequence = []
            recording = True
            print("Recording started...")

        elif key == ord("q"):
            break

finally:
    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()