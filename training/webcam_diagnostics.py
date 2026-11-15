import os
import sys
import csv
import cv2
import time
import joblib
import numpy as np
import mediapipe as mp

from datetime import datetime

# Project paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    ROOT, "saved_models", "asl_classifier_both_hands.pkl"
)

ENCODER_PATH = os.path.join(
    ROOT, "saved_models", "label_encoder.pkl"
)

HAND_MODEL_PATH = os.path.join(
    ROOT, "models", "hand_landmarker.task"
)

OUTPUT_DIR = os.path.join(ROOT, "data", "webcam_diagnostics")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_DIR, "diagnostic_results.csv")

# Load classifier
model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = HandLandmarker.create_from_options(options)

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Could not open webcam.")
    landmarker.close()
    sys.exit()

# CSV setup
file_exists = os.path.exists(CSV_PATH)

csv_file = open(CSV_PATH, "a", newline="")
writer = csv.writer(csv_file)

if not file_exists or os.path.getsize(CSV_PATH) == 0:
    writer.writerow([
        "timestamp",
        "actual_label",
        "hand_side",
        "predicted_label",
        "confidence",
        "top_2",
        "top_3",
        "landmarks"
    ])
    csv_file.flush()

actual_label = "UNKNOWN"
hand_side = "UNKNOWN"

start_time = time.monotonic()
last_timestamp = -1
save_message = ""
save_message_time = 0

print("\nSIGN SENSE - WEBCAM DIAGNOSTICS")
print("--------------------------------")
print("A-Z : Select sign being tested")
print("1   : Select LEFT hand")
print("2   : Select RIGHT hand")
print("S   : Save current sample")
print("Q   : Quit")
print("--------------------------------\n")

try:
    while True:

        success, frame = cap.read()

        if not success:
            print("Could not read webcam frame.")
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = int((time.monotonic() - start_time) * 1000)
        timestamp = max(timestamp, last_timestamp + 1)
        last_timestamp = timestamp

        result = landmarker.detect_for_video(mp_image, timestamp)

        prediction = "NO HAND"
        confidence = 0.0
        top_predictions = []
        landmarks_flat = []

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand],
                dtype=np.float32
            )

            normalized = landmarks - landmarks[0].copy()

            max_distance = np.max(
                np.linalg.norm(normalized, axis=1)
            )

            if max_distance > 0:
                normalized /= max_distance

            features = normalized.flatten().reshape(1, -1)

            probabilities = model.predict_proba(features)[0]

            top_indices = np.argsort(probabilities)[::-1][:3]

            for index in top_indices:

                class_number = model.classes_[index]

                label = encoder.inverse_transform(
                    [class_number]
                )[0]

                score = float(probabilities[index])

                top_predictions.append((label, score))

            prediction = top_predictions[0][0]
            confidence = top_predictions[0][1]

            landmarks_flat = landmarks.flatten().tolist()

            height, width = frame.shape[:2]

            for lm in hand:
                x = int(lm.x * width)
                y = int(lm.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )

        # Dashboard
        cv2.putText(
            frame,
            f"Actual: {actual_label} | Hand: {hand_side}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Prediction: {prediction}",
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence * 100:.1f}%",
            (15, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        for i, (label, score) in enumerate(top_predictions):

            cv2.putText(
                frame,
                f"{i+1}. {label}: {score*100:.1f}%",
                (15, 140 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        if time.monotonic() - save_message_time < 1.5:
            cv2.putText(
                frame,
                save_message,
                (15, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.imshow("SignSense Diagnostics", frame)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("s"):
            if landmarks_flat and top_predictions:

                top_2 = (
                    f"{top_predictions[1][0]}:"
                    f"{top_predictions[1][1]:.4f}"
                )

                top_3 = (
                    f"{top_predictions[2][0]}:"
                    f"{top_predictions[2][1]:.4f}"
                )

                writer.writerow([
                    datetime.now().isoformat(),
                    actual_label,
                    hand_side,
                    prediction,
                    round(confidence, 4),
                    top_2,
                    top_3,
                    ";".join(map(str, landmarks_flat))
                ])

                csv_file.flush()

                save_message = "SAMPLE SAVED!"
                save_message_time = time.monotonic()

                print(
                    f"SAMPLE SAVED | Actual={actual_label} | "
                    f"Hand={hand_side} | "
                    f"Predicted={prediction} | "
                    f"Confidence={confidence:.2%}"
                )

            else:
                save_message = "NO HAND DETECTED!"
                save_message_time = time.monotonic()
                print("No hand detected. Sample not saved.")

        elif key == ord("1"):
            hand_side = "LEFT"
            print("Selected: LEFT hand")

        elif key == ord("2"):
            hand_side = "RIGHT"
            print("Selected: RIGHT hand")

        elif ord("a") <= key <= ord("z"):
            actual_label = chr(key).upper()
            print(f"Testing sign: {actual_label}")

finally:

    cap.release()
    landmarker.close()
    csv_file.close()
    cv2.destroyAllWindows()

    print("\nDiagnostics saved to:")
    print(CSV_PATH)