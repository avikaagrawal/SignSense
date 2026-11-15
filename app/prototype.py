from collections import deque, Counter
import cv2
import mediapipe as mp
import joblib
import numpy as np
import time

MODEL_PATH = "saved_models/asl_classifier_both_hands.pkl"
ENCODER_PATH = "saved_models/label_encoder.pkl"
HAND_MODEL_PATH = "models/hand_landmarker.task"

# Prediction settings
prediction_history = deque(maxlen=10)

# Sentence builder
sentence = ""
last_added_letter = None
last_add_time = 0

REPEAT_DELAY = 2.0
# Load models
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

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam.")
    exit()

start_time = time.time()

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        # Convert BGR -> RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(
            (time.time() - start_time) * 1000
        )

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        prediction_text = "No hand"

        # =====================================
        # HAND DETECTED
        # =====================================

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            # ---------------------------------
            # NORMALIZE LANDMARKS
            # ---------------------------------

            landmarks_array = np.array([
                [
                    landmark.x,
                    landmark.y,
                    landmark.z
                ]
                for landmark in hand
            ])

            # Move wrist to origin
            wrist = landmarks_array[0].copy()

            landmarks_array = (
                landmarks_array - wrist
            )

            # Normalize hand size
            distances = np.linalg.norm(
                landmarks_array,
                axis=1
            )

            max_distance = np.max(distances)

            if max_distance > 0:
                landmarks_array = (
                    landmarks_array / max_distance
                )

            # 21 x 3 = 63 features
            features = (
                landmarks_array
                .flatten()
                .reshape(1, -1)
            )

            # ---------------------------------
            # MODEL PREDICTION
            # ---------------------------------

            probabilities = model.predict_proba(
                features
            )[0]

            best_index = np.argmax(probabilities)

            confidence = probabilities[best_index]

            predicted_number = model.classes_[
                best_index
            ]

            predicted_letter = (
                encoder.inverse_transform(
                    [predicted_number]
                )[0]
            )

            # ---------------------------------
            # SMOOTHING
            # ---------------------------------

            if confidence >= 0.55:
                prediction_history.append(
                    predicted_letter
                )

            if len(prediction_history) >= 6:

                stable_letter, count = Counter(
                    prediction_history
                ).most_common(1)[0]

                stability = (
                    count / len(prediction_history)
                )

                # -----------------------------
                               # -----------------------------
                # STABLE SIGN
                # -----------------------------

                if stability >= 0.60:

                    prediction_text = (
                        f"{stable_letter} "
                        f"{confidence * 100:.1f}%"
                    )

                    # Add character only when sign changes
                    current_time = time.time()

# New sign -> add immediately
if stable_letter != last_added_letter:

    if stable_letter == "space":
        sentence += " "

    elif stable_letter == "del":
        sentence = sentence[:-1]

    else:
        sentence += stable_letter

    last_added_letter = stable_letter
    last_add_time = current_time

# Same sign held -> repeat after delay
elif current_time - last_add_time >= REPEAT_DELAY:

    if stable_letter == "space":
        sentence += " "

    elif stable_letter == "del":
        sentence = sentence[:-1]

    else:
        sentence += stable_letter

    last_add_time = current_time
                else:
                    prediction_text = "Detecting..."

            else:
                prediction_text = "Detecting..."
            # ---------------------------------

            height, width, _ = frame.shape

            for landmark in hand:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )

        # =====================================
        # NO HAND
        # =====================================

        else:

            prediction_history.clear()

            # Allow next character
            last_added_letter = None

        # =====================================
        # DISPLAY
        # =====================================

        # Current prediction
        cv2.putText(
            frame,
            prediction_text,
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (0, 255, 0),
            3
        )

        # Built sentence
        cv2.putText(
            frame,
            f"Text: {sentence}",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Press Q to quit",
            (30, frame.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Real-Time ASL Interpreter",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()