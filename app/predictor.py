import joblib
import numpy as np

from config import MODEL_PATH, ENCODER_PATH


# ==========================================
# MODEL SELECTION
# ==========================================

# Options: "original" or "geometric"
MODEL_VARIANT = "geometric"

GEOMETRIC_MODEL_PATH = "saved_models/asl_geometric_model.pkl"
GEOMETRIC_ENCODER_PATH = "saved_models/asl_geometric_encoder.pkl"


# ==========================================
# LANDMARK CONFIGURATION
# ==========================================

FINGER_LANDMARKS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20]
}

FINGERTIPS = [4, 8, 12, 16, 20]


class Predictor:

    def __init__(self):

        if MODEL_VARIANT == "geometric":

            model_path = GEOMETRIC_MODEL_PATH
            encoder_path = GEOMETRIC_ENCODER_PATH

        elif MODEL_VARIANT == "original":

            model_path = MODEL_PATH
            encoder_path = ENCODER_PATH

        else:
            raise ValueError(
                "MODEL_VARIANT must be 'original' or 'geometric'"
            )

        self.model = joblib.load(model_path)
        self.encoder = joblib.load(encoder_path)

        self.use_geometric = MODEL_VARIANT == "geometric"

        if not hasattr(self.model, "predict_proba"):
            raise TypeError(
                "Model must support predict_proba()."
            )

        print(f"Loaded {MODEL_VARIANT} model: {model_path}")

    # ======================================
    # NORMALIZE LANDMARKS
    # ======================================

    def normalize_landmarks(self, hand):

        landmarks = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand],
            dtype=np.float32
        )

        if landmarks.shape != (21, 3):
            raise ValueError(
                "Expected 21 hand landmarks."
            )

        # Wrist as origin.
        landmarks -= landmarks[0].copy()

        # Normalize hand size.
        distances = np.linalg.norm(
            landmarks,
            axis=1
        )

        max_distance = np.max(distances)

        if max_distance > 0:
            landmarks /= max_distance

        return landmarks

    # ======================================
    # CALCULATE JOINT ANGLE
    # ======================================

    def calculate_angle(self, a, b, c):

        ba = a - b
        bc = c - b

        denominator = (
            np.linalg.norm(ba) *
            np.linalg.norm(bc)
        )

        if denominator < 1e-8:
            return 0.0

        cosine = np.dot(ba, bc) / denominator

        cosine = np.clip(
            cosine,
            -1.0,
            1.0
        )

        return np.degrees(
            np.arccos(cosine)
        )

    # ======================================
    # EXTRACT GEOMETRIC FEATURES
    # ======================================

    def extract_geometric_features(self, landmarks):

        features = []

        # ----------------------------------
        # 1. Finger joint angles (15)
        # ----------------------------------

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

                angle = self.calculate_angle(
                    landmarks[a],
                    landmarks[b],
                    landmarks[c]
                )

                features.append(angle / 180.0)

        # ----------------------------------
        # 2. Wrist-to-fingertip distances (5)
        # ----------------------------------

        for tip in FINGERTIPS:

            distance = np.linalg.norm(
                landmarks[tip] - landmarks[0]
            )

            features.append(distance)

        # ----------------------------------
        # 3. Fingertip distances (10)
        # ----------------------------------

        for i in range(len(FINGERTIPS)):

            for j in range(i + 1, len(FINGERTIPS)):

                distance = np.linalg.norm(
                    landmarks[FINGERTIPS[i]] -
                    landmarks[FINGERTIPS[j]]
                )

                features.append(distance)

        return np.array(
            features,
            dtype=np.float32
        )

    # ======================================
    # PREPARE MODEL INPUT
    # ======================================

    def prepare_features(self, hand):

        landmarks = self.normalize_landmarks(hand)

        coordinates = landmarks.flatten()

        if self.use_geometric:

            geometric = self.extract_geometric_features(
                landmarks
            )

            features = np.concatenate([
                coordinates,
                geometric
            ])

        else:

            features = coordinates

        expected_features = self.model.n_features_in_

        if len(features) != expected_features:
            raise ValueError(
                f"Model expects {expected_features} features, "
                f"but received {len(features)}."
            )

        return features.reshape(1, -1)

    # ======================================
    # PREDICTION
    # ======================================

    def predict_details(self, hand):

        features = self.prepare_features(hand)

        probabilities = self.model.predict_proba(
            features
        )[0]

        top_indices = np.argsort(
            probabilities
        )[::-1][:3]

        top_predictions = []

        for index in top_indices:

            class_number = self.model.classes_[index]

            letter = self.encoder.inverse_transform(
                [class_number]
            )[0]

            confidence = float(
                probabilities[index]
            )

            top_predictions.append({
                "letter": str(letter),
                "confidence": confidence
            })

        best = top_predictions[0]

        return (
            best["letter"],
            best["confidence"],
            top_predictions
        )

    def predict(self, hand):

        letter, confidence, _ = self.predict_details(hand)

        return letter, confidence