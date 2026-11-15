import time
import cv2
import mediapipe as mp

from config import HAND_MODEL_PATH


class HandTracker:

    def __init__(self):

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=HAND_MODEL_PATH
            ),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.landmarker = HandLandmarker.create_from_options(
            options
        )

        # Monotonic clock prevents timestamp inconsistencies.
        self.start_time = time.monotonic()
        self.last_timestamp = -1

    def detect(self, frame):

        # Preserve the original aspect ratio while resizing.
        height, width = frame.shape[:2]

        scale = min(640 / width, 360 / height)

        new_width = int(width * scale)
        new_height = int(height * scale)

        small_frame = cv2.resize(
            frame,
            (new_width, new_height)
        )

        # Convert BGR to RGB.
        rgb = cv2.cvtColor(
            small_frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # Generate strictly increasing timestamps.
        timestamp = int(
            (time.monotonic() - self.start_time) * 1000
        )

        timestamp = max(
            timestamp,
            self.last_timestamp + 1
        )

        self.last_timestamp = timestamp

        result = self.landmarker.detect_for_video(
            mp_image,
            timestamp
        )

        if result.hand_landmarks:

            # Return the first detected hand.
            return result.hand_landmarks[0]

        return None

    def close(self):

        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None