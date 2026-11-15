import cv2
import mediapipe as mp
import time

model_path = "models/hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
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

with HandLandmarker.create_from_options(options) as landmarker:

    start_time = time.time()

    while True:

        success, frame = cap.read()

        if not success:
            print("Could not read webcam frame.")
            break

        # Mirror webcam
        frame = cv2.flip(frame, 1)

        # BGR → RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # VIDEO mode requires timestamp
        timestamp_ms = int((time.time() - start_time) * 1000)

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # Draw landmarks
        if result.hand_landmarks:

            height, width, _ = frame.shape

            for hand in result.hand_landmarks:

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

        cv2.putText(
            frame,
            "Press Q to quit",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow("ASL - Live Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()