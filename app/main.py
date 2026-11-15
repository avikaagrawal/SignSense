import cv2
import time

cv2.setNumThreads(2)

from hand_tracker import HandTracker
from predictor import Predictor
from sentence_builder import SentenceBuilder
from ui import UI


def main():
    tracker = None
    cap = None

    try:
        tracker = HandTracker()
        predictor = Predictor()
        builder = SentenceBuilder()
        ui = UI()

        # Open webcam
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Could not open webcam.")
            return

        # Webcam optimisation
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        prev_time = time.monotonic()

        while True:
            success, frame = cap.read()

            if not success:
                print("Could not read webcam frame.")
                break

            # Mirror camera
            frame = cv2.flip(frame, 1)

            # Calculate FPS
            current_time = time.monotonic()
            elapsed = current_time - prev_time
            fps = 1 / elapsed if elapsed > 0 else 0
            prev_time = current_time

            # Detect hand
            hand = tracker.detect(frame)

            prediction = "-"
            confidence = 0.0
            top_predictions = []

            if hand is not None:
                letter, confidence, top_predictions = (
                    predictor.predict_details(hand)
                )
                prediction = builder.update(letter, confidence)
                print(f"Prediction: {letter} | Confidence: {confidence:.2f}")
            else:
                builder.reset()

            # Draw landmarks
            ui.draw_landmarks(frame, hand)

            # Render dashboard
            final_frame = ui.render(
                frame,
                prediction,
                builder.sentence,
                hand is not None,
                fps,
                confidence,
                top_predictions
            )

            cv2.imshow("SignSense", final_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as error:
        print("SignSense error:", error)

    finally:
        if cap is not None:
            cap.release()

        if tracker is not None:
            tracker.close()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()