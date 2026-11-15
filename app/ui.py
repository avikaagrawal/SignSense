import cv2
import time
import pyttsx3
import threading
import queue

cv2.setNumThreads(2)

from hand_tracker import HandTracker
from predictor import Predictor
from sentence_builder import SentenceBuilder
from ui import UI


# ==========================================
# TEXT-TO-SPEECH
# ==========================================

speech_queue = queue.Queue()


def speech_worker():

    engine = pyttsx3.init()

    engine.setProperty("rate", 155)
    engine.setProperty("volume", 1.0)

    while True:

        letter = speech_queue.get()

        if letter is None:
            break

        try:
            engine.say(letter)
            engine.runAndWait()

        except Exception as error:
            print("Speech error:", error)

        speech_queue.task_done()

    engine.stop()


def main():

    tracker = None
    cap = None

    speech_thread = threading.Thread(
        target=speech_worker,
        daemon=True
    )

    speech_thread.start()

    last_spoken_letter = None
    candidate_letter = None
    stable_frames = 0

    try:

        tracker = HandTracker()
        predictor = Predictor()
        builder = SentenceBuilder()
        ui = UI()

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Could not open webcam.")
            return

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

            frame = cv2.flip(frame, 1)

            current_time = time.monotonic()
            elapsed = current_time - prev_time
            fps = 1 / elapsed if elapsed > 0 else 0
            prev_time = current_time

            hand = tracker.detect(frame)

            prediction = "-"
            confidence = 0.0
            top_predictions = []

            if hand is not None:

                letter, confidence, top_predictions = (
                    predictor.predict_details(hand)
                )

                prediction = builder.update(letter, confidence)

                # SPEECH STABILITY CHECK
                if confidence >= 0.65:

                    if letter == candidate_letter:
                        stable_frames += 1

                    else:
                        candidate_letter = letter
                        stable_frames = 1

                    # Speak after 5 consecutive stable frames
                    if stable_frames >= 5:

                        if letter != last_spoken_letter:

                            speech_queue.put(letter)
                            last_spoken_letter = letter

                else:

                    candidate_letter = None
                    stable_frames = 0

            else:

                builder.reset()

                candidate_letter = None
                stable_frames = 0
                last_spoken_letter = None

            ui.draw_landmarks(frame, hand)

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

        speech_queue.put(None)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()