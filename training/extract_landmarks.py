import os
import csv
import cv2
import mediapipe as mp

DATASET_PATH = "dataset/asl_alphabet_train/asl_alphabet_train"
MODEL_PATH = "models/hand_landmarker.task"
OUTPUT_PATH = "data/landmarks.csv"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5
)

# Get all classes
classes = sorted([
    folder for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, folder))
])

print(f"Classes found: {len(classes)}")
print(classes)

total_success = 0
total_failed = 0

os.makedirs("data", exist_ok=True)

with HandLandmarker.create_from_options(options) as landmarker:

    with open(OUTPUT_PATH, "w", newline="") as file:

        writer = csv.writer(file)

        # CSV header
        header = ["label"]

        for i in range(21):
            header.extend([
                f"x{i}",
                f"y{i}",
                f"z{i}"
            ])

        writer.writerow(header)

        # Process every class
        for class_name in classes:

            folder = os.path.join(DATASET_PATH, class_name)

            images = [
                image for image in os.listdir(folder)
                if image.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            class_success = 0
            class_failed = 0

            print(f"\nProcessing {class_name} ({len(images)} images)...")

            for index, image_name in enumerate(images):

                image_path = os.path.join(folder, image_name)

                image = cv2.imread(image_path)

                if image is None:
                    class_failed += 1
                    continue

                rgb_image = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_image
                )

                try:
                    result = landmarker.detect(mp_image)

                    if result.hand_landmarks:

                        hand = result.hand_landmarks[0]

                        row = [class_name]

                        for landmark in hand:
                            row.extend([
                                landmark.x,
                                landmark.y,
                                landmark.z
                            ])

                        writer.writerow(row)

                        class_success += 1

                    else:
                        class_failed += 1

                except Exception as error:
                    class_failed += 1

                # Progress every 500 images
                if (index + 1) % 500 == 0:
                    print(
                        f"  {index + 1}/{len(images)} processed"
                    )

            total_success += class_success
            total_failed += class_failed

            print(
                f"{class_name}: "
                f"{class_success} successful | "
                f"{class_failed} failed"
            )

print("\n==============================")
print("LANDMARK EXTRACTION COMPLETE")
print("==============================")

print("Successful:", total_success)
print("Failed:", total_failed)
print("Total:", total_success + total_failed)

print("\nSaved to:", OUTPUT_PATH)