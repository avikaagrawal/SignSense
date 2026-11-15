import cv2
import mediapipe as mp

# Paths
image_path = "dataset/asl_alphabet_test/asl_alphabet_test/W_test.jpg"
model_path = "models/hand_landmarker.task"

# Load image
image = cv2.imread(image_path)

if image is None:
    print("Could not load image.")
    exit()

# Create Hand Landmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

with HandLandmarker.create_from_options(options) as landmarker:

    # Convert OpenCV BGR → RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image
    )

    # Detect
    result = landmarker.detect(mp_image)

    if result.hand_landmarks:

        print("Hand detected!")

        height, width, _ = image.shape

        for hand in result.hand_landmarks:

            for i, landmark in enumerate(hand):

                print(
                    f"Landmark {i}: "
                    f"x={landmark.x:.3f}, "
                    f"y={landmark.y:.3f}, "
                    f"z={landmark.z:.3f}"
                )

                # Convert normalized coordinates to pixels
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(image, (x, y), 4, (0, 255, 0), -1)

    else:
        print("No hand detected.")

cv2.imshow("ASL Hand Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()