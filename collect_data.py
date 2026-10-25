
import cv2
import mediapipe as mp
import csv
import os
import time

# Create a folder to store gesture examples
os.makedirs("data", exist_ok=True)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera could not open.")
    raise SystemExit

print("Press O to save OPEN palm")
print("Press F to save FIST")
print("Press P to save PEACE")
print("Press Q to quit")

start_time = time.monotonic()
last_timestamp = -1

with HandLandmarker.create_from_options(options) as detector:
    with open("data/gestures.csv", "a", newline="") as file:
        writer = csv.writer(file)

        while True:
            success, frame = camera.read()

            if not success:
                print("Camera frame failed.")
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp = int(
                (time.monotonic() - start_time) * 1000
            )
            timestamp = max(timestamp, last_timestamp + 1)
            last_timestamp = timestamp

            result = detector.detect_for_video(
                image, timestamp
            )

            hand_data = None

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]

                # Use wrist as the reference point
                wrist = landmarks[0]

                # Normalize for hand size
                scale = max(
                    max(
                        abs(p.x - wrist.x),
                        abs(p.y - wrist.y)
                    )
                    for p in landmarks
                )
                scale = max(scale, 0.000001)

                hand_data = []

                for point in landmarks:
                    hand_data.extend([
                        (point.x - wrist.x) / scale,
                        (point.y - wrist.y) / scale
                    ])

                # Draw landmark dots
                height, width, _ = frame.shape

                for point in landmarks:
                    x = int(point.x * width)
                    y = int(point.y * height)
                    cv2.circle(
                        frame, (x, y), 4,
                        (0, 255, 0), -1
                    )

            cv2.putText(
                frame,
                "O = OPEN | F = FIST | P = PEACE | Q = QUIT",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.imshow("SignBridge Data Collector", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key in (ord("o"), ord("f"), ord("p")):
                if hand_data is None:
                    print("No hand detected. Try again.")
                    continue

                if key == ord("o"):
                    label = "OPEN"
                elif key == ord("f"):
                    label = "FIST"
                else:
                    label = "PEACE"

                writer.writerow([label] + hand_data)
                file.flush()

                print("Saved:", label)

camera.release()
cv2.destroyAllWindows()
print("Dataset saved to data/gestures.csv")
