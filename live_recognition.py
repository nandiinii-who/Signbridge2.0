
import cv2
import mediapipe as mp
import subprocess
import time
from datetime import datetime
from collections import deque, Counter


def speak(text):
    subprocess.Popen(["say", text])


gesture_phrases = {
    "OPEN": "Hello!",
    "FIST": "I need help.",
    "PEACE": "Thank you!"
}

# Load MediaPipe's hand detection model
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

# Open the camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera could not open.")
    raise SystemExit

print("SignBridge live recognition started!")
print("Show an OPEN palm or a FIST.")
print("Press Q to quit.")

start_time = time.monotonic()
last_timestamp = -1
prediction = "Waiting for hand..."
prediction_history = deque(maxlen=5)
frame_count = 0

with HandLandmarker.create_from_options(options) as detector:
    while True:
        success, frame = camera.read()

        if not success:
            print("ERROR: Camera frame failed.")
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

        frame_count += 1

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            wrist = landmarks[0]

            # Normalize coordinates exactly as in collect_data.py
            scale = max(
                max(
                    abs(p.x - wrist.x),
                    abs(p.y - wrist.y)
                )
                for p in landmarks
            )
            scale = max(scale, 0.000001)

            coordinates = []

            for point in landmarks:
                coordinates.extend([
                    (point.x - wrist.x) / scale,
                    (point.y - wrist.y) / scale
                ])

            # Run C++ prediction every 5 frames
            if frame_count % 5 == 0:
                input_data = " ".join(
                    str(value) for value in coordinates
                )

                response = subprocess.run(
                    ["./predict_gesture"],
                    input=input_data,
                    text=True,
                    capture_output=True
                )

                if response.returncode == 0:
                    new_prediction = response.stdout.strip()

                    if new_prediction in ("OPEN", "FIST", "PEACE"):
                        prediction_history.append(new_prediction)

                        # Find the most common recent gesture
                        most_common, count = Counter(
                            prediction_history
                        ).most_common(1)[0]

                        # Update only when at least 3 predictions agree
                        if count >= 3:
                            prediction = most_common
                        else:
                            prediction = "Recognizing..."
                else:
                    prediction = "Prediction error"
                    prediction_history.clear()
                    print(response.stderr)

            # Draw green dots on the hand
            height, width, _ = frame.shape

            for point in landmarks:
                x = int(point.x * width)
                y = int(point.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )
        else:
            prediction = "No hand detected"
            prediction_history.clear()

        # Display the predicted gesture and its translated phrase
        display_phrase = gesture_phrases.get(
            prediction,
            "Waiting for a recognized gesture..."
        )

        # Make the top information panel taller
        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1], 160),
            (30, 30, 30),
            -1
        )

        # Show whether a hand is currently detected
        if result.hand_landmarks:
            status_text = "Hand detected"
            status_color = (0, 255, 0)
        else:
            status_text = "No hand detected"
            status_color = (0, 0, 255)

        cv2.putText(
            frame,
            status_text,
            (15, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_color,
            2
        )

        cv2.putText(
            frame,
            "Gesture: " + prediction,
            (15, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            display_phrase,
            (15, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
" S: Speak | H: History | C: Clear | Q: Quit",
            (15, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        cv2.imshow("SignBridge - Live Recognition", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            if prediction in gesture_phrases:
                phrase = gesture_phrases[prediction]

                # Speak the recognized phrase
                speak(phrase)
                print("Speaking:", phrase)

                # Save the phrase to communication history
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open("communication_history.txt", "a") as file:
                    file.write(f"[{timestamp}] {phrase}\n")

                print("Saved to communication history.")
            else:
                print("Show your hand before pressing S.")

        if key == ord("h"):
            print("\n--- Communication History ---")

            try:
                with open("communication_history.txt", "r") as file:
                    history = file.read()

                if history.strip():
                    print(history)
                else:
                    print("No messages saved yet.")

            except FileNotFoundError:
                print("No communication history found yet.")

            print("-----------------------------\n")

        if key == ord("c"):
            print("Press C again to confirm clearing history.")
            print("Press any other key to cancel.")

            confirmation = cv2.waitKey(0) & 0xFF

            if confirmation == ord("c"):
                with open("communication_history.txt", "w") as file:
                    pass

                print("Communication history cleared.")
            else:
                print("Clear cancelled.")

        if key == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
print("SignBridge stopped.")
