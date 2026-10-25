
import cv2
import mediapipe as mp
import time

# Load the AI hand detection model
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

# Open the MacBook camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera could not open.")
    raise SystemExit

print("Camera started! Show your hand.")
print("Press Q to quit.")

start_time = time.monotonic()
last_timestamp = -1

with HandLandmarker.create_from_options(options) as detector:
    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read camera frame.")
            break

        # Flip the image like a mirror
        frame = cv2.flip(frame, 1)

        # Convert OpenCV image to MediaPipe image
        rgb_frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # MediaPipe requires increasing timestamps
        timestamp_ms = int(
            (time.monotonic() - start_time) * 1000
        )
        timestamp_ms = max(
            timestamp_ms, last_timestamp + 1
        )
        last_timestamp = timestamp_ms

        # Detect hands
        result = detector.detect_for_video(
            mp_image, timestamp_ms
        )

        height, width, _ = frame.shape

        # Draw dots at all detected landmarks
        for hand in result.hand_landmarks:
            for landmark in hand:
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

        # Show the camera window
        cv2.imshow("SignBridge Hand Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
print("Hand tracking stopped.")
