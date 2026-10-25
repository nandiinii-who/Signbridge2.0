
import cv2

print("Starting SignBridge camera test...")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera could not open.")
else:
    print("Camera opened successfully!")

    while True:
        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read camera frame.")
            break

        cv2.imshow("SignBridge Camera Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
print("Camera test finished.")
