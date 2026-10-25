
import csv
import subprocess

# Read one saved gesture example
with open("data/gestures.csv", "r") as file:
    reader = csv.reader(file)
    example = next(reader)

actual_gesture = example[0]
coordinates = example[1:]

# Convert the coordinates into text for C++
input_data = " ".join(coordinates)

# Send coordinates to the C++ classifier
result = subprocess.run(
    ["./predict_gesture"],
    input=input_data,
    text=True,
    capture_output=True
)

if result.returncode != 0:
    print("C++ ERROR:", result.stderr)
else:
    print("Actual gesture:", actual_gesture)
    print("C++ prediction:", result.stdout.strip())
    print("Python and C++ are communicating!")
