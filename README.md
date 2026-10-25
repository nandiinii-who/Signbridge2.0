
# SignBridge — Gesture-Based Communication System

## Project Overview

SignBridge is a prototype that recognizes a small set of
custom hand gestures and converts them into text and speech.

The system uses a webcam to detect hand movements and
a C++ machine learning classifier to identify gestures.

## Technologies Used

- Python
- C++
- OpenCV
- MediaPipe
- K-Nearest Neighbors (KNN)
- macOS Text-to-Speech

## Supported Gestures

| Gesture | Message |
|---------|---------|
| OPEN | Hello! |
| FIST | I need help. |
| PEACE | Thank you! |

Note: These are custom gesture-to-message mappings,
not official sign-language translations.

## Features

- Real-time webcam hand detection
- Recognition of three hand gestures
- KNN classification implemented in C++
- Prediction smoothing
- Text display of recognized messages
- Speech output
- Communication history with timestamps
- View and clear communication history

## How to Run

1. Open the project folder in VS Code.
2. Activate the Python virtual environment.
3. Run `live_recognition.py`.

On macOS:

```bash
cd ~/SignBridge-Fresh
source .venv/bin/activate
python live_recognition.py
```

## Keyboard Controls

- S: Speak and save the recognized message
- H: View communication history in Terminal
- C: Clear communication history (press twice)
- Q: Quit

## How It Works

1. OpenCV captures video from the webcam.
2. MediaPipe detects 21 hand landmarks.
3. Python normalizes the landmark coordinates.
4. The 42 coordinate values are passed to C++.
5. A KNN classifier predicts the gesture.
6. Python displays and speaks the associated message.

## Limitations

- Recognizes only three custom gestures.
- Requires adequate lighting and a visible hand.
- Accuracy may vary with users and hand positions.
- Does not translate complete sign language.

## Future Improvements

- Support more gestures and messages.
- Test recognition with different users.
- Improve accuracy using a larger dataset.
- Develop a graphical user interface.
