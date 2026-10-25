
# SignBridge — Testing Results

## Project
SignBridge: Gesture-Based Communication System

## Classifier
- Algorithm: K-Nearest Neighbors (KNN)
- Programming language: C++
- K value: 3
- Input features: 42 normalized hand landmark coordinates
- Supported gestures: OPEN, FIST, PEACE

## Dataset
- OPEN: 165 examples
- FIST: 213 examples
- PEACE: 82 examples
- Total: 460 examples

## Test 1: Leave-One-Out Evaluation

Each saved example was classified using the other
459 examples as potential neighbors.

| Gesture | Correct | Total | Accuracy |
|---------|---------|-------|----------|
| OPEN | 162 | 165 | 98.18% |
| FIST | 209 | 213 | 98.12% |
| PEACE | 82 | 82 | 100.00% |
| Overall | 453 | 460 | 98.48% |

## Test 2: New Live Gesture Test

The webcam system was tested with 15 new attempts.

| Gesture | Correct | Total |
|---------|---------|-------|
| OPEN | 5 | 5 |
| FIST | 5 | 5 |
| PEACE | 5 | 5 |
| Overall | 15 | 15 |

Observed live test accuracy: 100%.

## Interpretation

The classifier performed well on both saved examples
and a small live test.

However, the results do not establish accuracy
across different users, lighting conditions, or
camera setups.

The leave-one-out score may be optimistic because
similar examples were collected during the same
training sessions.

## Future Testing

- Test with multiple users.
- Test in different lighting conditions.
- Use a separate held-out test dataset.
- Measure accuracy with more live attempts.
- Record a confusion matrix.
