# SignSense – Real-Time ASL Alphabet Recognition

SignSense is a computer vision and machine learning project that recognises American Sign Language (ASL) alphabet gestures through a webcam.

It uses MediaPipe to detect hand landmarks and a Random Forest classifier enhanced with geometric hand features to predict alphabet signs.

## Features

- Real-time hand landmark detection using MediaPipe.
- ASL alphabet recognition (A–Z).
- Geometric feature extraction using joint angles and fingertip distances.
- Live prediction with confidence scores.
- Sentence formation from recognised letters.

## Tech Stack

- Python
- OpenCV
- MediaPipe
- NumPy
- Scikit-learn
- Pandas
- Joblib

## Model Performance

| Model | Test Accuracy |
|---|---:|
| Original Random Forest | 98.11% |
| Geometric Feature Random Forest | 98.48% |

Evaluation was performed on a held-out test split. These results reflect the dataset evaluation and do not guarantee the same accuracy under real-world webcam conditions.

## Project Structure

```text
SignSense/
├── app/
│   ├── main.py
│   ├── hand_tracker.py
│   ├── predictor.py
│   ├── sentence_builder.py
│   ├── ui.py
│   └── config.py
├── models/
│   └── hand_landmarker.task
├── saved_models/
│   └── Evaluation reports
├── training/
├── word_recognition/
├── requirements.txt
└── README.md
```

## Current Status

The ASL alphabet recognition application is functional locally. Browser-based deployment is in progress.

## Scope and Limitations

SignSense currently focuses on static ASL alphabet recognition. It is not a complete sign language translation system and does not yet recognise full conversational sign language.

## Future Improvements

- Browser-based deployment.
- Improved recognition under different lighting and camera conditions.
- Better handling of visually similar alphabet signs.
- Expanded sign vocabulary.

## Author

Avika Agrawal

GitHub: https://github.com/avikaagrawal