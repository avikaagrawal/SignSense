import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# PATHS
# ==========================================

DATA_PATH = "data/normalized_landmarks.csv"

MODEL_PATH = "saved_models/asl_classifier_both_hands.pkl"
ENCODER_PATH = "saved_models/label_encoder.pkl"

REPORT_PATH = "saved_models/classification_report.csv"
CONFUSION_PATH = "saved_models/confusion_matrix.csv"


# ==========================================
# LOAD DATASET
# ==========================================

print("=" * 55)
print("SIGN SENSE - ALPHABET MODEL TRAINING")
print("=" * 55)

df = pd.read_csv(DATA_PATH)

df = df[df["label"] != "nothing"].copy()

print("\nOriginal dataset shape:", df.shape)

# Separate features and labels.
X = df.drop(columns=["label"]).astype(np.float32)
y = df["label"].astype(str)

if X.shape[1] != 63:
    raise ValueError(
        f"Expected 63 features, received {X.shape[1]}"
    )

# Check class distribution.
print("\nOriginal class distribution:")
print(y.value_counts().sort_index())


# ==========================================
# ENCODE LABELS
# ==========================================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

print("\nClasses:", list(encoder.classes_))
print("Number of classes:", len(encoder.classes_))


# ==========================================
# SPLIT BEFORE AUGMENTATION
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X.to_numpy(),
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# MIRROR AUGMENTATION
# ==========================================

def mirror_landmarks(features):

    mirrored = features.copy()

    # Reshape 63 features into 21 landmarks.
    mirrored = mirrored.reshape(21, 3)

    # Reflect X coordinates.
    mirrored[:, 0] *= -1

    return mirrored.flatten()


print("\nApplying augmentation to training data...")

mirrored_samples = np.array(
    [
        mirror_landmarks(sample)
        for sample in X_train
    ],
    dtype=np.float32
)

# Combine original and mirrored training samples.
X_train_augmented = np.concatenate(
    [X_train, mirrored_samples],
    axis=0
)

y_train_augmented = np.concatenate(
    [y_train, y_train],
    axis=0
)

print("Original training samples:", len(X_train))
print("Mirrored training samples:", len(mirrored_samples))
print("Total training samples:", len(X_train_augmented))

print("Testing samples remain unchanged:", len(X_test))


# ==========================================
# TRAIN MODEL
# ==========================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(
    X_train_augmented,
    y_train_augmented
)

print("Training completed!")


# ==========================================
# EVALUATION
# ==========================================

print("\nEvaluating model on original test samples...")

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n" + "=" * 55)
print("MODEL EVALUATION")
print("=" * 55)

print("Test accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")

report = classification_report(
    y_test,
    predictions,
    labels=np.arange(len(encoder.classes_)),
    target_names=encoder.classes_,
    output_dict=True,
    zero_division=0
)

print(
    classification_report(
        y_test,
        predictions,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0
    )
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    predictions,
    labels=np.arange(len(encoder.classes_))
)

confusion_df = pd.DataFrame(
    cm,
    index=encoder.classes_,
    columns=encoder.classes_
)

print("\nConfusion Matrix:")
print(confusion_df)

# ==========================================
# SAVE RESULTS
# ==========================================

os.makedirs("saved_models", exist_ok=True)

joblib.dump(model, MODEL_PATH)
joblib.dump(encoder, ENCODER_PATH)

pd.DataFrame(report).transpose().to_csv(
    REPORT_PATH
)

confusion_df.to_csv(
    CONFUSION_PATH
)

print("\n" + "=" * 55)
print("TRAINING COMPLETE")
print("=" * 55)

print("Model saved:", MODEL_PATH)
print("Encoder saved:", ENCODER_PATH)
print("Classification report:", REPORT_PATH)
print("Confusion matrix:", CONFUSION_PATH)