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

DATA_PATH = "data/geometric_features.csv"

MODEL_PATH = "saved_models/asl_geometric_model.pkl"
ENCODER_PATH = "saved_models/asl_geometric_encoder.pkl"

REPORT_PATH = "saved_models/geometric_classification_report.csv"
CONFUSION_PATH = "saved_models/geometric_confusion_matrix.csv"

# ==========================================
# LOAD DATA
# ==========================================

print("=" * 55)
print("SIGNSENSE - GEOMETRIC MODEL TRAINING")
print("=" * 55)

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
y = df["label"].astype(str)

print("\nDataset shape:", df.shape)
print("Feature count:", X.shape[1])

if X.shape[1] != 93:
    raise ValueError("Expected exactly 93 features.")

# ==========================================
# ENCODE LABELS
# ==========================================

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# ==========================================
# TRAIN/TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
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

    # First 63 features are landmark coordinates.
    landmarks = mirrored[:63].reshape(21, 3)

    landmarks[:, 0] *= -1

    mirrored[:63] = landmarks.flatten()

    # Recalculate geometric features after mirroring.
    # Distances and angles are invariant to reflection.
    return mirrored


print("\nApplying mirror augmentation...")

mirrored_samples = np.array(
    [mirror_landmarks(sample) for sample in X_train],
    dtype=np.float32
)

X_train_augmented = np.concatenate(
    [X_train, mirrored_samples],
    axis=0
)

y_train_augmented = np.concatenate(
    [y_train, y_train],
    axis=0
)

print("Augmented training samples:", len(X_train_augmented))

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

print("\nEvaluating model...")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n" + "=" * 55)
print("GEOMETRIC MODEL RESULTS")
print("=" * 55)

print("Test accuracy:", round(accuracy * 100, 2), "%")

report_text = classification_report(
    y_test,
    predictions,
    labels=np.arange(len(encoder.classes_)),
    target_names=encoder.classes_,
    zero_division=0
)

print("\nClassification Report:")
print(report_text)

report = classification_report(
    y_test,
    predictions,
    labels=np.arange(len(encoder.classes_)),
    target_names=encoder.classes_,
    output_dict=True,
    zero_division=0
)

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

# ==========================================
# SAVE RESULTS
# ==========================================

os.makedirs("saved_models", exist_ok=True)

joblib.dump(model, MODEL_PATH)
joblib.dump(encoder, ENCODER_PATH)

pd.DataFrame(report).transpose().to_csv(REPORT_PATH)

confusion_df.to_csv(CONFUSION_PATH)

print("\n" + "=" * 55)
print("EXPERIMENT COMPLETE")
print("=" * 55)

print("Model:", MODEL_PATH)
print("Encoder:", ENCODER_PATH)
print("Report:", REPORT_PATH)
print("Confusion matrix:", CONFUSION_PATH)