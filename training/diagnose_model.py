
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "augmented_landmarks.csv"
MODEL_PATH = ROOT / "saved_models" / "asl_classifier_both_hands.pkl"
ENCODER_PATH = ROOT / "saved_models" / "label_encoder.pkl"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Exclude the unusable 'nothing' class
df = df[df["label"] != "nothing"].copy()

X = df.drop(columns=["label"])
y = df["label"]

print("Dataset shape:", X.shape)
print("\nLoading trained model...")

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

# Use a stratified split to evaluate a held-out portion of the dataset.
# This is only a preliminary diagnostic, not a fully independent test.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

predictions = model.predict(X_test)

# Decode predictions if the model returns encoded labels
if hasattr(predictions, "dtype") and predictions.dtype.kind in "iu":
    predictions = encoder.inverse_transform(predictions)

print("\nAccuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")

print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

print("\nConfusion matrix for T, V, R, L, U:")
labels = ["T", "V", "R", "L", "U"]
cm = confusion_matrix(y_test, predictions, labels=labels)

print("Rows = actual labels; columns = predicted labels")
print("Column order:", labels)
print(cm)
