import pandas as pd

DATA_PATH = "data/landmarks.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

print("\nNumber of classes:")
print(df["label"].nunique())

print("\nSamples per class:")
counts = df["label"].value_counts().sort_index()
print(counts)

print("\nMinimum samples:")
print(counts.idxmin(), counts.min())

print("\nMaximum samples:")
print(counts.idxmax(), counts.max())

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())