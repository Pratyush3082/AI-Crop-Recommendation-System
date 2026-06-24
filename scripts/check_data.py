# scripts/check_data.py
import pandas as pd
import os

BASE = os.path.dirname(os.path.dirname(__file__))
path = os.path.join(BASE, "data", "crop_data.csv")
df = pd.read_csv(path)

print("Shape:", df.shape)
print("\nColumns and dtypes:\n", df.dtypes)
print("\nMissing values per column:\n", df.isna().sum())
print("\nCrop value counts:\n", df['crop'].value_counts())
print("\nNumeric summary (features):\n", df.describe().T)
