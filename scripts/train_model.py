# scripts/train_model.py
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.multiclass import unique_labels

# ---------- paths ----------
# ✅ FIXED: use _file_, not file
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root (croprec/)
data_path = os.path.join(BASE_DIR, "data", "crop_data.csv")
ml_dir = os.path.join(BASE_DIR, "recommender", "ml")
os.makedirs(ml_dir, exist_ok=True)  # ensure folder exists

# ---------- 1. Load data ----------
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found at: {data_path}\nPlease place crop_data.csv in the data/ folder.")

df = pd.read_csv(data_path)
print("✅ Data loaded successfully. Shape:", df.shape)

# ---------- 2. Select features & target ----------
features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
target = "crop"

missing = [c for c in features + [target] if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing columns in CSV: {missing}")

# ---------- 3. Clean data ----------
df = df.dropna(subset=features + [target])
X = df[features].copy()
y = df[target].copy()

# ---------- 4. Encode labels ----------
le = LabelEncoder()
y_enc = le.fit_transform(y)
print("Crop classes:", list(le.classes_))

# ---------- 5. Split data ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42
)

# ---------- 6. Train model ----------
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# ---------- 7. Evaluate ----------
y_pred = clf.predict(X_test)
print("✅ Training complete")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

labels_in_test = unique_labels(y_test)
print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=labels_in_test,
        target_names=[le.classes_[i] for i in labels_in_test]
    )
)

# ---------- 8. Save artifact ----------
artifact = {"model": clf, "label_encoder": le, "features": features}
out_path = os.path.join(ml_dir, "crop_recommender_v1.joblib")
joblib.dump(artifact, out_path)
print("💾 Saved model artifact to:", out_path)