# scripts/calibrate.py
import os, joblib, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.dirname(__file__))
df = pd.read_csv(os.path.join(BASE,"data","crop_data.csv")).dropna(subset=["crop"])
features = ["N","P","K","temperature","humidity","ph","rainfall"]
X = df[features].fillna(df[features].median())
y = df["crop"]
le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_calib, y_train, y_calib = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

base = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1)
base.fit(X_train, y_train)

calibrator = CalibratedClassifierCV(base, cv="prefit", method="isotonic")
calibrator.fit(X_calib, y_calib)

joblib.dump({"model":calibrator, "label_encoder":le, "features":features}, os.path.join(BASE,"recommender","ml","crop_recommender_v1.joblib"))
print("Saved calibrated model.")
