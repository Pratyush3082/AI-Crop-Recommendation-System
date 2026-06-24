# scripts/train_balanced.py
import os, joblib, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

BASE = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE, "data", "crop_data.csv")
ml_dir = os.path.join(BASE, "recommender", "ml")
os.makedirs(ml_dir, exist_ok=True)

df = pd.read_csv(data_path)
features = ["N","P","K","temperature","humidity","ph","rainfall"]
df = df.dropna(subset=features+["crop"])
X = df[features].fillna(df[features].median())
y = df["crop"]
le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

clf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

joblib.dump({"model":clf, "label_encoder":le, "features":features}, os.path.join(ml_dir,"crop_recommender_v1.joblib"))
print("Saved artifact.")
