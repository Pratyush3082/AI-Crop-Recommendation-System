# scripts/hyperparam_search.py
import os, joblib, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

BASE = os.path.dirname(os.path.dirname(__file__))
df = pd.read_csv(os.path.join(BASE,"data","crop_data.csv")).dropna(subset=["crop"])
features = ["N","P","K","temperature","humidity","ph","rainfall"]
X = df[features].fillna(df[features].median())
y = LabelEncoder().fit_transform(df["crop"])

param_dist = {
    "n_estimators":[200,400,800],
    "max_depth":[None,8,12,20],
    "min_samples_split":[2,5,10],
    "min_samples_leaf":[1,2,4],
    "class_weight":[None,"balanced"]
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=20, cv=cv, scoring="accuracy", n_jobs=-1, verbose=2, random_state=42)
search.fit(X,y)
print("Best params:", search.best_params_)
best = search.best_estimator_
joblib.dump({"model":best, "label_encoder":LabelEncoder().fit(df["crop"]), "features":features}, os.path.join(BASE, "recommender","ml","crop_recommender_v1.joblib"))
