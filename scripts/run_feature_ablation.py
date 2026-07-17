# scripts/run_feature_ablation.py
"""Feature-family x classifier ablation under identical 5-fold CV."""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

repo = Path(__file__).resolve().parent.parent
df = pd.read_csv(repo / "outputs" / "features.csv")
le = LabelEncoder()
y = le.fit_transform(df["class"])
all_feats = [c for c in df.columns if c not in {"class", "file_path"}]

# Categorize features by family
time_feats = [c for c in all_feats if c in {"rms","kurtosis","skewness","crest_factor","zero_crossing_rate","peak_to_peak","shannon_entropy"}]
freq_feats = [c for c in all_feats if c.startswith("spectral_") or c.startswith("band_energy_ratio_") or c == "dominant_frequency"]
wpt_feats = [c for c in all_feats if c.startswith("wpt_energy_")]
families = {"time": time_feats, "freq": freq_feats, "wpt": wpt_feats, "full": all_feats}

classifiers = {
    "XGBoost": XGBClassifier(random_state=42, eval_metric="mlogloss"),
    "SVM": SVC(random_state=42),
    "RF": RandomForestClassifier(random_state=42, n_estimators=100),
    "LogReg": LogisticRegression(random_state=42, max_iter=2000),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for fam, feats in families.items():
    X = df[feats].values
    for clf_name, clf in classifiers.items():
        scores = cross_val_score(clf, X, y, cv=skf, scoring="f1_macro")
        key = f"{fam}_{clf_name}"
        results[key] = {
            "family": fam, 
            "classifier": clf_name, 
            "n_features": len(feats),
            "mean_f1": float(scores.mean()), 
            "std_f1": float(scores.std()),
            "scores": [float(s) for s in scores]
        }
        print(f"{fam:>6} x {clf_name:>8} ({len(feats):>2} feats): {scores.mean():.4f} +/- {scores.std():.4f}")

with open(repo / "outputs" / "feature_ablation.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to outputs/feature_ablation.json")
