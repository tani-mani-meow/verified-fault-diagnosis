import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import joblib
from tqdm import tqdm

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# Import extract_features from features.py
from features import extract_features

def main():
    # Resolve paths relative to this script
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent
    data_dir = repo_dir / "data" / "AirCompressorDataset"
    outputs_dir = repo_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    
    if not data_dir.exists():
        print(f"Error: Dataset directory {data_dir} does not exist.")
        sys.exit(1)
        
    class_folders = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    print("Starting feature extraction for all 1800 files...")
    rows = []
    
    # Iterate all files with tqdm progress bar
    for folder in class_folders:
        class_name = folder.name
        files = sorted(list(folder.glob("*.wav")))
        print(f"Extracting features for class: {class_name} ({len(files)} files)...")
        for f in tqdm(files):
            try:
                y, sr = librosa.load(str(f), sr=None)
                feats = extract_features(y, sr)
                
                row = {
                    'class': class_name,
                    'file_path': f.relative_to(data_dir).as_posix()
                }
                # Update with features (maintaining order)
                row.update(feats)
                rows.append(row)
            except Exception as e:
                print(f"\nError processing file {f.name}: {e}")
                sys.exit(1)
                
    df = pd.DataFrame(rows)
    
    # Save the full matrix to outputs/features.csv
    features_csv_path = outputs_dir / "features.csv"
    df.to_csv(features_csv_path, index=False)
    print(f"Saved full feature matrix to {features_csv_path} (Shape: {df.shape})")
    
    # Separate features and targets
    X = df.drop(columns=['class', 'file_path'])
    y_str = df['class']
    
    # Map class labels to integers alphabetically
    class_names = sorted(y_str.unique())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    y = y_str.map(class_to_idx).values
    
    # Stratified train/test split 80/20, random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize features (fit scaler on train only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    scaler_path = outputs_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")
    
    # Train XGBoost model (multi:softprob, 8 classes, random_state=42)
    print("Training XGBoost classifier...")
    clf = XGBClassifier(
        objective='multi:softprob',
        num_class=8,
        random_state=42,
        eval_metric='mlogloss'
    )
    clf.fit(X_train_scaled, y_train)
    
    # Save model
    model_path = outputs_dir / "xgb_model.joblib"
    joblib.dump(clf, model_path)
    print(f"Saved XGBoost model to {model_path}")
    
    # Save feature names list
    feature_names = list(X.columns)
    feature_names_path = outputs_dir / "feature_names.json"
    with open(feature_names_path, "w") as f:
        json.dump(feature_names, f, indent=4)
    print(f"Saved feature names list to {feature_names_path}")
    
    # Evaluate model
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    class_report = classification_report(y_test, y_pred, target_names=class_names, digits=6)
    
    # Build CV Pipeline
    cv_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', XGBClassifier(
            objective='multi:softprob',
            num_class=8,
            random_state=42,
            eval_metric='mlogloss'
        ))
    ])
    
    # Run 5-fold stratified cross-validation on full dataset
    print("Running 5-fold Stratified Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(cv_pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    print(f"CV Accuracy: {cv_mean:.6f} +/- {cv_std:.6f}")
    
    # Format and save report
    report_lines = []
    report_lines.append("=== Baseline XGBoost Model Report ===")
    report_lines.append(f"Overall Accuracy: {acc:.6f}")
    report_lines.append(f"Macro-F1 Score: {macro_f1:.6f}")
    report_lines.append(f"5-Fold CV Accuracy: {cv_mean:.6f} +/- {cv_std:.6f}")
    report_lines.append("\nPer-class Metrics (Classification Report):")
    report_lines.append(class_report)
    
    report_content = "\n".join(report_lines)
    
    report_path = outputs_dir / "baseline_report.txt"
    with open(report_path, "w") as f:
        f.write(report_content)
    print("\n" + report_content)
    print(f"Saved report to {report_path}")
    
    # Save confusion matrix plot
    print("Plotting confusion matrix...")
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=class_names, cmap='Blues', ax=ax, xticks_rotation=45
    )
    plt.title("Baseline XGBoost Confusion Matrix (Test Set)")
    plt.tight_layout()
    cm_path = outputs_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_path}")

if __name__ == "__main__":
    main()
