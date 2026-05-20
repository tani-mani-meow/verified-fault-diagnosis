import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent
    outputs_dir = repo_dir / "outputs"
    
    features_csv = outputs_dir / "features.csv"
    xgb_model_file = outputs_dir / "xgb_model.joblib"
    scaler_file = outputs_dir / "scaler.joblib"
    feature_names_file = outputs_dir / "feature_names.json"
    confusion_matrix_img = outputs_dir / "confusion_matrix.png"
    shap_summary_img = outputs_dir / "shap_summary.png"
    baseline_report_file = outputs_dir / "baseline_report.txt"
    
    all_passed = True
    
    # 1. outputs/features.csv exists with 1800 rows and the expected feature columns
    if features_csv.exists():
        try:
            df = pd.read_csv(features_csv)
            # Check row count
            if len(df) == 1800:
                # Check for class and file_path and other features
                if 'class' in df.columns and 'file_path' in df.columns and len(df.columns) > 80:
                    print(f"outputs/features.csv exists and is valid (shape {df.shape}): PASS")
                else:
                    print(f"outputs/features.csv columns are incorrect (found {len(df.columns)} columns): FAIL")
                    all_passed = False
            else:
                print(f"outputs/features.csv row count is incorrect (found {len(df)} rows, expected 1800): FAIL")
                all_passed = False
        except Exception as e:
            print(f"outputs/features.csv failed to load: {e} : FAIL")
            all_passed = False
    else:
        print("outputs/features.csv does not exist: FAIL")
        all_passed = False
        
    # 2. outputs/xgb_model.joblib, outputs/scaler.joblib, outputs/feature_names.json exist
    files_check = {
        "outputs/xgb_model.joblib": xgb_model_file,
        "outputs/scaler.joblib": scaler_file,
        "outputs/feature_names.json": feature_names_file
    }
    
    for label, path in files_check.items():
        if path.exists() and path.stat().st_size > 0:
            print(f"{label} exists and is valid: PASS")
        else:
            print(f"{label} is missing or empty: FAIL")
            all_passed = False
            
    # 3. overall test accuracy >= 0.95
    # Let's verify by loading the model/scaler/features and evaluating
    if features_csv.exists() and xgb_model_file.exists() and scaler_file.exists() and feature_names_file.exists():
        try:
            # Recreate test split and evaluate
            df = pd.read_csv(features_csv)
            X = df.drop(columns=['class', 'file_path'])
            y_str = df['class']
            class_names = sorted(y_str.unique())
            class_to_idx = {name: idx for idx, name in enumerate(class_names)}
            y = y_str.map(class_to_idx).values
            
            _, X_test, _, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            scaler = joblib.load(scaler_file)
            X_test_scaled = scaler.transform(X_test)
            
            model = joblib.load(xgb_model_file)
            y_pred = model.predict(X_test_scaled)
            
            acc = accuracy_score(y_test, y_pred)
            
            if acc >= 0.95:
                print(f"overall test accuracy >= 0.95 (actual: {acc:.6f}): PASS")
            else:
                print(f"overall test accuracy >= 0.95 (actual: {acc:.6f}): FAIL")
                all_passed = False
        except Exception as e:
            print(f"Error evaluating test accuracy: {e} : FAIL")
            all_passed = False
    else:
        print("overall test accuracy check skipped due to missing files: FAIL")
        all_passed = False
        
    # 4. outputs/confusion_matrix.png and outputs/shap_summary.png exist
    plots_check = {
        "outputs/confusion_matrix.png": confusion_matrix_img,
        "outputs/shap_summary.png": shap_summary_img
    }
    
    for label, path in plots_check.items():
        if path.exists() and path.stat().st_size > 0:
            print(f"{label} exists and is valid: PASS")
        else:
            print(f"{label} is missing or empty: FAIL")
            all_passed = False
            
    if all_passed:
        print("\nAll Phase 1 verification checks passed.")
        sys.exit(0)
    else:
        print("\nPhase 1 verification checks failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
