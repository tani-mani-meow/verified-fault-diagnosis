# src/explain_shap.py
import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def main():
    # Resolve paths relative to this script
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent
    features_csv_path = repo_dir / "outputs" / "features.csv"
    model_path = repo_dir / "outputs" / "xgb_model.joblib"
    scaler_path = repo_dir / "outputs" / "scaler.joblib"
    feature_names_path = repo_dir / "outputs" / "feature_names.json"
    outputs_dir = repo_dir / "outputs"
    
    if not (features_csv_path.exists() and model_path.exists() and scaler_path.exists() and feature_names_path.exists()):
        print("Error: Missing required training outputs. Run train_baseline.py first.")
        sys.exit(1)
        
    # Load feature names
    with open(feature_names_path, "r") as f:
        feature_names = json.load(f)
        
    # Load features matrix
    df = pd.read_csv(features_csv_path)
    
    X = df.drop(columns=['class', 'file_path'])
    y_str = df['class']
    
    # Map class labels to integers alphabetically
    class_names = sorted(y_str.unique())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    y = y_str.map(class_to_idx).values
    
    # Stratified train/test split (80/20) - identical to train_baseline.py
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Load scaler and standardize X_test
    scaler = joblib.load(scaler_path)
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    # Load model
    model = joblib.load(model_path)
    
    # Initialize TreeExplainer
    print("Computing SHAP values using TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    
    # We will generate per-class SHAP analysis for all 7 fault classes
    fault_classes = ["LIV", "LOV", "NRV", "Bearing", "Flywheel", "Piston", "Riderbelt"]
    shap_results = {}
    
    for classname in fault_classes:
        if classname not in class_to_idx:
            print(f"Warning: Class {classname} not found in dataset classes.")
            continue
            
        class_idx = class_to_idx[classname]
        
        # Filter test set to only rows of this true class
        idx_mask = (y_test == class_idx)
        X_test_class = X_test_scaled_df[idx_mask]
        
        if len(X_test_class) == 0:
            print(f"Warning: No test samples found for class {classname}")
            continue
            
        print(f"\nComputing SHAP values for class {classname} on {len(X_test_class)} samples...")
        
        # Compute SHAP values for these samples
        shap_values = explainer.shap_values(X_test_class)
        
        # Extract target class SHAP values
        if isinstance(shap_values, list):
            shap_values_target = shap_values[class_idx]
        else:
            shap_values_target = shap_values[:, :, class_idx]
            
        # Compute mean |SHAP| across all samples of this class
        mean_abs = np.mean(np.abs(shap_values_target), axis=0)
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_abs
        }).sort_values(by='mean_abs_shap', ascending=False)
        
        # Store top-10 features info
        top_10 = []
        print(f"Top 10 features by mean |SHAP| for class {classname}:")
        for rank, (idx, row) in enumerate(importance_df.head(10).iterrows(), 1):
            feat_name = row['feature']
            feat_val = float(row['mean_abs_shap'])
            print(f"  {rank:2d}. {feat_name}: {feat_val:.6f}")
            top_10.append({
                "feature": feat_name,
                "mean_abs_shap": feat_val,
                "rank": rank
            })
            
        shap_results[classname] = top_10
        
        # Save summary beeswarm plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_target, X_test_class, show=False)
        plt.title(f"SHAP Summary Plot (Class: {classname})", fontsize=14, pad=15)
        plt.tight_layout()
        plot_path = outputs_dir / f"shap_{classname}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved SHAP plot to {plot_path}")
        
    # Write results to JSON
    json_path = outputs_dir / "shap_per_class.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(shap_results, f, indent=2)
    print(f"\nSaved per-class SHAP analysis to {json_path}")

if __name__ == "__main__":
    main()
