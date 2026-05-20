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
    df = pd.DataFrame(pd.read_csv(features_csv_path))
    
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
    print("Computing SHAP values on test set using TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled_df)
    
    # In SHAP, shap_values can be a numpy array of shape (num_samples, num_features, num_classes)
    # or a list of length num_classes (one per class). Let's convert list to numpy array for uniformity.
    if isinstance(shap_values, list):
        shap_values_arr = np.stack(shap_values, axis=-1)
    else:
        shap_values_arr = shap_values
        
    print(f"SHAP values shape: {shap_values_arr.shape}")
    
    # Compute mean |SHAP| across all samples and classes
    # Axis 0: samples, Axis 2: classes
    mean_abs_shap = np.mean(np.abs(shap_values_arr), axis=(0, 2))
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    }).sort_values(by='mean_abs_shap', ascending=False)
    
    print("\nTop 15 features by mean |SHAP|:")
    top_15_features = []
    for rank, (idx, row) in enumerate(importance_df.head(15).iterrows(), 1):
        feat_name = row['feature']
        feat_val = row['mean_abs_shap']
        print(f"{rank:2d}. {feat_name}: {feat_val:.6f}")
        top_15_features.append(feat_name)
        
    # Save SHAP Summary Plot (global feature importance beeswarm)
    # Since beeswarm is class-specific, let's plot it for the 'Healthy' class (class index 2)
    # or let's use the explanation object.
    # We will generate a beeswarm plot for the 'Healthy' class (or class index 2)
    # and label it clearly.
    healthy_idx = class_to_idx.get("Healthy", 2)
    print(f"\nGenerating SHAP beeswarm plot for class 'Healthy' (index {healthy_idx})...")
    
    plt.figure(figsize=(10, 8))
    # We slice shap_values_arr for healthy_idx
    shap.summary_plot(shap_values_arr[:, :, healthy_idx], X_test_scaled_df, show=False)
    plt.title(f"SHAP Beeswarm Plot (Class: Healthy)", fontsize=14, pad=15)
    plt.tight_layout()
    shap_summary_path = outputs_dir / "shap_summary.png"
    plt.savefig(shap_summary_path, dpi=150)
    plt.close()
    print(f"Saved SHAP summary beeswarm plot to {shap_summary_path}")

if __name__ == "__main__":
    main()
