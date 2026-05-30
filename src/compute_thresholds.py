import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent
    features_csv = repo_dir / "outputs" / "features.csv"
    output_json = repo_dir / "outputs" / "feature_thresholds.json"
    
    if not features_csv.exists():
        print(f"Error: features.csv not found at {features_csv}.")
        sys.exit(1)
        
    df = pd.read_csv(features_csv)
    print(f"Loaded features matrix of shape {df.shape}")
    
    # Stratified 80/20 train/test split, same as train_baseline.py
    # Columns 'class' and 'file_path' are excluded from feature calculation
    train_df, _ = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['class']
    )
    print(f"Using {len(train_df)} training rows for threshold calculations.")
    
    feature_cols = [c for c in df.columns if c not in ['class', 'file_path']]
    classes = sorted(df['class'].unique())
    
    # Initialize JSON structure
    thresholds = {
        "global": {},
        "classes": {},
        "distinctive": {}
    }
    
    # 1. Global percentiles
    for col in feature_cols:
        col_data = train_df[col].dropna()
        thresholds["global"][col] = {
            "p10": float(np.percentile(col_data, 10)),
            "p25": float(np.percentile(col_data, 25)),
            "p50": float(np.percentile(col_data, 50)),
            "p75": float(np.percentile(col_data, 75)),
            "p90": float(np.percentile(col_data, 90))
        }
        
    # 2. Per-class percentiles
    for c in classes:
        thresholds["classes"][c] = {}
        class_df = train_df[train_df['class'] == c]
        for col in feature_cols:
            col_data = class_df[col].dropna()
            thresholds["classes"][c][col] = {
                "p10": float(np.percentile(col_data, 10)) if len(col_data) > 0 else 0.0,
                "p25": float(np.percentile(col_data, 25)) if len(col_data) > 0 else 0.0,
                "p50": float(np.percentile(col_data, 50)) if len(col_data) > 0 else 0.0,
                "p75": float(np.percentile(col_data, 75)) if len(col_data) > 0 else 0.0,
                "p90": float(np.percentile(col_data, 90)) if len(col_data) > 0 else 0.0
            }
            
    # 3. Distinctive features
    # Median is in the top-2 or bottom-2 medians across all 8 classes for that feature
    distinctive_top = {c: [] for c in classes}
    distinctive_bottom = {c: [] for c in classes}
    
    for col in feature_cols:
        # Collect medians for each class
        class_medians = []
        for c in classes:
            class_df = train_df[train_df['class'] == c]
            median_val = float(class_df[col].median()) if len(class_df) > 0 else 0.0
            class_medians.append((c, median_val))
            
        # Sort by median value
        class_medians.sort(key=lambda x: x[1])
        
        # Bottom 2 classes
        bottom_classes = [class_medians[0][0], class_medians[1][0]]
        # Top 2 classes
        top_classes = [class_medians[-2][0], class_medians[-1][0]]
        
        for c in bottom_classes:
            distinctive_bottom[c].append(col)
        for c in top_classes:
            distinctive_top[c].append(col)
            
    for c in classes:
        thresholds["distinctive"][c] = {
            "top_2": distinctive_top[c],
            "bottom_2": distinctive_bottom[c]
        }
        
    # Write to outputs/feature_thresholds.json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=4)
        
    print(f"Successfully computed thresholds and saved to {output_json}")
    
    # Print short summary
    print("\nDistinctive Features Summary (Top 2 count / Bottom 2 count):")
    for c in classes:
        top_cnt = len(thresholds["distinctive"][c]["top_2"])
        bot_cnt = len(thresholds["distinctive"][c]["bottom_2"])
        print(f"  Class {c:12s}: {top_cnt:2d} distinctively high, {bot_cnt:2d} distinctively low")

if __name__ == "__main__":
    main()
