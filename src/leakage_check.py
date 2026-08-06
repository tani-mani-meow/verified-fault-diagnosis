import os
import sys
import re
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

def parse_reading_idx(file_path):
    name = Path(file_path).name
    # Match preprocess_Reading<N>.wav
    match = re.search(r'preprocess_Reading(\d+)\.wav', name)
    if match:
        return int(match.group(1))
    return 0

def main():
    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent
    features_csv_path = repo_dir / "outputs" / "features.csv"
    outputs_dir = repo_dir / "outputs"
    
    if not features_csv_path.exists():
        print(f"Error: features.csv not found at {features_csv_path}. Run train_baseline.py first.")
        sys.exit(1)
        
    # Load features matrix
    df = pd.read_csv(features_csv_path)
    print(f"Loaded feature matrix of shape: {df.shape}")
    
    # Add reading_index column
    df['reading_index'] = df['file_path'].apply(parse_reading_idx)
    
    # ------------------ 1. ADJACENT-SIMILARITY TEST ------------------
    print("\n--- Running Adjacent-Similarity Test ---")
    
    feature_cols = [c for c in df.columns if c not in ['class', 'file_path', 'reading_index']]
    
    # Standardize features for cosine similarity computation
    scaler = StandardScaler()
    scaled_feats = scaler.fit_transform(df[feature_cols])
    
    scaled_df = pd.DataFrame(scaled_feats, columns=feature_cols)
    scaled_df['class'] = df['class']
    scaled_df['reading_index'] = df['reading_index']
    
    classes = sorted(df['class'].unique())
    num_classes = len(classes)
    sim_report_lines = []
    sim_report_lines.append("Class,Mean Adjacent Sim,Mean Random Sim,Ratio")
    
    print(f"{'Class':12s} | {'Mean Adjacent Sim':18s} | {'Mean Random Sim':15s} | {'Ratio':8s}")
    print("-" * 63)
    
    np.random.seed(42)
    
    class_results = []
    
    for c in classes:
        class_df = scaled_df[scaled_df['class'] == c].sort_values(by='reading_index')
        X_c = class_df[feature_cols].values
        
        # Adjacent pairs (N and N+1)
        # Compute cosine similarity
        A = X_c[:-1]
        B = X_c[1:]
        dot_products = np.sum(A * B, axis=1)
        norms_A = np.linalg.norm(A, axis=1)
        norms_B = np.linalg.norm(B, axis=1)
        adjacent_sims = dot_products / (norms_A * norms_B + 1e-12)
        mean_adj = np.mean(adjacent_sims)
        
        # Random within-class pairs
        random_sims = []
        n_samples = len(X_c)
        for _ in range(1000):
            i, j = np.random.choice(n_samples, size=2, replace=False)
            u = X_c[i]
            v = X_c[j]
            sim = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
            random_sims.append(sim)
        mean_rand = np.mean(random_sims)
        
        ratio = mean_adj / mean_rand
        
        print(f"{c:12s} | {mean_adj:18.6f} | {mean_rand:15.6f} | {ratio:8.4f}")
        sim_report_lines.append(f"{c},{mean_adj:.6f},{mean_rand:.6f},{ratio:.4f}")
        class_results.append((c, mean_adj, mean_rand, ratio))
        
    # ------------------ 2. CONTIGUOUS-SPLIT STRESS TEST ------------------
    print("\n--- Running Contiguous-Split Stress Test ---")
    
    # Sort the full dataframe by class and reading_index
    df_sorted = df.sort_values(by=['class', 'reading_index']).copy()
    
    # Split: first 80% (180 samples) per class -> train, last 20% (45 samples) -> test
    train_indices = []
    test_indices = []
    for c in classes:
        class_idx = df_sorted[df_sorted['class'] == c].index.tolist()
        split_point = int(len(class_idx) * 0.8)
        train_indices.extend(class_idx[:split_point])
        test_indices.extend(class_idx[split_point:])
        
    train_df = df_sorted.loc[train_indices]
    test_df = df_sorted.loc[test_indices]
    
    print(f"Contiguous Train size: {len(train_df)} | Test size: {len(test_df)}")
    
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    y_train = train_df['class'].map(class_to_idx).values
    y_test = test_df['class'].map(class_to_idx).values
    
    # Standardize features (fit scaler on train only)
    scaler_split = StandardScaler()
    X_train_scaled = scaler_split.fit_transform(X_train)
    X_test_scaled = scaler_split.transform(X_test)
    
    # Train the SAME XGBoost config
    clf = XGBClassifier(
        objective='multi:softprob',
        num_class=num_classes,
        random_state=42,
        eval_metric='mlogloss'
    )
    clf.fit(X_train_scaled, y_train)
    
    y_pred = clf.predict(X_test_scaled)
    contiguous_acc = accuracy_score(y_test, y_pred)
    contiguous_f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"Contiguous-Split Accuracy: {contiguous_acc:.6f}")
    print(f"Contiguous-Split Macro-F1: {contiguous_f1:.6f}")
    
    # 3. GroupKFold (n_splits=5)
    # group_id = (reading_index // 5) within class
    # To keep groups unique across classes:
    groups = df_sorted['class'] + "_" + (df_sorted['reading_index'] // 5).astype(str)
    
    X_full = df_sorted[feature_cols]
    y_full = df_sorted['class'].map(class_to_idx).values
    
    gkf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', XGBClassifier(
            objective='multi:softprob',
            num_class=num_classes,
            random_state=42,
            eval_metric='mlogloss'
        ))
    ])
    
    print("Running 5-fold GroupKFold CV...")
    gkf = GroupKFold(n_splits=5)
    gkf_scores = cross_val_score(gkf_pipeline, X_full, y_full, cv=gkf, groups=groups, scoring='accuracy', n_jobs=-1)
    gkf_mean = np.mean(gkf_scores)
    gkf_std = np.std(gkf_scores)
    
    print(f"GroupKFold CV Accuracy: {gkf_mean:.6f} +/- {gkf_std:.6f}")
    
    # Verdict logic - parse baseline accuracy from saved report
    baseline_report_path = repo_dir / "outputs" / "baseline_report.txt"
    baseline_acc = 0.955556  # fallback default
    if baseline_report_path.exists():
        try:
            with open(baseline_report_path) as f:
                for line in f:
                    if "Accuracy:" in line:
                        baseline_acc = float(line.split(":")[1].strip())
                        break
        except Exception:
            print(f"Warning: could not parse baseline accuracy from {baseline_report_path}, using fallback")
    else:
        print(f"Warning: baseline report not found at {baseline_report_path}, using fallback baseline_acc = {baseline_acc}")
    # LOW LEAKAGE if contiguous/grouped accuracy is within ~2 points (0.02) of baseline
    # LEAKAGE DETECTED if it drops materially (>~5 points)
    if contiguous_acc >= (baseline_acc - 0.02):
        verdict = "LOW LEAKAGE"
    elif contiguous_acc < (baseline_acc - 0.05):
        verdict = "LEAKAGE DETECTED"
    else:
        verdict = "MODERATE LEAKAGE"
        
    print(f"\nVerdict: {verdict}")
    
    # Save report
    report_lines = []
    report_lines.append("=== Leakage Diagnostic Report ===")
    report_lines.append("\nAdjacent vs Random Cosine Similarity:")
    report_lines.append(f"{'Class':12s} | {'Mean Adjacent Sim':18s} | {'Mean Random Sim':15s} | {'Ratio':8s}")
    report_lines.append("-" * 63)
    for c, mean_adj, mean_rand, ratio in class_results:
        report_lines.append(f"{c:12s} | {mean_adj:18.6f} | {mean_rand:15.6f} | {ratio:8.4f}")
        
    report_lines.append("\nContiguous-Split Results:")
    report_lines.append(f"Contiguous-Split Accuracy: {contiguous_acc:.6f}")
    report_lines.append(f"Contiguous-Split Macro-F1: {contiguous_f1:.6f}")
    
    report_lines.append("\nGroupKFold Results:")
    report_lines.append(f"GroupKFold 5-Fold Accuracy: {gkf_mean:.6f} +/- {gkf_std:.6f}")
    
    report_lines.append(f"\nVERDICT: {verdict}")
    
    report_content = "\n".join(report_lines)
    
    report_path = outputs_dir / "leakage_report.txt"
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"\nSaved leakage report to {report_path}")

if __name__ == "__main__":
    main()
