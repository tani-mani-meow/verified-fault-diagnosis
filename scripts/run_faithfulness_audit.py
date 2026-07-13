# scripts/run_faithfulness_audit.py
import os
import sys
import json
import time
import queue
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Ensure we can import from src
script_dir = Path(__file__).resolve().parent
repo_dir = script_dir.parent
sys.path.insert(0, str(repo_dir / "src"))

from pipeline import run_pipeline
import llm_client

# Mutex to protect writes to audit_results and the JSON file
write_lock = threading.Lock()
audit_results = []
cases_run = {}

def audit_case(idx, original_idx, row):
    true_class = row['class']
    print(f"[{idx+1}/100] Auditing Index {original_idx} (True Class: {true_class})...")
    
    sample_dict = row.to_dict()
    
    # Call pipeline and measure latency
    start_time = time.time()
    try:
        report = run_pipeline(sample_dict, true_label=true_class, mock=False)
        latency = time.time() - start_time
        
        # Extract thread-local token usage
        usage = llm_client.get_last_usage()
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        
        # Parse fields
        pred_class = report["predicted"]
        confidence = report["confidence"]
        faithfulness = report.get("faithfulness", 0.0)
        class_consistency = report.get("class_consistency", 0.0)
        gate_decision = report["gate_decision"]
        gate_reason = report["reason"]
        explanation = report.get("explanation", "")
        warnings = report.get("mechanism_warnings", [])
        
        per_claim = report.get("per_claim", [])
        n_claims = len(report.get("claims", []))
        n_sample_true = sum(1 for pc in per_claim if pc.get("sample_true", False))
        n_atypical = sum(1 for pc in per_claim if pc.get("atypical", False))
        atypical_action_critical = any(pc.get("atypical", False) and pc.get("action_critical", False) for pc in per_claim)
        
        case_data = {
            "index": int(original_idx),
            "true_label": true_class,
            "predicted": pred_class,
            "confidence": confidence,
            "correct": bool(pred_class == true_class),
            "faithfulness": faithfulness,
            "class_consistency": class_consistency,
            "n_claims": n_claims,
            "n_sample_true": n_sample_true,
            "n_atypical": n_atypical,
            "atypical_action_critical": atypical_action_critical,
            "mechanism_warnings": warnings,
            "gate_decision": gate_decision,
            "gate_reason": gate_reason,
            "explanation": explanation,
            "per_claim": per_claim,
            "latency_s": latency,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        }
        
        # Thread-safe append and save
        with write_lock:
            audit_results.append(case_data)
            cases_run[int(original_idx)] = case_data
            audit_file = repo_dir / "outputs" / "faithfulness_audit.json"
            with open(audit_file, "w", encoding="utf-8") as f:
                json.dump(audit_results, f, indent=2)
                
        print(f"    Index {original_idx} DONE | Pred: {pred_class} | Correct: {case_data['correct']} | Gate: {gate_decision} | Latency: {latency:.2f}s")
        return True
    except Exception as e:
        print(f"Error auditing index {original_idx}: {e}")
        return False

def main():
    print("=== STARTING CERTIFICATE 3 FAITHFULNESS AUDIT (CONCURRENT/RESUME) ===")
    
    # 1. Load and split features.csv to match training
    features_csv = repo_dir / "outputs" / "features.csv"
    if not features_csv.exists():
        print(f"Error: features.csv not found at {features_csv}. Run training first.")
        sys.exit(1)
        
    df = pd.read_csv(features_csv)
    y_str = df['class']
    class_names = sorted(y_str.unique())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    y = y_str.map(class_to_idx).values
    
    # 80/20 train/test split, random_state=42
    _, df_test, _, _ = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Select only the 7 fault classes (exclude Healthy)
    fault_test = df_test[df_test['class'] != 'Healthy']
    
    # Stratified-sample exactly 100 cases
    audit_df, _ = train_test_split(
        fault_test, train_size=100, stratify=fault_test['class'], random_state=42
    )
    
    print(f"Selected {len(audit_df)} test rows for stratified audit.")
    print("Per-class counts:")
    print(audit_df['class'].value_counts())
    
    # Load existing cases if present (to allow resumes / incremental appends)
    audit_file = repo_dir / "outputs" / "faithfulness_audit.json"
    
    global audit_results, cases_run
    if audit_file.exists():
        try:
            with open(audit_file, "r", encoding="utf-8") as f:
                loaded_cases = json.load(f)
            # Filter out duplicates and ensure validity
            seen_indices = set()
            for c in loaded_cases:
                idx_val = int(c["index"])
                if idx_val not in seen_indices:
                    seen_indices.add(idx_val)
                    audit_results.append(c)
                    cases_run[idx_val] = c
            print(f"Loaded {len(audit_results)} existing audited cases from {audit_file}.")
        except Exception as e:
            print(f"Could not parse existing audit file ({e}); starting fresh.")
            audit_results = []
            cases_run = {}
            
    # Setup work queue for parallel execution with automatic retries
    q = queue.Queue()
    for idx, (original_idx, row) in enumerate(audit_df.iterrows()):
        if int(original_idx) not in cases_run:
            q.put((idx, original_idx, row))
            
    print(f"Queue size for execution: {q.qsize()} cases remaining.")
    
    start_audit_time = time.time()
    
    # Worker function for ThreadPool
    def worker():
        while True:
            try:
                item = q.get_nowait()
            except queue.Empty:
                break
            idx, original_idx, row = item
            success = audit_case(idx, original_idx, row)
            if not success:
                print(f"Index {original_idx} failed. Putting back in queue to retry.")
                q.put(item)
                time.sleep(5.0) # sleep slightly longer after a failure
            else:
                time.sleep(1.0) # pacing pacing pacing
            q.task_done()
            
    # Run with 3 workers
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for _ in range(3):
            futures.append(executor.submit(worker))
        q.join()
        
    print(f"\nAll 100 cases completed in {time.time() - start_audit_time:.2f}s!")
    
    print("\n=== AGGREGATING RESULTS ===")
    
    # 2. Aggregations & Calculations
    df_audit = pd.DataFrame(audit_results)
    
    # Faithfulness Stats
    f_mean = float(df_audit["faithfulness"].mean())
    f_median = float(df_audit["faithfulness"].median())
    f_std = float(df_audit["faithfulness"].std()) if len(df_audit) > 1 else 0.0
    f_min = float(df_audit["faithfulness"].min())
    
    # Histogram bins: 0, 0.2, 0.4, 0.6, 0.8, 1.0
    f_hist, _ = np.histogram(df_audit["faithfulness"], bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.001])
    f_hist_list = [int(x) for x in f_hist]
    
    # Gate outcomes
    total_cases = len(df_audit)
    release_pct = float((df_audit["gate_decision"] == "RELEASE").sum() / total_cases * 100)
    escalate_pct = float((df_audit["gate_decision"] == "ESCALATE").sum() / total_cases * 100)
    
    reason_counts = df_audit[df_audit["gate_decision"] == "ESCALATE"]["gate_reason"].value_counts().to_dict()
    reason_pct = {k: float(v / total_cases * 100) for k, v in reason_counts.items()}
    
    # SAFETY CROSS-TAB counts
    correct_release = int(((df_audit["gate_decision"] == "RELEASE") & (df_audit["correct"])).sum())
    correct_escalate = int(((df_audit["gate_decision"] == "ESCALATE") & (df_audit["correct"])).sum())
    misclass_release = int(((df_audit["gate_decision"] == "RELEASE") & (~df_audit["correct"])).sum())
    misclass_escalate = int(((df_audit["gate_decision"] == "ESCALATE") & (~df_audit["correct"])).sum())
    
    # Precision and Catch Rate
    misclass_total = misclass_release + misclass_escalate
    safety_catch_rate = float(misclass_escalate / misclass_total * 100) if misclass_total > 0 else 100.0
    
    released_total = correct_release + misclass_release
    release_precision = float(correct_release / released_total * 100) if released_total > 0 else 100.0
    
    # Per-class table
    class_stats = {}
    for name, group in df_audit.groupby("true_label"):
        class_stats[name] = {
            "mean_faithfulness": float(group["faithfulness"].mean()),
            "release_rate": float((group["gate_decision"] == "RELEASE").sum() / len(group)),
            "escalate_rate": float((group["gate_decision"] == "ESCALATE").sum() / len(group)),
            "mean_class_consistency": float(group["class_consistency"].mean())
        }
        
    # Warnings rate
    warning_cases = int(df_audit["mechanism_warnings"].apply(lambda w: len(w) > 0).sum())
    warning_rate = float(warning_cases / total_cases * 100)
    
    # Cost & Latency
    latency_mean = float(df_audit["latency_s"].mean())
    latency_median = float(df_audit["latency_s"].median())
    latency_p95 = float(df_audit["latency_s"].quantile(0.95))
    
    mean_prompt_tokens = float(df_audit["prompt_tokens"].mean())
    mean_completion_tokens = float(df_audit["completion_tokens"].mean())
    total_tokens = int(df_audit["prompt_tokens"].sum() + df_audit["completion_tokens"].sum())
    
    summary = {
        "faithfulness": {
            "mean": f_mean,
            "median": f_median,
            "std": f_std,
            "min": f_min,
            "histogram_bins_0_to_1": f_hist_list
        },
        "gate_outcomes": {
            "release_pct": release_pct,
            "escalate_pct": escalate_pct,
            "escalate_reason_pcts": reason_pct
        },
        "safety_cross_tab": {
            "correct_released": correct_release,
            "correct_escalated": correct_escalate,
            "misclassified_released": misclass_release,
            "misclassified_escalated": misclass_escalate,
            "safety_catch_rate_pct": safety_catch_rate,
            "release_precision_pct": release_precision
        },
        "per_class": class_stats,
        "mechanism_warnings_rate_pct": warning_rate,
        "cost_latency": {
            "mean_latency_s": latency_mean,
            "median_latency_s": latency_median,
            "p95_latency_s": latency_p95,
            "mean_prompt_tokens": mean_prompt_tokens,
            "mean_completion_tokens": mean_completion_tokens,
            "total_tokens": total_tokens
        }
    }
    
    summary_file = repo_dir / "outputs" / "faithfulness_audit_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Saved summary file to {summary_file}")
    
    # 3. Plot Figures
    # Figure 1: Faithfulness histogram
    plt.figure(figsize=(6, 4))
    plt.bar(["[0-0.2)", "[0.2-0.4)", "[0.4-0.6)", "[0.6-0.8)", "[0.8-1.0]"], f_hist, color="steelblue", edgecolor="black")
    plt.title("Explanation Faithfulness Distribution (Experiment 5)")
    plt.xlabel("Faithfulness Score")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig(repo_dir / "outputs" / "audit_faithfulness_hist.png", dpi=150)
    plt.close()
    
    # Figure 2: Release rate by class
    classes_sorted = sorted(class_stats.keys())
    release_rates = [class_stats[c]["release_rate"] * 100 for c in classes_sorted]
    plt.figure(figsize=(8, 4))
    plt.bar(classes_sorted, release_rates, color="forestgreen", edgecolor="black")
    plt.title("Explanation Gate Release Rate by Fault Class")
    plt.xlabel("True Fault Class")
    plt.ylabel("Release Rate (%)")
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(repo_dir / "outputs" / "audit_release_rate_by_class.png", dpi=150)
    plt.close()
    
    # Figure 3: Safety cross-tab gate vs correctness
    plt.figure(figsize=(6, 4))
    categories = ['Correct\n(N={})'.format(correct_release + correct_escalate), 'Misclassified\n(N={})'.format(misclass_release + misclass_escalate)]
    released_vals = [correct_release, misclass_release]
    escalated_vals = [correct_escalate, misclass_escalate]
    
    bar_width = 0.5
    r = np.arange(len(categories))
    plt.bar(r, released_vals, color='mediumseagreen', edgecolor='black', width=bar_width, label='Released')
    plt.bar(r, escalated_vals, bottom=released_vals, color='crimson', edgecolor='black', width=bar_width, label='Escalated')
    
    plt.title("Trust Gate Decisions vs Diagnostic Correctness")
    plt.xticks(r, categories)
    plt.ylabel("Count")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(repo_dir / "outputs" / "audit_gate_vs_correctness.png", dpi=150)
    plt.close()
    
    print("=== AUDIT PLOTS COMPLETED AND SAVED IN outputs/ ===")

if __name__ == "__main__":
    main()
