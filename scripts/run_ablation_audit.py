# scripts/run_ablation_audit.py
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

# Mutex for thread-safety in Part B
write_lock = threading.Lock()
classonly_results = []
cases_run = {}

def audit_case_classonly(idx, original_idx, row):
    true_class = row['class']
    print(f"[{idx+1}/100] Ablation Auditing Index {original_idx} (True Class: {true_class})...")
    
    sample_dict = row.to_dict()
    
    start_time = time.time()
    try:
        # Run with grounding_mode="class_only"
        report = run_pipeline(sample_dict, true_label=true_class, mock=False, grounding_mode="class_only")
        latency = time.time() - start_time
        
        usage = llm_client.get_last_usage()
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        
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
        
        with write_lock:
            classonly_results.append(case_data)
            cases_run[int(original_idx)] = case_data
            audit_file = repo_dir / "outputs" / "faithfulness_audit_classonly.json"
            with open(audit_file, "w", encoding="utf-8") as f:
                json.dump(classonly_results, f, indent=2)
                
        print(f"    Index {original_idx} DONE | Pred: {pred_class} | Correct: {case_data['correct']} | Gate: {gate_decision} | Latency: {latency:.2f}s")
        return True
    except Exception as e:
        print(f"Error ablation auditing index {original_idx}: {e}")
        return False

def run_part_a():
    print("=== PART A: NO-API ANALYSIS OF THE EXISTING AUDIT ===")
    
    existing_audit_file = repo_dir / "outputs" / "faithfulness_audit.json"
    if not existing_audit_file.exists():
        print(f"Error: {existing_audit_file} not found. Run faithfulness audit first.")
        sys.exit(1)
        
    with open(existing_audit_file, "r", encoding="utf-8") as f:
        existing_data = json.load(f)
        
    df = pd.DataFrame(existing_data)
    
    # Split into correct vs misclassified groups
    correct_grp = df[df["correct"] == True]
    misclass_grp = df[df["correct"] == False]
    
    def get_group_stats(group_df):
        n = len(group_df)
        if n == 0:
            return {"n": 0, "mean_atypical": 0.0, "mean_claims": 0.0, "mean_faithfulness": 0.0, "distribution": {}}
        mean_atypical = float(group_df["n_atypical"].mean())
        mean_claims = float(group_df["n_claims"].mean())
        mean_faith = float(group_df["faithfulness"].mean())
        dist = group_df["n_atypical"].value_counts().to_dict()
        # Convert integer keys to string for JSON serialization
        dist = {str(k): int(v) for k, v in dist.items()}
        return {
            "n": n,
            "mean_atypical": mean_atypical,
            "mean_claims": mean_claims,
            "mean_faithfulness": mean_faith,
            "distribution": dist
        }
        
    correct_stats = get_group_stats(correct_grp)
    misclass_stats = get_group_stats(misclass_grp)
    
    # Evaluate thresholds k=1..6
    thresholds_eval = {}
    for k in range(1, 7):
        correct_count = int((correct_grp["n_atypical"] >= k).sum())
        misclass_count = int((misclass_grp["n_atypical"] >= k).sum())
        thresholds_eval[f"k={k}"] = {
            "correct_count": correct_count,
            "correct_pct": float(correct_count / len(correct_grp) * 100) if len(correct_grp) > 0 else 0.0,
            "misclassified_count": misclass_count,
            "misclassified_pct": float(misclass_count / len(misclass_grp) * 100) if len(misclass_grp) > 0 else 0.0,
        }
        
    analysis = {
        "correct_group": correct_stats,
        "misclassified_group": misclass_stats,
        "thresholds_evaluation": thresholds_eval
    }
    
    analysis_file = repo_dir / "outputs" / "atypical_analysis.json"
    with open(analysis_file, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
        
    print(json.dumps(analysis, indent=2))
    print(f"Saved analysis to {analysis_file}\n")

def run_part_b_c():
    print("=== PART B: CLASS-ONLY ABLATION AUDIT ===")
    
    # Load and split features.csv to match exactly the 100 cases
    features_csv = repo_dir / "outputs" / "features.csv"
    df = pd.read_csv(features_csv)
    y_str = df['class']
    class_names = sorted(y_str.unique())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    y = y_str.map(class_to_idx).values
    _, df_test, _, _ = train_test_split(df, y, test_size=0.2, random_state=42, stratify=y)
    fault_test = df_test[df_test['class'] != 'Healthy']
    audit_df, _ = train_test_split(fault_test, train_size=100, stratify=fault_test['class'], random_state=42)
    
    # Load existing classonly cases if present (to allow resumes)
    audit_file = repo_dir / "outputs" / "faithfulness_audit_classonly.json"
    global classonly_results, cases_run
    if audit_file.exists():
        try:
            with open(audit_file, "r", encoding="utf-8") as f:
                loaded_cases = json.load(f)
            seen_indices = set()
            for c in loaded_cases:
                idx_val = int(c["index"])
                if idx_val not in seen_indices:
                    seen_indices.add(idx_val)
                    classonly_results.append(c)
                    cases_run[idx_val] = c
            print(f"Loaded {len(classonly_results)} existing audited classonly cases from {audit_file}.")
        except Exception as e:
            print(f"Could not parse existing classonly file ({e}); starting fresh.")
            classonly_results = []
            cases_run = {}
            
    # Setup queue for execution
    q = queue.Queue()
    for idx, (original_idx, row) in enumerate(audit_df.iterrows()):
        if int(original_idx) not in cases_run:
            q.put((idx, original_idx, row))
            
    print(f"Queue size for execution: {q.qsize()} cases remaining.")
    
    start_audit_time = time.time()
    
    def worker():
        while True:
            try:
                item = q.get_nowait()
            except queue.Empty:
                break
            idx, original_idx, row = item
            success = audit_case_classonly(idx, original_idx, row)
            if not success:
                print(f"Index {original_idx} failed. Putting back in queue to retry.")
                q.put(item)
                time.sleep(5.0)
            else:
                time.sleep(1.0)
            q.task_done()
            
    if q.qsize() > 0:
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for _ in range(3):
                futures.append(executor.submit(worker))
            q.join()
            
    print(f"\nAll 100 class-only cases completed in {time.time() - start_audit_time:.2f}s!")
    
    print("\n=== AGGREGATING CLASS-ONLY SUMMARY ===")
    df_ablation = pd.DataFrame(classonly_results)
    
    # Faithfulness stats
    f_mean = float(df_ablation["faithfulness"].mean())
    f_median = float(df_ablation["faithfulness"].median())
    f_std = float(df_ablation["faithfulness"].std()) if len(df_ablation) > 1 else 0.0
    f_min = float(df_ablation["faithfulness"].min())
    
    f_hist, _ = np.histogram(df_ablation["faithfulness"], bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.001])
    f_hist_list = [int(x) for x in f_hist]
    
    # Gate outcomes
    total_cases = len(df_ablation)
    release_pct = float((df_ablation["gate_decision"] == "RELEASE").sum() / total_cases * 100)
    escalate_pct = float((df_ablation["gate_decision"] == "ESCALATE").sum() / total_cases * 100)
    
    reason_counts = df_ablation[df_ablation["gate_decision"] == "ESCALATE"]["gate_reason"].value_counts().to_dict()
    reason_pct = {k: float(v / total_cases * 100) for k, v in reason_counts.items()}
    
    # Warnings rate
    warning_cases = int(df_ablation["mechanism_warnings"].apply(lambda w: len(w) > 0).sum())
    warning_rate = float(warning_cases / total_cases * 100)
    
    # Cost & Latency
    latency_mean = float(df_ablation["latency_s"].mean())
    latency_median = float(df_ablation["latency_s"].median())
    latency_p95 = float(df_ablation["latency_s"].quantile(0.95))
    
    mean_prompt_tokens = float(df_ablation["prompt_tokens"].mean())
    mean_completion_tokens = float(df_ablation["completion_tokens"].mean())
    total_tokens = int(df_ablation["prompt_tokens"].sum() + df_ablation["completion_tokens"].sum())
    
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
    
    summary_file = repo_dir / "outputs" / "faithfulness_audit_classonly_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Saved class-only summary to {summary_file}")
    
    # Part C: Compare the two conditions
    # Load original summary
    with open(repo_dir / "outputs" / "faithfulness_audit_summary.json", "r", encoding="utf-8") as f:
        orig_summary = json.load(f)
        
    print("\n" + "="*50)
    print("        SIDE-BY-SIDE CONDITION COMPARISON")
    print("="*50)
    print(f"{'Metric':<30} | {'Sample-Grounded':<20} | {'Class-Only Ablation':<20}")
    print("-"*76)
    print(f"{'Mean Faithfulness':<30} | {orig_summary['faithfulness']['mean']:<20.4f} | {summary['faithfulness']['mean']:<20.4f}")
    print(f"{'Median Faithfulness':<30} | {orig_summary['faithfulness']['median']:<20.4f} | {summary['faithfulness']['median']:<20.4f}")
    print(f"{'Std Faithfulness':<30} | {orig_summary['faithfulness']['std']:<20.4f} | {summary['faithfulness']['std']:<20.4f}")
    print(f"{'Min Faithfulness':<30} | {orig_summary['faithfulness']['min']:<20.4f} | {summary['faithfulness']['min']:<20.4f}")
    print(f"{'RELEASE Rate (%)':<30} | {orig_summary['gate_outcomes']['release_pct']:<20.1f} | {summary['gate_outcomes']['release_pct']:<20.1f}")
    print(f"{'ESCALATE Rate (%)':<30} | {orig_summary['gate_outcomes']['escalate_pct']:<20.1f} | {summary['gate_outcomes']['escalate_pct']:<20.1f}")
    print(f"{'Mechanism Warning Rate (%)':<30} | {orig_summary['mechanism_warnings_rate_pct']:<20.1f} | {summary['mechanism_warnings_rate_pct']:<20.1f}")
    print(f"{'Histogram [0-0.2)':<30} | {orig_summary['faithfulness']['histogram_bins_0_to_1'][0]:<20} | {summary['faithfulness']['histogram_bins_0_to_1'][0]:<20}")
    print(f"{'Histogram [0.2-0.4)':<30} | {orig_summary['faithfulness']['histogram_bins_0_to_1'][1]:<20} | {summary['faithfulness']['histogram_bins_0_to_1'][1]:<20}")
    print(f"{'Histogram [0.4-0.6)':<30} | {orig_summary['faithfulness']['histogram_bins_0_to_1'][2]:<20} | {summary['faithfulness']['histogram_bins_0_to_1'][2]:<20}")
    print(f"{'Histogram [0.6-0.8)':<30} | {orig_summary['faithfulness']['histogram_bins_0_to_1'][3]:<20} | {summary['faithfulness']['histogram_bins_0_to_1'][3]:<20}")
    print(f"{'Histogram [0.8-1.0]':<30} | {orig_summary['faithfulness']['histogram_bins_0_to_1'][4]:<20} | {summary['faithfulness']['histogram_bins_0_to_1'][4]:<20}")
    print("="*50)
    
    # Figure: overlaid histograms
    plt.figure(figsize=(8, 5))
    x_bins = ["[0-0.2)", "[0.2-0.4)", "[0.4-0.6)", "[0.6-0.8)", "[0.8-1.0]"]
    x = np.arange(len(x_bins))
    width = 0.35
    
    plt.bar(x - width/2, orig_summary['faithfulness']['histogram_bins_0_to_1'], width, label='Sample-Grounded (System)', color='dodgerblue', edgecolor='black')
    plt.bar(x + width/2, summary['faithfulness']['histogram_bins_0_to_1'], width, label='Class-Only (Ablated)', color='coral', edgecolor='black')
    
    plt.title("Faithfulness Distribution: Sample-Grounded vs Class-Only Prompting")
    plt.xlabel("Faithfulness Score Range")
    plt.ylabel("Number of Samples (out of 100)")
    plt.xticks(x, x_bins)
    plt.legend()
    plt.tight_layout()
    plt.savefig(repo_dir / "outputs" / "audit_condition_comparison.png", dpi=150)
    plt.close()
    print("Saved comparison chart to outputs/audit_condition_comparison.png")

def main():
    run_part_a()
    run_part_b_c()

if __name__ == "__main__":
    main()
