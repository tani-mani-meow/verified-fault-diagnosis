# scripts/run_valuesonly_audit.py
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
    print(f"[{idx+1}/100] Values-Only Auditing Index {original_idx} (True Class: {true_class})...")
    
    sample_dict = row.to_dict()
    
    start_time = time.time()
    try:
        report = run_pipeline(sample_dict, true_label=true_class, mock=False, grounding_mode="values_only")
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
            audit_results.append(case_data)
            cases_run[int(original_idx)] = case_data
            audit_file = repo_dir / "outputs" / "faithfulness_audit_valuesonly.json"
            with open(audit_file, "w", encoding="utf-8") as f:
                json.dump(audit_results, f, indent=2)
                
        print(f"    Index {original_idx} DONE | Pred: {pred_class} | Correct: {case_data['correct']} | Gate: {gate_decision} | Latency: {latency:.2f}s")
        return True
    except Exception as e:
        print(f"Error values-only auditing index {original_idx}: {e}")
        return False

def main():
    print("=== STARTING FULL 100-CASE VALUES-ONLY ABLATION ===")
    
    # 1. Load and split features.csv to match exactly the 100 cases
    features_csv = repo_dir / "outputs" / "features.csv"
    if not features_csv.exists():
        print(f"Error: features.csv not found. Run training first.")
        sys.exit(1)
        
    df = pd.read_csv(features_csv)
    y_str = df['class']
    class_names = sorted(y_str.unique())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    y = y_str.map(class_to_idx).values
    
    _, df_test, _, _ = train_test_split(df, y, test_size=0.2, random_state=42, stratify=y)
    fault_test = df_test[df_test['class'] != 'Healthy']
    audit_df, _ = train_test_split(fault_test, train_size=100, stratify=fault_test['class'], random_state=42)
    
    print(f"Selected {len(audit_df)} test rows for stratified audit.")
    
    # Load pilot cases and existing audit cases (to allow resumes)
    pilot_file = repo_dir / "outputs" / "faithfulness_audit_valuesonly_pilot.json"
    audit_file = repo_dir / "outputs" / "faithfulness_audit_valuesonly.json"
    
    global audit_results, cases_run
    
    # First load pilot cases
    if pilot_file.exists():
        try:
            with open(pilot_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            for c in loaded:
                idx_val = int(c["index"])
                if idx_val not in cases_run:
                    audit_results.append(c)
                    cases_run[idx_val] = c
            print(f"Loaded {len(loaded)} cases from pilot file.")
        except Exception as e:
            print(f"Could not load pilot file: {e}")
            
    # Then load existing audit cases if present
    if audit_file.exists():
        try:
            with open(audit_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            for c in loaded:
                idx_val = int(c["index"])
                if idx_val not in cases_run:
                    audit_results.append(c)
                    cases_run[idx_val] = c
            print(f"Loaded {len(loaded)} cases from existing audit file.")
        except Exception as e:
            print(f"Could not load existing audit file: {e}")
            
    # Save the consolidated list to the audit file at start
    with open(audit_file, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)
        
    # Setup work queue for remaining cases
    q = queue.Queue()
    for idx, (original_idx, row) in enumerate(audit_df.iterrows()):
        if int(original_idx) not in cases_run:
            q.put((idx, original_idx, row))
            
    print(f"Queue size for execution: {q.qsize()} cases remaining.")
    
    start_audit_time = time.time()
    
    # Worker function
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
            
    # Final aggregation and save
    print(f"\nAll 100 cases completed in {time.time() - start_audit_time:.2f}s!")
    
    # 2. Aggregations & Summary
    df_audit = pd.DataFrame(audit_results)
    f_mean = float(df_audit["faithfulness"].mean())
    f_median = float(df_audit["faithfulness"].median())
    f_std = float(df_audit["faithfulness"].std()) if len(df_audit) > 1 else 0.0
    f_min = float(df_audit["faithfulness"].min())
    f_hist, _ = np.histogram(df_audit["faithfulness"], bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.001])
    f_hist_list = [int(x) for x in f_hist]
    
    release_pct = float((df_audit["gate_decision"] == "RELEASE").sum() / len(df_audit) * 100)
    escalate_pct = float((df_audit["gate_decision"] == "ESCALATE").sum() / len(df_audit) * 100)
    reason_counts = df_audit[df_audit["gate_decision"] == "ESCALATE"]["gate_reason"].value_counts().to_dict()
    reason_pct = {k: float(v / len(df_audit) * 100) for k, v in reason_counts.items()}
    warning_cases = int(df_audit["mechanism_warnings"].apply(lambda w: len(w) > 0).sum())
    warning_rate = float(warning_cases / len(df_audit) * 100)
    
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
        "mechanism_warnings_rate_pct": warning_rate
    }
    
    summary_file = repo_dir / "outputs" / "faithfulness_audit_valuesonly_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved values-only summary to {summary_file}")

if __name__ == "__main__":
    main()
