# scripts/run_valuesonly_pilot.py
import os
import sys
import json
import time
import queue
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure we can import from src
script_dir = Path(__file__).resolve().parent
repo_dir = script_dir.parent
sys.path.insert(0, str(repo_dir / "src"))

from pipeline import run_pipeline
import llm_client

# Mutex for thread-safe output writes
write_lock = threading.Lock()
pilot_results = []

def audit_case(idx, original_idx, row):
    true_class = row['class']
    print(f"[{idx+1}/20] Pilot Auditing Index {original_idx} (True Class: {true_class})...")
    
    sample_dict = row.to_dict()
    
    start_time = time.time()
    try:
        # Run with grounding_mode="values_only"
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
            pilot_results.append(case_data)
            output_file = repo_dir / "outputs" / "faithfulness_audit_valuesonly_pilot.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(pilot_results, f, indent=2)
                
        print(f"    Index {original_idx} DONE | Pred: {pred_class} | Correct: {case_data['correct']} | Gate: {gate_decision} | Latency: {latency:.2f}s")
        return True
    except Exception as e:
        print(f"Error pilot auditing index {original_idx}: {e}")
        return False

def main():
    print("=== STARTING 20-CASE VALUES-ONLY PILOT ABLATION ===")
    
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
    
    # Take the first 20 cases from the stratified 100 cases
    pilot_df = audit_df.head(20)
    print(f"Selected {len(pilot_df)} cases for pilot.")
    print(pilot_df['class'].value_counts())
    
    # Setup work queue
    q = queue.Queue()
    for idx, (original_idx, row) in enumerate(pilot_df.iterrows()):
        q.put((idx, original_idx, row))
        
    def worker():
        while True:
            try:
                item = q.get_nowait()
            except queue.Empty:
                break
            idx, original_idx, row = item
            success = audit_case(idx, original_idx, row)
            if not success:
                print(f"Index {original_idx} failed. Putting back in queue.")
                q.put(item)
                time.sleep(5.0)
            else:
                time.sleep(1.0)
            q.task_done()
            
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(worker) for _ in range(3)]
        q.join()
        for f in futures:
            if f.exception() is not None:
                print(f"Worker thread failed: {f.exception()}")
        
    print("\nPilot completed successfully!")

if __name__ == "__main__":
    main()
