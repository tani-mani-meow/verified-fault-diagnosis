# scripts/run_pipeline_demo.py
import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Adjust paths to make sure we can import from src
script_dir = Path(__file__).resolve().parent
repo_dir = script_dir.parent
sys.path.insert(0, str(repo_dir / "src"))

from pipeline import run_pipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Explain-Verify-Gate Pipeline Demo")
    parser.add_argument("--real", action="store_true", help="Run in real mode using Gemini API")
    return parser.parse_args()

def get_test_samples():
    features_csv = repo_dir / "outputs" / "features.csv"
    if not features_csv.exists():
        print(f"Error: features.csv not found at {features_csv}. Run train_baseline.py first.")
        sys.exit(1)
        
    df = pd.read_csv(features_csv)
    y_str = df['class']
    class_names = sorted(y_str.unique())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    y = y_str.map(class_to_idx).values
    
    # 80/20 train/test split, random_state=42, stratify by class
    _, df_test, _, _ = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )
    
    samples = {}
    fault_classes = ['Bearing', 'Flywheel', 'LIV', 'LOV', 'NRV', 'Piston', 'Riderbelt']
    for c in fault_classes:
        class_subset = df_test[df_test['class'] == c]
        if class_subset.empty:
            print(f"Error: No test samples found for class {c}")
            sys.exit(1)
        # Convert first row to a dict
        samples[c] = class_subset.iloc[0].to_dict()
        
    return samples

def run_mock_demo(samples):
    print("=== RUNNING MOCK PIPELINE DEMO ===")
    liv_sample = samples["LIV"]
    
    # (a) Correct fixture -> expect RELEASE
    print("\n--- Running Fixture (a): Fully Correct Explanation ---")
    report_a = run_pipeline(liv_sample, true_label="LIV", mock="correct")
    print(f"Predicted Class: {report_a['predicted']} (Confidence: {report_a['confidence']:.4f})")
    print(f"Faithfulness Score: {report_a['faithfulness']:.4f}")
    print(f"Gate Decision: {report_a['gate_decision']} (Reason: {report_a['reason']})")
    print(f"Explanation: {report_a['explanation']}")
    assert report_a["gate_decision"] == "RELEASE", f"Expected RELEASE, got {report_a['gate_decision']}"
    
    # (b) Flipped normal claim -> expect ESCALATE (low faithfulness)
    print("\n--- Running Fixture (b): Flipped Direction on Normal Claim ---")
    report_b = run_pipeline(liv_sample, true_label="LIV", mock="flipped")
    print(f"Predicted Class: {report_b['predicted']} (Confidence: {report_b['confidence']:.4f})")
    print(f"Faithfulness Score: {report_b['faithfulness']:.4f}")
    print(f"Gate Decision: {report_b['gate_decision']} (Reason: {report_b['reason']})")
    print(f"Explanation: {report_b['explanation']}")
    assert report_b["gate_decision"] == "ESCALATE", f"Expected ESCALATE, got {report_b['gate_decision']}"
    assert report_b["reason"] == "low faithfulness score", f"Expected 'low faithfulness score', got '{report_b['reason']}'"
    
    # (c) False action-critical claim -> expect ESCALATE
    print("\n--- Running Fixture (c): False Action-Critical Claim ---")
    report_c = run_pipeline(liv_sample, true_label="LIV", mock="critical")
    print(f"Predicted Class: {report_c['predicted']} (Confidence: {report_c['confidence']:.4f})")
    print(f"Faithfulness Score: {report_c['faithfulness']:.4f}")
    print(f"Gate Decision: {report_c['gate_decision']} (Reason: {report_c['reason']})")
    print(f"Explanation: {report_c['explanation']}")
    assert report_c["gate_decision"] == "ESCALATE", f"Expected ESCALATE, got {report_c['gate_decision']}"
    assert report_c["reason"] == "action-critical claim failed", f"Expected 'action-critical claim failed', got '{report_c['reason']}'"
    
    print("\nALL MOCK GATE DEMO ASSERTIONS PASSED SUCCESSFULLY!")

def run_real_demo(samples):
    print("=== RUNNING REAL PIPELINE DEMO WITH GEMINI ===")
    
    # Verify API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key in ["dummy_key", "your_key_here"]:
        print("WARNING: GEMINI_API_KEY is missing or set to placeholder in .env. Skipping real demo.")
        return
        
    results = {}
    for c, sample_row in samples.items():
        print(f"\nProcessing real pipeline for class: {c}...")
        try:
            report = run_pipeline(sample_row, true_label=c, mock=False)
            results[c] = report
            print(f"Predicted: {report['predicted']} (Confidence: {report['confidence']:.4f})")
            print(f"Faithfulness Score: {report['faithfulness']:.4f}")
            print(f"Gate Decision: {report['gate_decision']} (Reason: {report['reason']})")
            print(f"Explanation: {report['explanation']}")
            print("-" * 50)
            
            # Politeness delay to prevent free tier rate limits
            import time
            time.sleep(1.5)
        except Exception as e:
            print(f"Error running pipeline for class {c}: {e}")
            
    # Save outputs/pipeline_demo_report.json
    output_path = repo_dir / "outputs" / "pipeline_demo_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved real Gemini pipeline reports to {output_path}")

def main():
    args = parse_args()
    samples = get_test_samples()
    
    if args.real:
        run_real_demo(samples)
    else:
        run_mock_demo(samples)

if __name__ == "__main__":
    main()
