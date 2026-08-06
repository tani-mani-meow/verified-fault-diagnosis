# src/calibrate_kb.py
import os
import sys
import json
from pathlib import Path

# Add src to sys.path to load KNOWLEDGE_BASE
sys.path.insert(0, str(Path(__file__).resolve().parent))
from knowledge_base import KNOWLEDGE_BASE, normalize_direction

def interpolate_p40(p25, p50):
    return p25 + 0.6 * (p50 - p25)

def interpolate_p60(p50, p75):
    return p50 + 0.4 * (p75 - p50)

def main():
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent
    thresholds_path = repo_dir / "outputs" / "feature_thresholds.json"
    calibration_path = repo_dir / "outputs" / "kb_calibration.json"
    
    if not thresholds_path.exists():
        print(f"Error: thresholds file not found at {thresholds_path}. Run compute_thresholds.py first.")
        sys.exit(1)
        
    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds = json.load(f)
        
    calibration = {}
    mismatch_list = []
    grand_total_mismatches = 0
    grand_total_matches = 0
    
    classes = sorted(KNOWLEDGE_BASE.keys())
    
    for c in classes:
        calibration[c] = {}
        class_matches = 0
        class_mismatches = 0
        
        grounding_feats = KNOWLEDGE_BASE[c].get("grounding_features", {})
        for feature, kb_dir_raw in grounding_feats.items():
            # Get class median (p50)
            class_p50 = thresholds["classes"].get(c, {}).get(feature, {}).get("p50", 0.0)
            
            # Get global percentiles
            global_feat = thresholds["global"].get(feature, {})
            p25 = global_feat.get("p25", 0.0)
            p50 = global_feat.get("p50", 0.0)
            p75 = global_feat.get("p75", 0.0)
            
            # Interpolate
            global_p40 = interpolate_p40(p25, p50)
            global_p60 = interpolate_p60(p50, p75)
            
            # Determine data direction
            if class_p50 >= global_p60:
                data_dir = "high"
            elif class_p50 <= global_p40:
                data_dir = "low"
            else:
                data_dir = "neutral"
                
            # Normalize KB direction
            kb_dir = normalize_direction(kb_dir_raw)
            
            # Check match
            match = (kb_dir == "unspecified") or (kb_dir == data_dir)
            
            calibration[c][feature] = {
                "data_direction": data_dir,
                "kb_direction": kb_dir,
                "match": match,
                "class_p50": class_p50,
                "global_p40": global_p40,
                "global_p60": global_p60
            }
            
            if match:
                class_matches += 1
                grand_total_matches += 1
            else:
                class_mismatches += 1
                grand_total_mismatches += 1
                mismatch_list.append((c, feature, kb_dir_raw, kb_dir, data_dir, class_p50, global_p40, global_p60))
                
        print(f"Class {c:12s}: Matches = {class_matches:2d}, Mismatches = {class_mismatches:2d}")
        
    # Print mismatches
    if mismatch_list:
        print("\n--- Detailed Mismatches ---")
        print(f"{'Class':10s} | {'Feature':25s} | {'KB Raw':25s} | {'KB Norm':8s} | {'Data Dir':8s} | {'Class p50':10s} | {'Glob p40':10s} | {'Glob p60':10s}")
        print("-" * 123)
        for row in mismatch_list:
            c, feat, raw, kb, data, cp50, gp40, gp60 = row
            print(f"{c:10s} | {feat:25s} | {raw:25s} | {kb:8s} | {data:8s} | {cp50:10.4f} | {gp40:10.4f} | {gp60:10.4f}")
            
    print(f"\nGrand Total Matches: {grand_total_matches}")
    print(f"Grand Total Mismatches: {grand_total_mismatches}")
    
    # Save calibration file
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    with open(calibration_path, "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=4)
    print(f"Saved calibration file to {calibration_path}")

if __name__ == "__main__":
    main()
