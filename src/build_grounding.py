# src/build_grounding.py
import os
import sys
import json
from pathlib import Path

def interpolate_p40(p25, p50):
    return p25 + 0.6 * (p50 - p25)

def interpolate_p60(p50, p75):
    return p50 + 0.4 * (p75 - p50)

def main():
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent
    thresholds_path = repo_dir / "outputs" / "feature_thresholds.json"
    grounding_path = repo_dir / "outputs" / "grounding_features.json"
    
    if not thresholds_path.exists():
        print(f"Error: thresholds file not found at {thresholds_path}. Run compute_thresholds.py first.")
        sys.exit(1)
        
    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds = json.load(f)
        
    fault_classes = ["LIV", "LOV", "NRV", "Bearing", "Flywheel", "Piston", "Riderbelt"]
    feature_cols = list(thresholds["global"].keys())
    
    grounding_results = {}
    
    for c in fault_classes:
        strong_candidates = []
        moderate_candidates = []
        
        for feature in feature_cols:
            # Get global percentiles
            global_feat = thresholds["global"][feature]
            p25 = global_feat["p25"]
            p50 = global_feat["p50"]
            p75 = global_feat["p75"]
            
            spread = p75 - p25
            if spread <= 1e-9:
                continue  # Skip degenerate/all-zero features
                
            # Class median
            class_p50 = thresholds["classes"][c][feature]["p50"]
            
            # Separation score
            sep = (class_p50 - p50) / spread
            
            # Strong rule
            strong_dir = None
            if class_p50 >= p75:
                strong_dir = "high"
            elif class_p50 <= p25:
                strong_dir = "low"
                
            # Moderate rule
            global_p40 = interpolate_p40(p25, p50)
            global_p60 = interpolate_p60(p50, p75)
            
            mod_dir = None
            if class_p50 >= global_p60:
                mod_dir = "high"
            elif class_p50 <= global_p40:
                mod_dir = "low"
                
            info = {
                "feature": feature,
                "sep": sep,
                "class_p50": class_p50,
                "global_p50": p50,
                "global_p40": global_p40,
                "global_p60": global_p60
            }
            
            if strong_dir is not None:
                info["direction"] = strong_dir
                info["tier"] = "strong"
                strong_candidates.append(info)
            elif mod_dir is not None:
                info["direction"] = mod_dir
                info["tier"] = "moderate"
                moderate_candidates.append(info)
                
        # Sort candidates by |sep| descending
        strong_candidates.sort(key=lambda x: abs(x["sep"]), reverse=True)
        moderate_candidates.sort(key=lambda x: abs(x["sep"]), reverse=True)
        
        selected = []
        strong_count = len(strong_candidates)
        
        if strong_count >= 6:
            selected = strong_candidates[:6]
        else:
            selected = list(strong_candidates)
            needed = 6 - len(selected)
            selected.extend(moderate_candidates[:needed])
            
        grounding_results[c] = selected
        
        # Print class summary
        print(f"\nClass {c} Grounding Selection (Strong count = {strong_count}):")
        for idx, item in enumerate(selected):
            print(f"  {idx+1}. {item['feature']:25s} | dir={item['direction']:4s} | sep={item['sep']:8.4f} | tier={item['tier']}")
            
        # Warning check
        if strong_count < 4:
            print(f"  WARNING: Class {c} has only {strong_count} strong features (grounding weakness)!")
            
    # Save to outputs/grounding_features.json
    grounding_path.parent.mkdir(parents=True, exist_ok=True)
    with open(grounding_path, "w", encoding="utf-8") as f:
        json.dump(grounding_results, f, indent=4)
    print(f"\nSaved data-driven grounding features to {grounding_path}")

if __name__ == "__main__":
    main()
