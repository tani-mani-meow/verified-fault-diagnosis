import os
import sys
import json
from pathlib import Path

# Add src to python path if needed to import KNOWLEDGE_BASE
sys.path.append(str(Path(__file__).resolve().parent))
from knowledge_base import KNOWLEDGE_BASE

def load_thresholds():
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent
    thresholds_path = repo_dir / "outputs" / "feature_thresholds.json"
    
    if not thresholds_path.exists():
        raise FileNotFoundError(f"Thresholds file not found at {thresholds_path}. Run compute_thresholds.py first.")
        
    with open(thresholds_path, "r", encoding="utf-8") as f:
        return json.load(f)

def verify_claim(recording_features, predicted_class, claim, thresholds):
    """
    Verifies a single claim against the computed thresholds.
    claim format: {"feature": str, "direction": str, "critical": bool}
    """
    feature = claim.get("feature")
    direction = claim.get("direction", "").lower()
    
    # 1. Grounding check
    if predicted_class not in KNOWLEDGE_BASE:
        return {
            "claim": claim,
            "passed": False,
            "actual_value": None,
            "reference": None,
            "reason": f"Class '{predicted_class}' not in knowledge base"
        }
        
    class_grounding = KNOWLEDGE_BASE[predicted_class].get("grounding_features", {})
    if feature not in class_grounding:
        return {
            "claim": claim,
            "passed": False,
            "actual_value": recording_features.get(feature),
            "reference": None,
            "reason": "feature not grounded for this class"
        }
        
    if feature not in recording_features:
        return {
            "claim": claim,
            "passed": False,
            "actual_value": None,
            "reference": None,
            "reason": f"feature '{feature}' not found in recording features"
        }
        
    val = recording_features[feature]
    
    # Fetch thresholds
    global_th = thresholds["global"].get(feature)
    class_th = thresholds["classes"].get(predicted_class, {}).get(feature)
    
    if not global_th or not class_th:
        return {
            "claim": claim,
            "passed": False,
            "actual_value": val,
            "reference": None,
            "reason": f"thresholds for feature '{feature}' are missing"
        }
        
    global_p50 = global_th["p50"]
    global_p75 = global_th["p75"]
    global_p25 = global_th["p25"]
    class_median = class_th["p50"]
    
    passed = False
    reference = ""
    reason = ""
    
    # Determine checking logic based on direction
    is_high_direction = any(d in direction for d in ["elevated", "high", "shifted", "widened", "moderately"])
    is_low_direction = any(d in direction for d in ["low", "lower"])
    
    if is_high_direction:
        # elevated/high: class median > global p75 and recording val > global p50
        passed = (class_median > global_p75) and (val > global_p50)
        reference = f"class_median ({class_median:.4f}) > global_p75 ({global_p75:.4f}) AND val ({val:.4f}) > global_p50 ({global_p50:.4f})"
        if not passed:
            reason = f"Failed high/elevated checks: class median must exceed global p75 and recording value must exceed global p50."
            
    elif is_low_direction:
        # low/lower: class median < global p25 and recording val < global p50
        passed = (class_median < global_p25) and (val < global_p50)
        reference = f"class_median ({class_median:.4f}) < global_p25 ({global_p25:.4f}) AND val ({val:.4f}) < global_p50 ({global_p50:.4f})"
        if not passed:
            reason = f"Failed low/lower checks: class median must be below global p25 and recording value must be below global p50."
            
    elif direction == "distinctive":
        # distinctive: class median is in top-2 or bottom-2
        distinctive_info = thresholds["distinctive"].get(predicted_class, {})
        top_2 = distinctive_info.get("top_2", [])
        bottom_2 = distinctive_info.get("bottom_2", [])
        
        if feature in top_2:
            passed = (val > global_p50)
            reference = f"feature is distinctively high for class. val ({val:.4f}) > global_p50 ({global_p50:.4f})"
            if not passed:
                reason = f"Failed distinctive check: feature is in top-2 but recording value did not exceed global p50."
        elif feature in bottom_2:
            passed = (val < global_p50)
            reference = f"feature is distinctively low for class. val ({val:.4f}) < global_p50 ({global_p50:.4f})"
            if not passed:
                reason = f"Failed distinctive check: feature is in bottom-2 but recording value was not below global p50."
        else:
            passed = False
            reference = f"feature '{feature}' is not in top-2 or bottom-2 medians for class {predicted_class}"
            reason = f"Feature is not distinctive (top-2 or bottom-2) for this class."
            
    elif direction == "context_dependent":
        # context-dependent: passes if grounded
        passed = True
        reference = "Grounded context-dependent feature"
        
    else:
        # Fallback if direction unknown
        passed = False
        reason = f"Unknown direction '{direction}' in claim"
        
    return {
        "claim": claim,
        "passed": passed,
        "actual_value": val,
        "reference": reference,
        "reason": reason
    }

def verify_explanation(recording_features, predicted_class, claims, action_critical_idx=None, thresholds=None):
    """
    Verifies all claims in an explanation.
    """
    if thresholds is None:
        thresholds = load_thresholds()
        
    per_claim = []
    passed_count = 0
    escalate = False
    
    for idx, claim in enumerate(claims):
        res = verify_claim(recording_features, predicted_class, claim, thresholds)
        is_critical = claim.get("critical", False) or (action_critical_idx is not None and idx in action_critical_idx)
        res["critical"] = is_critical
        per_claim.append(res)
        
        if res["passed"]:
            passed_count += 1
        else:
            if is_critical:
                escalate = True
                
    faithfulness_score = passed_count / len(claims) if claims else 1.0
    
    if faithfulness_score < 0.8:
        escalate = True
        
    return {
        "per_claim": per_claim,
        "faithfulness_score": faithfulness_score,
        "escalate": escalate
    }
