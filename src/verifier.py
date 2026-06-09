# src/verifier.py
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

def load_calibration():
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent
    calibration_path = repo_dir / "outputs" / "kb_calibration.json"
    
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration file not found at {calibration_path}. Run calibrate_kb.py first.")
        
    with open(calibration_path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_claim_dir(claim_dir):
    d = claim_dir.lower()
    if any(w in d for w in ["elevated", "high", "higher", "shifted", "widened", "moderately"]):
        return "high"
    elif any(w in d for w in ["low", "lower"]):
        return "low"
    else:
        return "unspecified"

def verify_claim(recording_features, predicted_class, claim, thresholds, calibration):
    """
    Verifies a single claim against the data-derived calibration direction.
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
    
    # 2. Get calibration data
    cal_class = calibration.get(predicted_class, {})
    if feature not in cal_class:
        return {
            "claim": claim,
            "passed": False,
            "actual_value": val,
            "reference": None,
            "reason": f"calibration data for feature '{feature}' in class '{predicted_class}' is missing"
        }
        
    cal_data = cal_class[feature]
    data_direction = cal_data["data_direction"] # "high", "low", or "neutral"
    
    global_th = thresholds["global"].get(feature, {})
    global_p50 = global_th.get("p50", 0.0)
    
    # Normalize claim direction
    claim_dir_norm = normalize_claim_dir(direction)
    
    passed = False
    reason = ""
    reference = f"data_direction: {data_direction}, global_p50: {global_p50:.4f}, val: {val:.4f}"
    
    if data_direction == "neutral":
        if claim_dir_norm in ["high", "low"]:
            passed = False
            reason = "feature not directionally characteristic for this class"
        else:
            # unspecified or soft claims (like context_dependent) pass if grounded
            passed = True
            reason = ""
            
    elif data_direction in ["high", "low"]:
        if claim_dir_norm == data_direction:
            if data_direction == "high":
                passed = (val > global_p50)
                if not passed:
                    reason = f"recording value {val:.4f} is not above global p50 {global_p50:.4f}"
            else: # "low"
                passed = (val < global_p50)
                if not passed:
                    reason = f"recording value {val:.4f} is not below global p50 {global_p50:.4f}"
        elif claim_dir_norm == "unspecified":
            # If claim is "distinctive" or "context_dependent"
            if "distinctive" in direction:
                if data_direction == "high":
                    passed = (val > global_p50)
                    if not passed:
                        reason = f"recording value {val:.4f} is not above global p50 {global_p50:.4f} for distinctive feature"
                else: # "low"
                    passed = (val < global_p50)
                    if not passed:
                        reason = f"recording value {val:.4f} is not below global p50 {global_p50:.4f} for distinctive feature"
            else: # context_dependent etc.
                passed = True
        else:
            # Mismatch: claim asserts high but data is low, or vice versa
            passed = False
            reason = f"direction mismatch: claim asserts '{claim_dir_norm}' but data shows '{data_direction}'"
            
    return {
        "claim": claim,
        "passed": passed,
        "actual_value": val,
        "reference": reference,
        "reason": reason
    }

def verify_explanation(recording_features, predicted_class, claims, action_critical_idx=None, thresholds=None, calibration=None):
    """
    Verifies all claims in an explanation.
    """
    if thresholds is None:
        thresholds = load_thresholds()
    if calibration is None:
        calibration = load_calibration()
        
    per_claim = []
    passed_count = 0
    escalate = False
    
    for idx, claim in enumerate(claims):
        res = verify_claim(recording_features, predicted_class, claim, thresholds, calibration)
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
