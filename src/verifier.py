# src/verifier.py
import os
import sys
import json
from pathlib import Path

# Add src to python path if needed to import KNOWLEDGE_BASE
sys.path.insert(0, str(Path(__file__).resolve().parent))
from knowledge_base import KNOWLEDGE_BASE, normalize_direction

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
    global_p25 = global_th.get("p25", 0.0)
    global_p75 = global_th.get("p75", 0.0)
    
    # Normalize claim direction
    claim_dir_norm = normalize_direction(direction)
    
    passed = False
    reason = ""
    reference = f"data_direction: {data_direction}, global_p25: {global_p25:.4f}, global_p75: {global_p75:.4f}, val: {val:.4f}"
    
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
                passed = (val >= global_p75)
                if not passed:
                    reason = f"recording value {val:.4f} is not above global p75 {global_p75:.4f}"
            else: # "low"
                passed = (val <= global_p25)
                if not passed:
                    reason = f"recording value {val:.4f} is not below global p25 {global_p25:.4f}"
        elif claim_dir_norm == "unspecified":
            # If claim is "distinctive" or "context_dependent"
            if "distinctive" in direction:
                if data_direction == "high":
                    passed = (val >= global_p75)
                    if not passed:
                        reason = f"recording value {val:.4f} is not above global p75 {global_p75:.4f} for distinctive feature"
                else: # "low"
                    passed = (val <= global_p25)
                    if not passed:
                        reason = f"recording value {val:.4f} is not below global p25 {global_p25:.4f} for distinctive feature"
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
                
    faithfulness_score = passed_count / len(claims) if claims else 0.0
    
    if faithfulness_score < 0.8 or len(claims) == 0:
        escalate = True
        
    return {
        "per_claim": per_claim,
        "faithfulness_score": faithfulness_score,
        "escalate": escalate
    }

def check_sample_faithfulness(recording_features, predicted_class, claims, thresholds=None, calibration=None):
    """
    Computes both sample_true (against actual p75/p25 direction) and class_consistent (against class calibration).
    """
    if thresholds is None:
        thresholds = load_thresholds()
    if calibration is None:
        calibration = load_calibration()
        
    per_claim = []
    sample_true_count = 0
    class_consistent_count = 0
    
    class_grounding = KNOWLEDGE_BASE.get(predicted_class, {}).get("grounding_features", {})
    
    for c in claims:
        feature = c.get("feature")
        direction = c.get("direction", "")
        action_critical = c.get("critical", False) or c.get("action_critical", False)
        
        # Grounding check
        if feature not in class_grounding:
            per_claim.append({
                "feature": feature,
                "claimed_dir": direction,
                "sample_dir": "neutral",
                "class_dir": "neutral",
                "sample_true": False,
                "class_consistent": False,
                "atypical": False,
                "action_critical": action_critical,
                "reason": "feature not grounded for this class"
            })
            continue
            
        if feature not in recording_features:
            per_claim.append({
                "feature": feature,
                "claimed_dir": direction,
                "sample_dir": "neutral",
                "class_dir": "neutral",
                "sample_true": False,
                "class_consistent": False,
                "atypical": False,
                "action_critical": action_critical,
                "reason": f"feature '{feature}' not found in recording features"
            })
            continue
            
        value = recording_features[feature]
        global_feat = thresholds["global"].get(feature, {})
        global_p25 = global_feat.get("p25", 0.0)
        global_p75 = global_feat.get("p75", 0.0)
        
        # Determine sample actual direction
        if value >= global_p75:
            sample_dir = "high"
        elif value <= global_p25:
            sample_dir = "low"
        else:
            sample_dir = "neutral"
            
        claim_dir_norm = normalize_direction(direction)
        
        # sample_true check
        if claim_dir_norm == "high":
            sample_true = (sample_dir == "high")
        elif claim_dir_norm == "low":
            sample_true = (sample_dir == "low")
        else:
            sample_true = (sample_dir == "neutral")
            
        # class_consistent check
        cal_class = calibration.get(predicted_class, {})
        class_dir = cal_class.get(feature, {}).get("data_direction", "neutral")
        if claim_dir_norm == "high":
            class_consistent = (class_dir == "high")
        elif claim_dir_norm == "low":
            class_consistent = (class_dir == "low")
        else:
            # Unspecified claim direction is always class-consistent
            class_consistent = True
            
        atypical = sample_true and not class_consistent
        
        if sample_true:
            sample_true_count += 1
        if class_consistent:
            class_consistent_count += 1
            
        per_claim.append({
            "feature": feature,
            "claimed_dir": direction,
            "sample_dir": sample_dir,
            "class_dir": class_dir,
            "sample_true": sample_true,
            "class_consistent": class_consistent,
            "atypical": atypical,
            "action_critical": action_critical
        })
        
    num_claims = len(claims)
    faithfulness = sample_true_count / num_claims if num_claims > 0 else 0.0
    class_consistency_score = class_consistent_count / num_claims if num_claims > 0 else 0.0
    
    return {
        "per_claim": per_claim,
        "faithfulness": faithfulness,
        "class_consistency": class_consistency_score
    }

