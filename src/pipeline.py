# src/pipeline.py
import os
import sys
import json
import re
from pathlib import Path
import numpy as np
import joblib

sys.path.append(str(Path(__file__).resolve().parent))
from knowledge_base import KNOWLEDGE_BASE
from verifier import load_thresholds, load_calibration, verify_explanation, verify_claim
import llm_client

MOCK_FIXTURES = {
    "correct": {
        "claims": [
            {"feature": "spectral_rolloff_85", "direction": "low", "action_critical": False},
            {"feature": "skewness", "direction": "low", "action_critical": False},
            {"feature": "spectral_centroid", "direction": "low", "action_critical": False},
            {"feature": "wpt_energy_15", "direction": "low", "action_critical": False},
            {"feature": "wpt_energy_18", "direction": "low", "action_critical": False},
            {"feature": "wpt_energy_50", "direction": "low", "action_critical": False}
        ],
        "explanation": "suction valve leakage (LIV) is confirmed as high frequency roll-off under 85% is low and spectral centroid is low, showing reduced high-frequency noise."
    },
    "flipped": {
        "claims": [
            {"feature": "spectral_rolloff_85", "direction": "low", "action_critical": False},
            {"feature": "skewness", "direction": "low", "action_critical": False},
            {"feature": "spectral_centroid", "direction": "high", "action_critical": False}, # Flipped direction!
            {"feature": "wpt_energy_15", "direction": "high", "action_critical": False}, # Flipped direction!
            {"feature": "wpt_energy_18", "direction": "low", "action_critical": False},
            {"feature": "wpt_energy_50", "direction": "low", "action_critical": False}
        ],
        "explanation": "suction valve leakage shows some low features, but spectral centroid and wpt_energy_15 are incorrectly claimed as high."
    },
    "critical": {
        "claims": [
            {"feature": "spectral_rolloff_85", "direction": "low", "action_critical": False},
            {"feature": "skewness", "direction": "low", "action_critical": False},
            {"feature": "spectral_centroid", "direction": "low", "action_critical": False},
            {"feature": "wpt_energy_15", "direction": "low", "action_critical": False},
            {"feature": "wpt_energy_18", "direction": "low", "action_critical": False},
            {"feature": "wpt_energy_50", "direction": "high", "action_critical": True} # Flipped and critical!
        ],
        "explanation": "suction valve leakage explanation with a false action-critical claim."
    }
}

def run_pipeline(sample_row, true_label=None, mock=False):
    """
    Runs explain -> verify -> gate pipeline.
    mock parameter can be True (defaults to 'correct'), or a string key in MOCK_FIXTURES.
    """
    repo_dir = Path(__file__).resolve().parent.parent
    
    # 1. Load thresholds and calibration
    thresholds = load_thresholds()
    calibration = load_calibration()
    
    # 2. Predict Class and Confidence (Cert 1)
    model_path = repo_dir / "outputs" / "xgb_model.joblib"
    scaler_path = repo_dir / "outputs" / "scaler.joblib"
    feature_names_path = repo_dir / "outputs" / "feature_names.json"
    
    if not model_path.exists() or not scaler_path.exists() or not feature_names_path.exists():
        raise FileNotFoundError("Baseline model binaries or feature names not found. Run training first.")
        
    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(feature_names_path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)
        
    # Order sample row features
    import pandas as pd
    feature_values = [sample_row[feat] for feat in feature_names]
    X_raw = pd.DataFrame([feature_values], columns=feature_names)
    X_scaled = scaler.transform(X_raw)
    
    probs = clf.predict_proba(X_scaled)[0]
    pred_idx = np.argmax(probs)
    confidence = float(probs[pred_idx])
    
    class_names = ['Bearing', 'Flywheel', 'Healthy', 'LIV', 'LOV', 'NRV', 'Piston', 'Riderbelt']
    predicted_class = class_names[pred_idx]
    
    # 3. Retrieve KB & Build Grounding Context (Cert 2)
    kb_entry = KNOWLEDGE_BASE.get(predicted_class)
    if not kb_entry:
        # Healthy or ungrounded class: escalate
        return {
            "predicted": predicted_class,
            "confidence": confidence,
            "true_label": true_label,
            "claims": [],
            "per_claim": [],
            "faithfulness": 0.0,
            "gate_decision": "ESCALATE",
            "reason": f"Class '{predicted_class}' has no grounding entry in KNOWLEDGE_BASE",
            "explanation": "No diagnostic explanation generated for Healthy/unsupported class."
        }
        
    # Load glossary
    glossary_path = repo_dir / "outputs" / "feature_glossary.json"
    with open(glossary_path, "r", encoding="utf-8") as f:
        glossary = json.load(f)
        
    grounding_feats = kb_entry.get("grounding_features", {})
    context_list = []
    
    for feat in sorted(grounding_feats.keys()):
        expected_dir = grounding_feats[feat]
        actual_val = sample_row[feat]
        gloss = glossary.get(feat, {})
        label = gloss.get("label", feat)
        
        # global and class percentiles
        global_p50 = thresholds["global"].get(feat, {}).get("p50", 0.0)
        class_p50 = thresholds["classes"].get(predicted_class, {}).get(feat, {}).get("p50", 0.0)
        global_p25 = thresholds["global"].get(feat, {}).get("p25", 0.0)
        global_p75 = thresholds["global"].get(feat, {}).get("p75", 0.0)
        
        context_list.append({
            "feature": feat,
            "label": label,
            "expected_direction": expected_dir,
            "actual_value": actual_val,
            "class_p50": class_p50,
            "global_p50": global_p50,
            "global_p25": global_p25,
            "global_p75": global_p75
        })
        
    # 4. LLM Generation or Mock Fixture
    parsed_json = None
    unparseable = False
    
    if mock:
        fixture_key = mock if isinstance(mock, str) else "correct"
        parsed_json = MOCK_FIXTURES.get(fixture_key, MOCK_FIXTURES["correct"])
    else:
        # Check if real API key is available
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key or api_key in ["dummy_key", "your_key_here"]:
            raise ValueError("GEMINI_API_KEY is not set. Run in mock mode or update .env file.")
            
        # Build prompt
        context_str = json.dumps(context_list, indent=2)
        system_prompt = (
            "You are an expert diagnostic assistant for air compressor maintenance.\n"
            "Your task is to generate a technician-facing explanation and a structured list of claims based ONLY on the provided grounding context.\n\n"
            "Strict Rules:\n"
            "1. ONLY make claims about features present in the grounding context. Any other feature will fail verification.\n"
            "2. For each claim, specify:\n"
            "   - 'feature': exact canonical feature name (e.g. 'spectral_rolloff_85').\n"
            "   - 'direction': 'high' or 'low'.\n"
            "   - 'action_critical': true if this feature is critical for the diagnosis/action, false otherwise.\n"
            "3. In the 'explanation' string:\n"
            "   - Write a plain-English explanation for a technician (maximum 120 words).\n"
            "   - Reference frequency bands or human descriptions (e.g. 'energy near 1750-1875 Hz' or 'frequency below which 85% of energy lies'), NEVER use raw canonical names (like 'wpt_energy_15' or 'spectral_rolloff_85') in the explanation prose.\n"
            "   - Incorporate the physics mechanism (from the knowledge base) and how the measured features support this.\n"
            "4. Output MUST be a single, strict JSON object. No Markdown wrappers except optionally standard json block:\n"
            "{\n"
            "  'claims': [\n"
            "    {'feature': 'canonical_name', 'direction': 'high|low', 'action_critical': true/false}\n"
            "  ],\n"
            "  'explanation': 'Technician explanation...'\n"
            "}"
        )
        
        prompt = (
            f"Predicted fault class: {predicted_class}\n"
            f"Physics Mechanism: {kb_entry.get('mechanism')}\n\n"
            f"Grounding Context features (percentiles and actual values):\n{context_str}\n\n"
            "Generate the JSON response matching the schema above."
        )
        
        response_text = llm_client.generate(prompt, system=system_prompt)
        
        # Parse JSON defensively
        clean_text = response_text.strip()
        if clean_text.startswith("```"):
            m = re.search(r"```(?:json)?\s*([\s\S]*?)```", clean_text)
            if m:
                clean_text = m.group(1).strip()
                
        try:
            parsed_json = json.loads(clean_text)
        except Exception:
            unparseable = True
            parsed_json = {
                "claims": [],
                "explanation": f"Failed to parse LLM output as JSON. Raw output: {response_text}"
            }
            
    # 5. Run Verifier (Cert 3)
    if unparseable:
        return {
            "predicted": predicted_class,
            "confidence": confidence,
            "true_label": true_label,
            "claims": [],
            "per_claim": [],
            "faithfulness": 0.0,
            "gate_decision": "ESCALATE",
            "reason": "unparseable_llm_output",
            "explanation": parsed_json["explanation"]
        }
        
    claims_list = parsed_json.get("claims", [])
    verifier_claims = []
    for c in claims_list:
        vc = dict(c)
        vc["critical"] = vc.get("action_critical", False)
        verifier_claims.append(vc)
        
    res = verify_explanation(sample_row, predicted_class, verifier_claims, thresholds=thresholds, calibration=calibration)
    
    # 6. Trust Gate
    faithfulness = res["faithfulness_score"]
    escalate_gate = res["escalate"]
    
    # Double check if any action-critical claim failed
    action_critical_failed = False
    for r in res["per_claim"]:
        if r["critical"] and not r["passed"]:
            action_critical_failed = True
            
    if escalate_gate or action_critical_failed or faithfulness < 0.8:
        gate_decision = "ESCALATE"
        if action_critical_failed:
            reason = "action-critical claim failed"
        elif faithfulness < 0.8:
            reason = "low faithfulness score"
        else:
            reason = "verification failed"
    else:
        gate_decision = "RELEASE"
        reason = "passed verification"
        
    return {
        "predicted": predicted_class,
        "confidence": confidence,
        "true_label": true_label,
        "claims": claims_list,
        "per_claim": res["per_claim"],
        "faithfulness": faithfulness,
        "gate_decision": gate_decision,
        "reason": reason,
        "explanation": parsed_json.get("explanation", "")
    }
