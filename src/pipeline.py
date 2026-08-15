import os
import sys
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.exceptions import InconsistentVersionWarning

sys.path.insert(0, str(Path(__file__).resolve().parent))
from knowledge_base import KNOWLEDGE_BASE, ALL_CLASSES
from verifier import load_thresholds, load_calibration, check_sample_faithfulness, check_mechanism_words
from verification_contract import (STRICT_POLICY, decide_gate, finite_number,
                                   sample_direction, validate_response, parse_response)
import llm_client

MOCK_FIXTURES = {
    "correct": {
        "claims": [
            {"feature": "spectral_rolloff_85", "direction": "low", "action_critical": False},
            {"feature": "spectral_centroid", "direction": "low", "action_critical": False}
        ],
        "explanation": "suction valve leakage (LIV) is confirmed as high frequency roll-off under 85% is low and spectral centroid is low, showing reduced high-frequency noise."
    },
    "flipped": {
        "claims": [
            {"feature": "spectral_rolloff_85", "direction": "low", "action_critical": False},
            {"feature": "skewness", "direction": "low", "action_critical": False},
            {"feature": "spectral_centroid", "direction": "low", "action_critical": False}
        ],
        "explanation": "suction valve leakage shows some low features, but skewness is claimed as low when it is actually neutral."
    },
    "critical": {
        "claims": [
            {"feature": "spectral_rolloff_85", "direction": "low", "action_critical": False},
            {"feature": "skewness", "direction": "low", "action_critical": True}
        ],
        "explanation": "suction valve leakage explanation with a false action-critical claim."
    },
    "atypical_non_critical": {
        "claims": [
            {"feature": "spectral_rolloff_85", "direction": "low", "action_critical": False},
            {"feature": "spectral_centroid", "direction": "low", "action_critical": False},
            {"feature": "skewness", "direction": "high", "action_critical": False}
        ],
        "explanation": "suction valve leakage explanation with atypical features."
    },
    "atypical_critical": {
        "claims": [
            {"feature": "spectral_rolloff_85", "direction": "low", "action_critical": False},
            {"feature": "spectral_centroid", "direction": "low", "action_critical": False},
            {"feature": "skewness", "direction": "high", "action_critical": True}
        ],
        "explanation": "suction valve leakage explanation with atypical action-critical features."
    },
    "impulsive_unsupported": {
        "claims": [],
        "explanation": "The analysis shows a repeated impulsive event on the discharge side."
    },
    "impulsive_supported": {
        "claims": [],
        "explanation": "The analysis shows a repeated impulsive event once per rotation."
    }
}

def run_pipeline(sample_row, true_label=None, mock=False, grounding_mode="sample", artifact_dir=None):
    repo_dir = Path(__file__).resolve().parent.parent
    if grounding_mode not in ('sample', 'class_only', 'values_only'):
        raise ValueError('Unknown grounding mode')
    artifact_dir = Path(artifact_dir or os.environ.get('VFD_ARTIFACT_DIR') or repo_dir / 'outputs')
    
    thresholds = load_thresholds(artifact_dir)
    calibration = load_calibration(artifact_dir)
    
    model_path = artifact_dir / "xgb_model.joblib"
    scaler_path = artifact_dir / "scaler.joblib"
    feature_names_path = artifact_dir / "feature_names.json"
    
    if not model_path.exists() or not scaler_path.exists() or not feature_names_path.exists():
        raise FileNotFoundError("Baseline model binaries or feature names not found. Run training first.")
        
    with warnings.catch_warnings():
        warnings.simplefilter('error', InconsistentVersionWarning)
        try:
            clf = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
        except InconsistentVersionWarning as exc:
            raise ValueError('Incompatible historical model. Run scripts/reproduce_baseline.py '
                             'and pass its output directory as artifact_dir.') from exc
    with open(feature_names_path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)
        
    feature_values = [sample_row[feat] for feat in feature_names]
    if not all(finite_number(v) for v in feature_values):
        raise ValueError('Model inputs must be finite numeric feature values')
    X_raw = pd.DataFrame([feature_values], columns=feature_names)
    X_scaled = scaler.transform(X_raw)
    
    probs = clf.predict_proba(X_scaled)[0]
    pred_idx = np.argmax(probs)
    confidence = float(probs[pred_idx])
    
    metadata_path = artifact_dir / 'model_metadata.json'
    class_names = json.loads(metadata_path.read_text())['class_names'] if metadata_path.exists() else ALL_CLASSES
    if len(class_names) != len(probs) or list(clf.classes_) != list(range(len(class_names))):
        raise ValueError('Model class mapping does not match classifier outputs')
    predicted_class = class_names[pred_idx]
    
    kb_entry = KNOWLEDGE_BASE.get(predicted_class)
    if not kb_entry:
        return {
            "predicted": predicted_class,
            "confidence": confidence,
            "true_label": true_label,
            "claims": [],
            "per_claim": [],
            "faithfulness": 0.0,
            "class_consistency": 0.0,
            "mechanism_warnings": [],
            "atypical_features": [],
            "gate_decision": "ESCALATE",
            "reason": f"Class '{predicted_class}' has no grounding entry in KNOWLEDGE_BASE",
            "explanation": "",
            "candidate_explanation": "",
            "gate_policy": STRICT_POLICY
        }
        
    glossary_path = artifact_dir / "feature_glossary.json"
    with open(glossary_path, "r", encoding="utf-8") as f:
        glossary = json.load(f)
        
    grounding_feats = kb_entry.get("grounding_features", {})
    # Validate prompt evidence before generation; never substitute missing bounds with zero.
    for feat in grounding_feats:
        sample_direction(sample_row.get(feat), thresholds.get('global', {}).get(feat))
        if not finite_number(thresholds['global'][feat].get('p50')):
            raise ValueError(f'Missing/invalid median for {feat}')
    if grounding_mode == "class_only":
        context_lines = ["feature | human label + freq band", "---|---"]
        for feat in sorted(grounding_feats.keys()):
            label = glossary.get(feat, {}).get("label", feat)
            context_lines.append(f"{feat} | {label}")
        context_str = "\n".join(context_lines)
    elif grounding_mode == "values_only":
        context_lines = ["feature | human label + freq band | this sample's value | global p25 | global p75", "---|---|---|---|---"]
        for feat in sorted(grounding_feats.keys()):
            actual_val = sample_row[feat]
            label = glossary.get(feat, {}).get("label", feat)
            global_p25 = thresholds["global"][feat]["p25"]
            global_p75 = thresholds["global"][feat]["p75"]
            context_lines.append(f"{feat} | {label} | {actual_val:.4f} | {global_p25:.4f} | {global_p75:.4f}")
        context_str = "\n".join(context_lines)
    else:
        context_lines = ["feature | human label + freq band | this sample's value | global p25 | p50 | p75 | sample_dir", "---|---|---|---|---|---|---"]
        for feat in sorted(grounding_feats.keys()):
            actual_val = sample_row[feat]
            label = glossary.get(feat, {}).get("label", feat)
            global_p50 = thresholds["global"][feat]["p50"]
            global_p25 = thresholds["global"][feat]["p25"]
            global_p75 = thresholds["global"][feat]["p75"]
            
            if actual_val >= global_p75:
                sample_dir = "high"
            elif actual_val <= global_p25:
                sample_dir = "low"
            else:
                sample_dir = "normal"
                
            context_lines.append(f"{feat} | {label} | {actual_val:.4f} | {global_p25:.4f} | {global_p50:.4f} | {global_p75:.4f} | {sample_dir}")
        context_str = "\n".join(context_lines)
        
    parsed_json = None
    unparseable = False
    
    if mock:
        fixture_key = mock if isinstance(mock, str) else "correct"
        parsed_json = MOCK_FIXTURES[fixture_key]
    else:
        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key or api_key in ["dummy_key", "your_nvidia_key_here"]:
            raise ValueError("NVIDIA_API_KEY is not set. Run in mock mode or update .env file.")
            
        if grounding_mode == "class_only":
            system_prompt = (
                "You are an expert diagnostic assistant for air compressor maintenance.\n"
                "Your task is to generate a technician-facing explanation and a structured list of claims based ONLY on the provided grounding features list.\n\n"
                "Strict Rules:\n"
                "1. ONLY make claims about features present in the grounding list. Any other feature will fail verification.\n"
                "2. For each claim, specify:\n"
                "   - 'feature': exact canonical feature name (e.g. 'spectral_rolloff_85').\n"
                "   - 'direction': 'high' or 'low'.\n"
                "   - 'action_critical': true if this feature is critical for the diagnosis/action, false otherwise.\n"
                "3. State each feature's direction from your own general class knowledge of the predicted class (i.e. whether it is typically elevated or low during the fault).\n"
                "4. In the 'explanation' string:\n"
                "   - Write a plain-English explanation for a technician (maximum 120 words).\n"
                "   - Reference frequency bands or human descriptions (e.g. 'energy near 1750-1875 Hz' or 'frequency below which 85% of FFT magnitude lies'), NEVER use raw canonical names (like 'wpt_energy_15' or 'spectral_rolloff_85') in the explanation prose.\n"
                "   - Use ONLY the provided Physics Mechanism description for causal wording; do NOT introduce new physical mechanisms or adjectives (e.g., do not call the signal 'impulsive' unless the mechanism text says so).\n"
                "5. Output the structured claims list containing the features you used.\n"
                "6. Output MUST be a single, strict JSON object. No Markdown wrappers except optionally standard json block:\n"
                "{\n"
                '  "claims": [\n'
                '    {"feature": "canonical_name", "direction": "high|low", "action_critical": true/false}\n'
                '  ],\n'
                '  "explanation": "Technician explanation..."\n'
                "}"
            )
            prompt = (
                f"Predicted fault class: {predicted_class}\n"
                f"Physics Mechanism VERBATIM: {kb_entry.get('mechanism')}\n\n"
                f"Grounding Features list:\n{context_str}\n\n"
                "Generate the JSON response matching the schema above."
            )
        elif grounding_mode == "values_only":
            system_prompt = (
                "You are an expert diagnostic assistant for air compressor maintenance.\n"
                "Your task is to generate a technician-facing explanation and a structured list of claims based ONLY on the provided grounding features table.\n\n"
                "Strict Rules:\n"
                "1. ONLY make claims about features present in the grounding table. Any other feature will fail verification.\n"
                "2. For each claim, specify:\n"
                "   - 'feature': exact canonical feature name (e.g. 'spectral_rolloff_85').\n"
                "   - 'direction': 'high' or 'low'.\n"
                "   - 'action_critical': true if this feature is critical for the diagnosis/action, false otherwise.\n"
                "3. Reason about each feature's direction: compare each feature's value to the provided p25 and p75 thresholds. State whether it is high (above p75), low (below p25), or within normal range (between p25 and p75).\n"
                "4. In the 'explanation' string:\n"
                "   - Write a plain-English explanation for a technician (maximum 120 words).\n"
                "   - Reference frequency bands or human descriptions (e.g. 'energy near 1750-1875 Hz' or 'frequency below which 85% of FFT magnitude lies'), NEVER use raw canonical names (like 'wpt_energy_15' or 'spectral_rolloff_85') in the explanation prose.\n"
                "   - Explain the diagnosis using ONLY features you determine to be 'high' or 'low'. For 'normal' features, do NOT assert any direction (say 'within normal range' or omit). Do NOT mention any feature not in this table.\n"
                "   - Use ONLY the provided Physics Mechanism description for causal wording; do NOT introduce new physical mechanisms or adjectives (e.g., do not call the signal 'impulsive' unless the mechanism text says so).\n"
                "5. Output the structured claims list containing ONLY the high/low features you used.\n"
                "6. Output MUST be a single, strict JSON object. No Markdown wrappers except optionally standard json block:\n"
                "{\n"
                '  "claims": [\n'
                '    {"feature": "canonical_name", "direction": "high|low", "action_critical": true/false}\n'
                '  ],\n'
                '  "explanation": "Technician explanation..."\n'
                "}"
            )
            prompt = (
                f"Predicted fault class: {predicted_class}\n"
                f"Physics Mechanism VERBATIM: {kb_entry.get('mechanism')}\n\n"
                f"Grounding Context Table:\n{context_str}\n\n"
                "Generate the JSON response matching the schema above."
            )
        else:
            system_prompt = (
                "You are an expert diagnostic assistant for air compressor maintenance.\n"
                "Your task is to generate a technician-facing explanation and a structured list of claims based ONLY on the provided grounding context table.\n\n"
                "Strict Rules:\n"
                "1. ONLY make claims about features present in the grounding context table. Any other feature will fail verification.\n"
                "2. For each claim, specify:\n"
                "   - 'feature': exact canonical feature name (e.g. 'spectral_rolloff_85').\n"
                "   - 'direction': 'high' or 'low'.\n"
                "   - 'action_critical': true if this feature is critical for the diagnosis/action, false otherwise.\n"
                "3. In the 'explanation' string:\n"
                "   - Write a plain-English explanation for a technician (maximum 120 words).\n"
                "   - Reference frequency bands or human descriptions (e.g. 'energy near 1750-1875 Hz' or 'frequency below which 85% of FFT magnitude lies'), NEVER use raw canonical names (like 'wpt_energy_15' or 'spectral_rolloff_85') in the explanation prose.\n"
                "   - Explain the diagnosis using ONLY features whose sample_dir is 'high' or 'low'. For 'normal' features, do NOT assert any direction (say 'within normal range' or omit). Do NOT mention any feature not in this table.\n"
                "   - Use ONLY the provided Physics Mechanism description for causal wording; do NOT introduce new physical mechanisms or adjectives (e.g., do not call the signal 'impulsive' unless the mechanism text says so).\n"
                "4. Output the structured claims list containing ONLY the high/low features you used.\n"
                "5. Output MUST be a single, strict JSON object. No Markdown wrappers except optionally standard json block:\n"
                "{\n"
                '  "claims": [\n'
                '    {"feature": "canonical_name", "direction": "high|low", "action_critical": true/false}\n'
                '  ],\n'
                '  "explanation": "Technician explanation..."\n'
                "}"
            )
            prompt = (
                f"Predicted fault class: {predicted_class}\n"
                f"Physics Mechanism VERBATIM: {kb_entry.get('mechanism')}\n\n"
                f"Grounding Context Table:\n{context_str}\n\n"
                "Generate the JSON response matching the schema above."
            )
        
        response_text = llm_client.generate(prompt, system=system_prompt)
        
        try:
            parsed_json = parse_response(response_text)
        except ValueError:
            unparseable = True
            parsed_json = {"claims": [], "explanation": response_text}

    schema_error = None
    if not unparseable:
        try:
            validate_response(parsed_json)
        except ValueError as exc:
            schema_error = str(exc)
            
    if unparseable or schema_error:
        return {
            "predicted": predicted_class,
            "confidence": confidence,
            "true_label": true_label,
            "claims": [],
            "per_claim": [],
            "faithfulness": 0.0,
            "class_consistency": 0.0,
            "gate_decision": "ESCALATE",
            "reason": "unparseable_llm_output" if unparseable else "invalid_llm_schema",
            "validation_error": schema_error,
            "explanation": "",
            "candidate_explanation": parsed_json.get('explanation', '') if isinstance(parsed_json, dict) else '',
            "gate_policy": STRICT_POLICY,
            "mechanism_warnings": []
        }
        
    claims_list = parsed_json.get("claims", [])
    verifier_claims = []
    for c in claims_list:
        vc = dict(c)
        vc["critical"] = vc.get("action_critical", False)
        verifier_claims.append(vc)
        
    res = check_sample_faithfulness(sample_row, predicted_class, verifier_claims, thresholds=thresholds, calibration=calibration)
    
    explanation_text = parsed_json.get("explanation", "")
    mechanism_warnings = check_mechanism_words(explanation_text, predicted_class, thresholds)
    faithfulness = res["faithfulness"]
    class_consistency = res["class_consistency"]
    per_claim = res["per_claim"]
    gate_decision, reason = decide_gate(per_claim, mechanism_warnings)
    atypical_features = [pc["feature"] for pc in per_claim if pc["atypical"]]

    report = {
        "predicted": predicted_class,
        "confidence": confidence,
        "true_label": true_label,
        "claims": claims_list,
        "per_claim": per_claim,
        "faithfulness": faithfulness,
        "class_consistency": class_consistency,
        "gate_decision": gate_decision,
        "reason": reason,
        "explanation": explanation_text if gate_decision == "RELEASE" else "",
        "candidate_explanation": explanation_text,
        "gate_policy": STRICT_POLICY,
        "mechanism_warnings": mechanism_warnings
    }
    
    if atypical_features and gate_decision == "RELEASE":
        report["atypical_features"] = atypical_features
        
    return report
