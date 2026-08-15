import os
import sys
import json
from pathlib import Path
import pytest

# Adjust paths to make sure we can import from src
script_dir = Path(__file__).resolve().parent
repo_dir = script_dir.parent
sys.path.insert(0, str(repo_dir / "src"))

from verifier import verify_claim, verify_explanation
from knowledge_base import KNOWLEDGE_BASE

# Define mock thresholds representing real data-selected features for LIV
MOCK_THRESHOLDS = {
    "global": {
        "spectral_rolloff_85": {"p10": 1000.0, "p25": 1200.0, "p50": 1600.0, "p75": 2000.0, "p90": 2200.0},
        "skewness": {"p10": -0.5, "p25": 0.0, "p50": 0.5, "p75": 1.0, "p90": 1.5},
        "spectral_centroid": {"p10": 400.0, "p25": 600.0, "p50": 800.0, "p75": 1000.0, "p90": 1200.0},
        "wpt_energy_15": {"p10": 0.5, "p25": 1.0, "p50": 2.0, "p75": 3.0, "p90": 4.0},
        "wpt_energy_18": {"p10": 0.5, "p25": 1.0, "p50": 2.0, "p75": 3.0, "p90": 4.0},
    },
    "classes": {
        "LIV": {
            "spectral_rolloff_85": {"p50": 1100.0},
            "skewness": {"p50": -0.2},
            "spectral_centroid": {"p50": 550.0},
            "wpt_energy_15": {"p50": 0.8},
            "wpt_energy_18": {"p50": 1.8},
        }
    },
    "distinctive": {
        "LIV": {
            "top_2": [],
            "bottom_2": ["spectral_rolloff_85"]
        }
    }
}

# Mock calibration aligned with thresholds and new grounding features
MOCK_CALIBRATION = {
    "LIV": {
        "spectral_rolloff_85": {
            "data_direction": "low",
            "kb_direction": "low",
            "match": True
        },
        "skewness": {
            "data_direction": "low",
            "kb_direction": "low",
            "match": True
        },
        "spectral_centroid": {
            "data_direction": "low",
            "kb_direction": "low",
            "match": True
        },
        "wpt_energy_15": {
            "data_direction": "low",
            "kb_direction": "low",
            "match": True
        },
        "wpt_energy_18": {
            "data_direction": "neutral",
            "kb_direction": "unspecified",
            "match": True
        }
    }
}

# Mock LIV recording features matching the low directions
MOCK_LIV_RECORDING = {
    "spectral_rolloff_85": 1000.0,
    "skewness": -0.1,
    "spectral_centroid": 600.0,
    "wpt_energy_15": 1.0,
    "wpt_energy_18": 1.5,
    "rms": 1.2  # Not in LIV grounding features
}

def test_true_claim():
    """
    1. TRUE claim: pick a real LIV recording, assert a claim that its spectral_rolloff_85
       is 'low' -> should PASS.
    """
    claim = {"feature": "spectral_rolloff_85", "direction": "low"}
    res = verify_claim(MOCK_LIV_RECORDING, "LIV", claim, MOCK_THRESHOLDS, MOCK_CALIBRATION)
    assert res["passed"] is True, f"Claim failed: {res['reason']}"
    assert res["actual_value"] == MOCK_LIV_RECORDING["spectral_rolloff_85"]

def test_false_claim():
    """
    2. FALSE claim: same recording, claim skewness is 'elevated' when it is actually low ->
       should FAIL, and if marked action-critical -> escalate=True.
    """
    claim = {"feature": "skewness", "direction": "elevated", "critical": True}
    res = verify_claim(MOCK_LIV_RECORDING, "LIV", claim, MOCK_THRESHOLDS, MOCK_CALIBRATION)
    assert res["passed"] is False
    assert "direction mismatch" in res["reason"]
    
    # Test explanation level escalation for critical claim failure
    exp_res = verify_explanation(MOCK_LIV_RECORDING, "LIV", [claim], thresholds=MOCK_THRESHOLDS, calibration=MOCK_CALIBRATION)
    assert exp_res["escalate"] is True
    assert exp_res["faithfulness_score"] == 0.0

def test_ungrounded_claim():
    """
    3. UNGROUNDED claim: claim a feature NOT in LIV's grounding_features -> should FAIL
       with the 'not grounded' reason.
    """
    claim = {"feature": "rms", "direction": "low"}
    res = verify_claim(MOCK_LIV_RECORDING, "LIV", claim, MOCK_THRESHOLDS, MOCK_CALIBRATION)
    assert res["passed"] is False
    assert res["reason"] == "feature not grounded for this class"

def test_neutral_claim():
    """
    Test that claiming 'low' or 'high' for a data-derived neutral feature fails.
    """
    claim = {"feature": "wpt_energy_18", "direction": "low"}
    res = verify_claim(MOCK_LIV_RECORDING, "LIV", claim, MOCK_THRESHOLDS, MOCK_CALIBRATION)
    assert res["passed"] is False
    assert res["reason"] == "feature not directionally characteristic for this class"

def test_mixed_explanation():
    """
    4. MIXED explanation: 4 true + 1 false claim -> faithfulness_score=0.8, escalate=True
       even if the false claim is non-critical.
    """
    claims_non_critical = [
        {"feature": "spectral_rolloff_85", "direction": "low", "critical": False},
        {"feature": "skewness", "direction": "low", "critical": False},
        {"feature": "spectral_centroid", "direction": "low", "critical": False},
        {"feature": "wpt_energy_18", "direction": "neutral", "critical": False},
        {"feature": "rms", "direction": "low", "critical": False},  # Ungrounded -> Fails
    ]
    
    res = verify_explanation(MOCK_LIV_RECORDING, "LIV", claims_non_critical, thresholds=MOCK_THRESHOLDS, calibration=MOCK_CALIBRATION)
    assert res["faithfulness_score"] == 0.8
    assert res["escalate"] is True
    
    # 3 true + 2 false -> score = 0.6 < 0.8 -> escalate = True
    claims_more_false = [
        {"feature": "spectral_rolloff_85", "direction": "low", "critical": False},
        {"feature": "skewness", "direction": "low", "critical": False},
        {"feature": "spectral_centroid", "direction": "low", "critical": False},
        {"feature": "skewness", "direction": "high", "critical": False},  # Fails
        {"feature": "rms", "direction": "low", "critical": False},  # Ungrounded -> Fails
    ]
    res_esc = verify_explanation(MOCK_LIV_RECORDING, "LIV", claims_more_false, thresholds=MOCK_THRESHOLDS, calibration=MOCK_CALIBRATION)
    assert res_esc["faithfulness_score"] == 0.6
    assert res_esc["escalate"] is True
    
    # 4 true + 1 false (critical) -> score = 0.8, escalate = True
    claims_critical = [
        {"feature": "spectral_rolloff_85", "direction": "low", "critical": False},
        {"feature": "skewness", "direction": "low", "critical": False},
        {"feature": "spectral_centroid", "direction": "low", "critical": False},
        {"feature": "wpt_energy_18", "direction": "neutral", "critical": False},
        {"feature": "rms", "direction": "low", "critical": True},  # Ungrounded and critical -> Fails
    ]
    res_crit = verify_explanation(MOCK_LIV_RECORDING, "LIV", claims_critical, thresholds=MOCK_THRESHOLDS, calibration=MOCK_CALIBRATION)
    assert res_crit["faithfulness_score"] == 0.8
    assert res_crit["escalate"] is True

def test_glossary_completeness():
    """
    Test that every feature in any class's grounding_features in KNOWLEDGE_BASE
    has a corresponding entry in outputs/feature_glossary.json.
    """
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent
    glossary_path = repo_dir / "outputs" / "feature_glossary.json"
    
    assert glossary_path.exists(), "feature_glossary.json does not exist"
    
    with open(glossary_path, "r", encoding="utf-8") as f:
        glossary = json.load(f)
        
    for c, data in KNOWLEDGE_BASE.items():
        grounding_feats = data.get("grounding_features", {})
        for feature in grounding_feats:
            assert feature in glossary, f"Feature '{feature}' from class '{c}' is missing from feature_glossary.json"


def test_zero_claims_escalates():
    """If the LLM produces 0 verifiable claims, faithfulness must be 0.0 and gate must ESCALATE."""
    from verifier import verify_explanation, check_sample_faithfulness

    recording = {"spectral_rolloff_85": 1000.0, "spectral_centroid": 600.0}

    # Test verify_explanation
    result = verify_explanation(
        recording, "LIV", [],
        thresholds=MOCK_THRESHOLDS, calibration=MOCK_CALIBRATION
    )
    assert result["faithfulness_score"] == 0.0, f"Expected 0.0 for 0 claims, got {result['faithfulness_score']}"
    assert result["escalate"] is True, f"Expected escalate=True for 0 claims"

    # Test check_sample_faithfulness
    result2 = check_sample_faithfulness(
        recording, "LIV", [],
        thresholds=MOCK_THRESHOLDS, calibration=MOCK_CALIBRATION
    )
    assert result2["faithfulness"] == 0.0, f"Expected 0.0 for 0 claims, got {result2['faithfulness']}"
    assert result2["class_consistency"] == 0.0, f"Expected 0.0 class_consistency for 0 claims, got {result2['class_consistency']}"

