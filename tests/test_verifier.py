import os
import sys
from pathlib import Path
import pytest

# Adjust paths to make sure we can import from src
script_dir = Path(__file__).resolve().parent
repo_dir = script_dir.parent
sys.path.insert(0, str(repo_dir / "src"))

from verifier import verify_claim, verify_explanation
from knowledge_base import KNOWLEDGE_BASE

# Define mock thresholds to ensure tests are deterministic and independent of dataset content
MOCK_THRESHOLDS = {
    "global": {
        "kurtosis": {"p10": 1.0, "p25": 1.5, "p50": 2.0, "p75": 3.0, "p90": 4.0},
        "crest_factor": {"p10": 2.0, "p25": 3.0, "p50": 4.0, "p75": 5.0, "p90": 6.0},
        "band_energy_ratio_2": {"p10": 0.1, "p25": 0.2, "p50": 0.3, "p75": 0.4, "p90": 0.5},
        "band_energy_ratio_3": {"p10": 0.1, "p25": 0.2, "p50": 0.3, "p75": 0.4, "p90": 0.5},
        "zero_crossing_rate": {"p10": 0.05, "p25": 0.1, "p50": 0.15, "p75": 0.2, "p90": 0.25},
    },
    "classes": {
        "LIV": {
            "kurtosis": {"p50": 3.5},
            "crest_factor": {"p50": 5.5},
            "band_energy_ratio_2": {"p50": 0.45},
            "band_energy_ratio_3": {"p50": 0.45},
            "zero_crossing_rate": {"p50": 0.18},
        }
    },
    "distinctive": {
        "LIV": {
            "top_2": ["kurtosis"],
            "bottom_2": []
        }
    }
}

# Mock calibration aligned with thresholds
MOCK_CALIBRATION = {
    "LIV": {
        "kurtosis": {
            "data_direction": "high",
            "kb_direction": "high",
            "match": True
        },
        "crest_factor": {
            "data_direction": "high",
            "kb_direction": "high",
            "match": True
        },
        "band_energy_ratio_2": {
            "data_direction": "high",
            "kb_direction": "high",
            "match": True
        },
        "band_energy_ratio_3": {
            "data_direction": "high",
            "kb_direction": "high",
            "match": True
        },
        "zero_crossing_rate": {
            "data_direction": "neutral",
            "kb_direction": "unspecified",
            "match": True
        }
    }
}

MOCK_LIV_RECORDING = {
    "kurtosis": 4.5,
    "crest_factor": 6.0,
    "band_energy_ratio_2": 0.8,
    "band_energy_ratio_3": 0.8,
    "zero_crossing_rate": 0.2,
    "rms": 1.2  # Not in LIV grounding features
}

def test_true_claim():
    """
    1. TRUE claim: pick a real LIV recording, assert a claim that its band_energy_ratio_3
       is 'elevated' (high) -> should PASS.
    """
    claim = {"feature": "band_energy_ratio_3", "direction": "elevated"}
    res = verify_claim(MOCK_LIV_RECORDING, "LIV", claim, MOCK_THRESHOLDS, MOCK_CALIBRATION)
    assert res["passed"] is True, f"Claim failed: {res['reason']}"
    assert res["actual_value"] == MOCK_LIV_RECORDING["band_energy_ratio_3"]

def test_false_claim():
    """
    2. FALSE claim: same recording, claim kurtosis is 'low' when it is actually high ->
       should FAIL, and if marked action-critical -> escalate=True.
    """
    claim = {"feature": "kurtosis", "direction": "low", "critical": True}
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
    claim = {"feature": "rms", "direction": "elevated"}
    res = verify_claim(MOCK_LIV_RECORDING, "LIV", claim, MOCK_THRESHOLDS, MOCK_CALIBRATION)
    assert res["passed"] is False
    assert res["reason"] == "feature not grounded for this class"

def test_neutral_claim():
    """
    Test that claiming 'elevated' (high) or 'low' for a data-derived neutral feature fails.
    """
    claim = {"feature": "zero_crossing_rate", "direction": "elevated"}
    res = verify_claim(MOCK_LIV_RECORDING, "LIV", claim, MOCK_THRESHOLDS, MOCK_CALIBRATION)
    assert res["passed"] is False
    assert res["reason"] == "feature not directionally characteristic for this class"

def test_mixed_explanation():
    """
    4. MIXED explanation: 4 true + 1 false claim -> faithfulness_score=0.8, escalate=False
       if the false claim is non-critical.
    """
    claims_non_critical = [
        {"feature": "kurtosis", "direction": "elevated", "critical": False},
        {"feature": "crest_factor", "direction": "elevated", "critical": False},
        {"feature": "band_energy_ratio_2", "direction": "elevated", "critical": False},
        {"feature": "zero_crossing_rate", "direction": "context_dependent", "critical": False},
        {"feature": "rms", "direction": "elevated", "critical": False},  # Ungrounded -> Fails
    ]
    
    res = verify_explanation(MOCK_LIV_RECORDING, "LIV", claims_non_critical, thresholds=MOCK_THRESHOLDS, calibration=MOCK_CALIBRATION)
    assert res["faithfulness_score"] == 0.8
    assert res["escalate"] is False
    
    # 3 true + 2 false -> score = 0.6 < 0.8 -> escalate = True
    claims_more_false = [
        {"feature": "kurtosis", "direction": "elevated", "critical": False},
        {"feature": "crest_factor", "direction": "elevated", "critical": False},
        {"feature": "band_energy_ratio_2", "direction": "elevated", "critical": False},
        {"feature": "kurtosis", "direction": "low", "critical": False},  # Fails
        {"feature": "rms", "direction": "elevated", "critical": False},  # Ungrounded -> Fails
    ]
    res_esc = verify_explanation(MOCK_LIV_RECORDING, "LIV", claims_more_false, thresholds=MOCK_THRESHOLDS, calibration=MOCK_CALIBRATION)
    assert res_esc["faithfulness_score"] == 0.6
    assert res_esc["escalate"] is True
    
    # 4 true + 1 false (critical) -> score = 0.8, escalate = True
    claims_critical = [
        {"feature": "kurtosis", "direction": "elevated", "critical": False},
        {"feature": "crest_factor", "direction": "elevated", "critical": False},
        {"feature": "band_energy_ratio_2", "direction": "elevated", "critical": False},
        {"feature": "zero_crossing_rate", "direction": "context_dependent", "critical": False},
        {"feature": "rms", "direction": "elevated", "critical": True},  # Ungrounded and critical -> Fails
    ]
    res_crit = verify_explanation(MOCK_LIV_RECORDING, "LIV", claims_critical, thresholds=MOCK_THRESHOLDS, calibration=MOCK_CALIBRATION)
    assert res_crit["faithfulness_score"] == 0.8
    assert res_crit["escalate"] is True
