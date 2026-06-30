import os
import sys
import json
from pathlib import Path
import pytest
import pandas as pd
from sklearn.model_selection import train_test_split

# Adjust paths to make sure we can import from src
script_dir = Path(__file__).resolve().parent
repo_dir = script_dir.parent
sys.path.insert(0, str(repo_dir / "src"))

from pipeline import run_pipeline

@pytest.fixture
def base_liv_sample():
    features_csv = repo_dir / "outputs" / "features.csv"
    assert features_csv.exists(), f"features.csv not found at {features_csv}"
    
    df = pd.read_csv(features_csv)
    y_str = df['class']
    class_names = sorted(y_str.unique())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    y = y_str.map(class_to_idx).values
    
    # 80/20 train/test split, random_state=42
    _, df_test, _, _ = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )
    
    liv_subset = df_test[df_test['class'] == 'LIV']
    return liv_subset.iloc[0].to_dict()

def test_pipeline_correct(base_liv_sample):
    # For a correct explanation with only sample-true features:
    # spectral_rolloff_85 and spectral_centroid are both actually low (<= p25) in this sample
    report = run_pipeline(base_liv_sample, true_label="LIV", mock="correct")
    assert report["gate_decision"] == "RELEASE"
    assert report["faithfulness"] == 1.0
    assert report["reason"] == "passed_verification"

def test_pipeline_flipped(base_liv_sample):
    # Flipped normal claim: skewness is claimed low but is actually neutral in the sample
    report = run_pipeline(base_liv_sample, true_label="LIV", mock="flipped")
    assert report["gate_decision"] == "ESCALATE"
    assert report["faithfulness"] < 0.8
    assert report["reason"] == "low_faithfulness"

def test_pipeline_false_critical(base_liv_sample):
    # False action-critical claim: skewness is claimed low (which is sample-false) and critical
    report = run_pipeline(base_liv_sample, true_label="LIV", mock="critical")
    assert report["gate_decision"] == "ESCALATE"
    assert report["reason"] == "false_action_critical"

def test_pipeline_atypical_non_critical(base_liv_sample):
    # Atypical non-critical claim: skewness value is modified to be 1.0 (which is sample-true high)
    # but class calibrated direction for skewness is low. So it is atypical but non-critical.
    sample = dict(base_liv_sample)
    sample["skewness"] = 1.0  # Force it to be high (global_p75 is around -0.0005)
    
    report = run_pipeline(sample, true_label="LIV", mock="atypical_non_critical")
    assert report["gate_decision"] == "RELEASE"
    assert report["reason"] == "passed_verification"
    assert "skewness" in report["atypical_features"]

def test_pipeline_atypical_critical(base_liv_sample):
    # Atypical action-critical claim: skewness is modified to be 1.0 (sample-true high)
    # but class calibration is low, and the claim is marked critical.
    sample = dict(base_liv_sample)
    sample["skewness"] = 1.0  # Force it to be high
    
    report = run_pipeline(sample, true_label="LIV", mock="atypical_critical")
    assert report["gate_decision"] == "ESCALATE"
    assert report["reason"] == "atypical_action_critical"
