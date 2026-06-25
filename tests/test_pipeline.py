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
def liv_sample():
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

def test_pipeline_correct(liv_sample):
    report = run_pipeline(liv_sample, true_label="LIV", mock="correct")
    assert report["gate_decision"] == "RELEASE"
    assert report["faithfulness"] == 1.0
    assert report["reason"] == "passed verification"

def test_pipeline_flipped(liv_sample):
    report = run_pipeline(liv_sample, true_label="LIV", mock="flipped")
    assert report["gate_decision"] == "ESCALATE"
    assert report["faithfulness"] < 0.8
    assert report["reason"] == "low faithfulness score"

def test_pipeline_critical(liv_sample):
    report = run_pipeline(liv_sample, true_label="LIV", mock="critical")
    assert report["gate_decision"] == "ESCALATE"
    assert report["reason"] == "action-critical claim failed"
