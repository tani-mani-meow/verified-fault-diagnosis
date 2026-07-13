# tests/test_audit.py
import json
from pathlib import Path

def test_audit_summary_exists_and_valid():
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent
    summary_path = repo_dir / "outputs" / "faithfulness_audit_summary.json"
    
    # Assert summary file exists
    assert summary_path.exists(), f"faithfulness_audit_summary.json not found at {summary_path}"
    
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
        
    # Assert required top-level keys exist
    required_keys = ["faithfulness", "gate_outcomes", "safety_cross_tab", "per_class", "mechanism_warnings_rate_pct", "cost_latency"]
    for key in required_keys:
        assert key in summary, f"Required key '{key}' not found in audit summary"
        
    # Assert RELEASE% + ESCALATE% == 100
    release_pct = summary["gate_outcomes"].get("release_pct", 0.0)
    escalate_pct = summary["gate_outcomes"].get("escalate_pct", 0.0)
    
    # Allow small float imprecision (e.g. 99.99999)
    assert abs((release_pct + escalate_pct) - 100.0) < 1e-5, f"RELEASE% ({release_pct}) + ESCALATE% ({escalate_pct}) is not 100%"
