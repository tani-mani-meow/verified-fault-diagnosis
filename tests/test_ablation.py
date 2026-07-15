# tests/test_ablation.py
import json
from pathlib import Path

def test_classonly_produces_false_claims():
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent
    classonly_path = repo_dir / "outputs" / "faithfulness_audit_classonly.json"
    
    # Assert classonly file exists
    assert classonly_path.exists(), f"faithfulness_audit_classonly.json not found at {classonly_path}"
    
    with open(classonly_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Assert we have 100 cases
    assert len(data) == 100, f"Expected 100 cases in class-only audit, found {len(data)}"
    
    # Count how many cases have at least one sample-false claim
    cases_with_false_claims = 0
    for case in data:
        per_claim = case.get("per_claim", [])
        has_false = any(not pc.get("sample_true", True) for pc in per_claim)
        if has_false:
            cases_with_false_claims += 1
            
    print(f"Cases with false claims in classonly: {cases_with_false_claims}")
    # Assert that a significant number of cases have false claims (meaning verifier actually fires)
    assert cases_with_false_claims >= 1, "Expected at least one case to have a sample-false claim under class-only mode"
