"""
Investigate the 2 real audit FAILs and confirm the 15 threshold FAILs
are audit-script errors (wrong data subset).
"""

import json, sys, re

import pandas as pd
import numpy as np
from pathlib import Path

repo = Path(__file__).resolve().parent.parent

print("=" * 80)
print("AUDIT FIX INVESTIGATION")
print("=" * 80)

# ============================================================
# PART 1: Confirm thresholds were computed on TRAIN split (not full CSV)
# ============================================================
print("\n--- PART 1: Threshold subset verification ---")

from sklearn.model_selection import train_test_split

df = pd.read_csv(repo / "outputs" / "features.csv")
y_str = df["class"]
class_names = sorted(y_str.unique())
class_to_idx = {name: idx for idx, name in enumerate(class_names)}
y = y_str.map(class_to_idx).values

# Reproduce the exact train/test split
df_train, df_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=42, stratify=y
)

with open(repo / "outputs" / "feature_thresholds.json") as f:
    th = json.load(f)

print(f"Full CSV rows: {len(df)}")
print(f"Train rows: {len(df_train)}")
print(f"Test rows: {len(df_test)}")

# Recompute on TRAIN split only
spot_features = ["kurtosis", "crest_factor", "spectral_centroid", "wpt_energy_4", "band_energy_ratio_4"]
all_match_train = True
all_match_full = True

for feat in spot_features:
    if feat not in df.columns:
        continue
    train_p25 = float(np.percentile(df_train[feat], 25))
    train_p50 = float(np.percentile(df_train[feat], 50))
    train_p75 = float(np.percentile(df_train[feat], 75))
    full_p25 = float(np.percentile(df[feat], 25))
    full_p50 = float(np.percentile(df[feat], 50))
    full_p75 = float(np.percentile(df[feat], 75))
    stored_p25 = th["global"].get(feat, {}).get("p25", None)
    stored_p50 = th["global"].get(feat, {}).get("p50", None)
    stored_p75 = th["global"].get(feat, {}).get("p75", None)

    train_match = (abs(train_p25 - stored_p25) < 0.001 and
                   abs(train_p50 - stored_p50) < 0.001 and
                   abs(train_p75 - stored_p75) < 0.001)
    full_match = (abs(full_p25 - stored_p25) < 0.001 and
                  abs(full_p50 - stored_p50) < 0.001 and
                  abs(full_p75 - stored_p75) < 0.001)

    print(f"\n  {feat}:")
    print(f"    TRAIN: p25={train_p25:.4f} p50={train_p50:.4f} p75={train_p75:.4f} -> match={train_match}")
    print(f"    FULL:  p25={full_p25:.4f} p50={full_p50:.4f} p75={full_p75:.4f} -> match={full_match}")
    print(f"    STORED: p25={stored_p25:.4f} p50={stored_p50:.4f} p75={stored_p75:.4f}")
    if not train_match:
        all_match_train = False
    if not full_match:
        all_match_full = False

print(f"\n   *** VERDICT: thresholds match TRAIN split: {all_match_train} ***")
print(f"   *** VERDICT: thresholds match FULL CSV: {all_match_full} ***")
if all_match_train and not all_match_full:
    print("   -> CONFIRMED: 15 threshold FAILs were audit-script errors (checked wrong subset)")
    print("   -> The pipeline is CORRECT: thresholds computed on train split to prevent test leakage")

# Also check class-level p50s on train split
print("\n   Class-level p50 verification (train split):")
classes = ["LIV", "LOV", "NRV", "Bearing", "Flywheel", "Piston", "Riderbelt"]
class_all_match = True
for c in classes:
    train_class = df_train[df_train["class"] == c]
    for feat in ["kurtosis", "crest_factor"]:
        train_p50 = float(np.percentile(train_class[feat], 50))
        stored_p50 = th["classes"].get(c, {}).get(feat, {}).get("p50", None)
        match = abs(train_p50 - stored_p50) < 0.01 if stored_p50 is not None else False
        if not match:
            class_all_match = False
            print(f"    {c}/{feat}: train={train_p50:.4f} stored={stored_p50:.4f} MISMATCH")
if class_all_match:
    print("    All class-level p50s match train split -> 11 class FAILs confirmed audit-script errors")

# ============================================================
# PART 2: Investigate case 993 (empty per_claim, faithfulness=1.0)
# ============================================================
print("\n--- PART 2: Case 993 empty per_claim investigation ---")

with open(repo / "outputs" / "faithfulness_audit.json") as f:
    audit = json.load(f)

case_993 = [c for c in audit if c["index"] == 993]
if case_993:
    c = case_993[0]
    print(f"  index: {c['index']}")
    print(f"  true_label: {c['true_label']}")
    print(f"  predicted: {c['predicted']}")
    print(f"  correct: {c['correct']}")
    print(f"  faithfulness: {c['faithfulness']}")
    print(f"  n_claims: {c.get('n_claims', 'N/A')}")
    print(f"  gate_decision: {c['gate_decision']}")
    print(f"  gate_reason: {c['gate_reason']}")
    print(f"  per_claim: {c.get('per_claim', 'MISSING')}")
    print(f"  explanation: {c.get('explanation', 'MISSING')[:500]}")
    print(f"  class_consistency: {c.get('class_consistency', 'N/A')}")

    # The key question: is faithfulness=1.0 when n_claims=0 a bug?
    n_claims = c.get("n_claims", 0)
    per_claims = c.get("per_claim", [])
    if n_claims == 0 or len(per_claims) == 0:
        print("\n   *** BUG CONFIRMED: faithfulness=1.0 with 0 claims is vacuously true ***")
        print("   *** FIX NEEDED: if n_claims=0, faithfulness should be 0.0 or gate should ESCALATE ***")
        print("   *** This case should be ESCALATE with reason='no_verifiable_claims' ***")
    else:
        print("  Case 993 has per_claim data — audit script may have had a loading issue")
else:
    print("  Case 993 not found in audit JSON")

# Check if there are other cases with 0 claims
zero_claim_cases = [c for c in audit if c.get("n_claims", 0) == 0 or len(c.get("per_claim", [])) == 0]
print(f"\n  Total cases with 0 claims: {len(zero_claim_cases)}")
for zc in zero_claim_cases:
    print(f"    index={zc['index']}, true={zc['true_label']}, faith={zc['faithfulness']}, gate={zc['gate_decision']}")

# ============================================================
# PART 3: Investigate escalation field name mismatch
# ============================================================
print("\n--- PART 3: Escalation field name investigation ---")
escalated = [c for c in audit if c["gate_decision"] == "ESCALATE"]
print(f"  Total escalations: {len(escalated)}")

for esc in escalated:
    idx = esc["index"]
    reason = esc["gate_reason"]
    per_claims = esc.get("per_claim", [])
    print(f"\n  Case {idx} ({esc['true_label']} -> {esc['predicted']}):")
    print(f"    gate_reason: {reason}")
    print(f"    n_claims: {esc.get('n_claims', 'N/A')}")
    print(f"    n_atypical: {esc.get('n_atypical', 'N/A')}")
    print(f"    faithfulness: {esc['faithfulness']:.4f}")
    if per_claims:
        # Print ALL keys in the first per_claim entry
        print(f"    per_claim[0] keys: {list(per_claims[0].keys())}")
        # Print all per_claims with their atypical/critical fields
        for i, pc in enumerate(per_claims):
            print(f"    claim[{i}]: feature={pc.get('feature','?')}, "
                  f"claimed_dir={pc.get('claimed_dir','?')}, sample_dir={pc.get('sample_dir','?')}, "
                  f"sample_true={pc.get('sample_true','?')}, class_dir={pc.get('class_dir','?')}, "
                  f"class_consistent={pc.get('class_consistent','?')}, "
                  f"atypical={pc.get('atypical','?')}, critical={pc.get('critical','?')}, "
                  f"action_critical={pc.get('action_critical','?')}")
    else:
        print(f"    NO per_claim data!")

    # Check: which field name is used for "critical"?
    if per_claims:
        possible_critical_fields = ["critical", "action_critical", "is_critical", "is_action_critical", "atypical_action_critical"]
        for field in possible_critical_fields:
            values = [pc.get(field, "MISSING") for pc in per_claims]
            print(f"    field '{field}' values: {values}")

# ============================================================
# PART 4: Check the verifier source code for the actual field names
# ============================================================
print("\n--- PART 4: Verifier source code field names ---")

verifier_path = repo / "src" / "verifier.py"
with open(verifier_path, "r") as f:
    vsrc = f.read()

# Find lines that build per_claim dicts
for i, line in enumerate(vsrc.split("\n"), 1):
    if "critical" in line.lower() or "action" in line.lower():
        print(f"  verifier.py:{i}: {line.strip()}")

# Also check pipeline.py for the escalation condition
print("\n  --- pipeline.py escalation logic ---")
pipeline_path = repo / "src" / "pipeline.py"
with open(pipeline_path, "r") as f:
    psrc = f.read()
for i, line in enumerate(psrc.split("\n"), 1):
    if "atypical" in line.lower() and ("critical" in line.lower() or "action" in line.lower() or "escalat" in line.lower()):
        print(f"  pipeline.py:{i}: {line.strip()}")

print("\n" + "=" * 80)
print("INVESTIGATION COMPLETE")
print("=" * 80)
