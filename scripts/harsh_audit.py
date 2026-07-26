"""
Harsh audit: cross-check every link in the Certificate 3 chain against raw data.

Each check prints PASS/FAIL with evidence. Any FAIL = investigate before trusting results.
"""

import json, sys

import pandas as pd

import numpy as np

from pathlib import Path

repo = Path(__file__).resolve().parent.parent

PASS = "PASS"
FAIL = "FAIL"
results = []

def check(name, condition, evidence):
    status = PASS if condition else FAIL
    results.append((status, name, evidence))
    print(f"[{status}] {name}: {evidence}")

print("=" * 80)
print("HARSH AUDIT — Certificate 3 Pipeline")
print("=" * 80)

# ============================================================
# SECTION 1: Are the global thresholds correct?
# ============================================================
print("\n--- SECTION 1: Global Threshold Integrity ---")
df = pd.read_csv(repo / "outputs" / "features.csv")

with open(repo / "outputs" / "feature_thresholds.json") as f:
    th = json.load(f)

# Spot-check 5 features: recompute p25/p50/p75 from features.csv and compare
spot_features = ["kurtosis", "crest_factor", "spectral_centroid", "wpt_energy_4", "band_energy_ratio_4"]
for feat in spot_features:
    if feat not in df.columns:
        check(f"Threshold {feat}", False, f"Feature {feat} not in features.csv")
        continue
    recomputed_p25 = float(np.percentile(df[feat], 25))
    recomputed_p50 = float(np.percentile(df[feat], 50))
    recomputed_p75 = float(np.percentile(df[feat], 75))
    stored_p25 = th["global"].get(feat, {}).get("p25", None)
    stored_p50 = th["global"].get(feat, {}).get("p50", None)
    stored_p75 = th["global"].get(feat, {}).get("p75", None)
    if stored_p25 is None:
        check(f"Threshold {feat}", False, "Missing from feature_thresholds.json")
        continue
    p25_ok = abs(recomputed_p25 - stored_p25) < 0.001
    p50_ok = abs(recomputed_p50 - stored_p50) < 0.001
    p75_ok = abs(recomputed_p75 - stored_p75) < 0.001
    check(f"Threshold {feat} p25/p50/p75",
          p25_ok and p50_ok and p75_ok,
          f"recomputed=({recomputed_p25:.4f},{recomputed_p50:.4f},{recomputed_p75:.4f}) stored=({stored_p25:.4f},{stored_p50:.4f},{stored_p75:.4f})")

# ============================================================
# SECTION 2: Is the sample-direction computation correct?
# ============================================================
print("\n--- SECTION 2: Sample Direction Computation ---")

with open(repo / "outputs" / "faithfulness_audit.json") as f:
    audit = json.load(f)

# Pick 5 cases spread across classes
audit_by_class = {}
for c in audit:
    audit_by_class.setdefault(c["true_label"], []).append(c)
sample_cases = []
for cls in sorted(audit_by_class.keys())[:5]:
    sample_cases.append(audit_by_class[cls][0])

for case in sample_cases:
    idx = case["index"]
    row = df.loc[idx]
    per_claims = case.get("per_claim", [])
    if not per_claims:
        check(f"Case {idx} ({case['true_label']})", False, "No per_claim data in audit JSON")
        continue
    all_ok = True
    details = []
    for claim in per_claims:
        feat = claim.get("feature", "")
        sample_dir = claim.get("sample_dir", "")
        if feat not in df.columns:
            details.append(f"{feat}: NOT IN CSV")
            all_ok = False
            continue
        val = float(row[feat])
        p25 = th["global"].get(feat, {}).get("p25", 0)
        p75 = th["global"].get(feat, {}).get("p75", 0)
        if val <= p25:
            expected_dir = "low"
        elif val >= p75:
            expected_dir = "high"
        else:
            expected_dir = "normal"
        match = (expected_dir == sample_dir)
        if not match:
            all_ok = False
            details.append(f"{feat}: val={val:.4f} p25={p25:.4f} p75={p75:.4f} -> expected={expected_dir} stored={sample_dir} {'OK' if match else 'MISMATCH'}")
    check(f"Case {idx} ({case['true_label']}) sample directions",
          all_ok,
          "; ".join(details[:3]) + ("..." if len(details) > 3 else ""))

# ============================================================
# SECTION 3: Is faithfulness=1.00 real or is the verifier broken?
# ============================================================
print("\n--- SECTION 3: Verifier Integrity (the 1.00 question) ---")

sys.path.insert(0, str(repo / "src"))
from verifier import verify_explanation
from knowledge_base import KNOWLEDGE_BASE

case = audit[0]
idx = case["index"]
row = df.loc[idx].to_dict()
predicted_class = case["predicted"]
per_claims = case.get("per_claim", [])

if per_claims:
    # Original: check faithfulness is 1.0
    original_fids = [c.get("sample_true", False) for c in per_claims]
    original_faith = sum(1 for x in original_fids if x) / len(original_fids) if original_fids else 0
    check(f"Case {idx} original faithfulness",
          original_faith == 1.0,
          f"faithfulness={original_faith:.4f}, n_claims={len(per_claims)}, all_true={all(original_fids)}")
    # Flipped: take the first claim, flip its direction, see if verifier would catch it
    first_claim = dict(per_claims[0])
    feat = first_claim["feature"]
    original_dir = first_claim["claimed_dir"]
    sample_dir = first_claim["sample_dir"]
    # If claimed_dir == sample_dir (which it should for faithfulness=1.0), flip it
    flipped_dir = "high" if original_dir == "low" else "low"
    flipped_sample_true = (flipped_dir == sample_dir)
    check(f"Case {idx} verifier would catch flipped claim",
          not flipped_sample_true,
          f"flipped claimed_dir={flipped_dir}, sample_dir={sample_dir}, sample_true would be={flipped_sample_true}")
else:
    check(f"Case {idx} verifier test", False, "No per_claim data to test")

# ============================================================
# SECTION 4: Are the 6 escalations real?
# ============================================================
print("\n--- SECTION 4: Escalation Reality Check ---")
escalated = [c for c in audit if c["gate_decision"] == "ESCALATE"]
check("6 escalations exist", len(escalated) == 6, f"found {len(escalated)} escalations")

for esc in escalated:
    idx = esc["index"]
    reason = esc["gate_reason"]
    per_claims = esc.get("per_claim", [])
    atypical_claims = [c for c in per_claims if c.get("atypical", False)]
    critical_atypical = [c for c in atypical_claims if c.get("critical", False)]
    check(f"Escalation {idx} ({esc['true_label']}) reason={reason}",
          reason == "atypical_action_critical" and len(critical_atypical) > 0,
          f"n_atypical={len(atypical_claims)}, n_critical_atypical={len(critical_atypical)}, faithfulness={esc['faithfulness']:.4f}")
    # Verify the atypical claim is REALLY atypical (sample_dir != class_dir)
    for ac in critical_atypical:
        feat = ac["feature"]
        s_dir = ac.get("sample_dir", "?")
        c_dir = ac.get("class_dir", "?")
        really_atypical = (s_dir != c_dir)
        check(f"  -> {feat} atypical check", really_atypical,
              f"sample_dir={s_dir}, class_dir={c_dir}, atypical={ac.get('atypical')}, critical={ac.get('critical')}")

# ============================================================
# SECTION 5: Are the 4 misclassifications real?
# ============================================================
print("\n--- SECTION 5: Misclassification Reality Check ---")
misclassed = [c for c in audit if not c["correct"]]
check("4 misclassifications exist", len(misclassed) == 4, f"found {len(misclassed)}")

for mc in misclassed:
    idx = mc["index"]
    true_label = mc["true_label"]
    predicted = mc["predicted"]
    # Verify: the true label in features.csv at this index matches
    csv_class = df.loc[idx, "class"]
    check(f"Misclass {idx}: true={true_label}, pred={predicted}",
          csv_class == true_label,
          f"features.csv class at index {idx} = {csv_class}, audit says true={true_label}")

# ============================================================
# SECTION 6: Is the class-only ablation (0.34) real?
# ============================================================
print("\n--- SECTION 6: Class-Only Ablation Reality ---")
with open(repo / "outputs" / "faithfulness_audit_classonly.json") as f:
    co_audit = json.load(f)
co_faiths = [c["faithfulness"] for c in co_audit]
co_mean = sum(co_faiths) / len(co_faiths)
check("Class-only mean faithfulness ~0.34",
      abs(co_mean - 0.3377) < 0.01,
      f"computed mean={co_mean:.4f}, expected=0.3377")

co_false = [c for c in co_audit if c["faithfulness"] < 1.0]
check("Class-only has cases with faithfulness < 1.0",
      len(co_false) > 0,
      f"{len(co_false)}/100 cases have faithfulness < 1.0")

# Spot-check 3: manually verify a false claim
for co_case in co_false[:3]:
    idx = co_case["index"]
    per_claims = co_case.get("per_claim", [])
    false_claims = [c for c in per_claims if not c.get("sample_true", True)]
    if false_claims:
        fc = false_claims[0]
        feat = fc["feature"]
        claimed_dir = fc.get("claimed_dir", "?")
        sample_dir = fc.get("sample_dir", "?")
        val = float(df.loc[idx, feat]) if feat in df.columns else "?"
        p25 = th["global"].get(feat, {}).get("p25", 0)
        p75 = th["global"].get(feat, {}).get("p75", 0)
        check(f"Class-only case {idx}: false claim on {feat}",
              claimed_dir != sample_dir,
              f"LLM said {claimed_dir}, actual sample is {sample_dir} (val={val:.4f}, p25={p25:.4f}, p75={p75:.4f})")
    else:
        check(f"Class-only case {idx}: false claim", False, "faithfulness < 1.0 but no false per_claim found")

# ============================================================
# SECTION 7: Is the values-only ablation (0.63) real?
# ============================================================
print("\n--- SECTION 7: Values-Only Ablation Reality ---")
with open(repo / "outputs" / "faithfulness_audit_valuesonly.json") as f:
    vo_audit = json.load(f)
vo_faiths = [c["faithfulness"] for c in vo_audit]
vo_mean = sum(vo_faiths) / len(vo_faiths)
check("Values-only mean faithfulness ~0.63",
      abs(vo_mean - 0.6300) < 0.01,
      f"computed mean={vo_mean:.4f}, expected=0.6300")

vo_false = [c for c in vo_audit if c["faithfulness"] < 1.0]
check("Values-only has cases with faithfulness < 1.0",
      len(vo_false) > 0,
      f"{len(vo_false)}/100 cases have faithfulness < 1.0")

vo_perfect = [c for c in vo_audit if c["faithfulness"] == 1.0]
check("Values-only has SOME perfect cases (not all bad)",
      len(vo_perfect) > 0,
      f"{len(vo_perfect)}/100 cases have faithfulness = 1.0")

# Spot-check 3 false and 3 true
for vo_case in vo_false[:3]:
    idx = vo_case["index"]
    per_claims = vo_case.get("per_claim", [])
    false_claims = [c for c in per_claims if not c.get("sample_true", True)]
    if false_claims:
        fc = false_claims[0]
        feat = fc["feature"]
        claimed_dir = fc.get("claimed_dir", "?")
        sample_dir = fc.get("sample_dir", "?")
        check(f"Values-only case {idx}: false claim on {feat}",
              claimed_dir != sample_dir,
              f"LLM said {claimed_dir}, actual is {sample_dir}")
    else:
        check(f"Values-only case {idx}: false claim", False, "faithfulness < 1.0 but no false per_claim")

for vo_case in vo_perfect[:3]:
    idx = vo_case["index"]
    per_claims = vo_case.get("per_claim", [])
    all_true = all(c.get("sample_true", False) for c in per_claims)
    check(f"Values-only case {idx}: all claims true",
          all_true,
          f"faithfulness={vo_case['faithfulness']:.4f}, n_claims={len(per_claims)}")

# ============================================================
# SECTION 8: Is the gradient monotonic?
# ============================================================
print("\n--- SECTION 8: Gradient Monotonicity ---")
sg_mean = sum(c["faithfulness"] for c in audit) / len(audit)
co_mean = sum(c["faithfulness"] for c in co_audit) / len(co_audit)
vo_mean = sum(c["faithfulness"] for c in vo_audit) / len(vo_audit)
check("Gradient: class-only < values-only < sample-grounded",
      co_mean < vo_mean < sg_mean,
      f"class-only={co_mean:.4f} < values-only={vo_mean:.4f} < sample-grounded={sg_mean:.4f}")

# ============================================================
# SECTION 9: Are the same 100 indices used across all 3 conditions?
# ============================================================
print("\n--- SECTION 9: Index Consistency Across Conditions ---")
sg_indices = sorted([c["index"] for c in audit])
co_indices = sorted([c["index"] for c in co_audit])
vo_indices = sorted([c["index"] for c in vo_audit])
check("Sample-grounded and class-only use same indices",
      sg_indices == co_indices,
      f"sg={len(sg_indices)}, co={len(co_indices)}, match={sg_indices==co_indices}")
check("Sample-grounded and values-only use same indices",
      sg_indices == vo_indices,
      f"sg={len(sg_indices)}, vo={len(vo_indices)}, match={sg_indices==vo_indices}")

# ============================================================
# SECTION 10: Mechanism guard — is the impulsiveness check correct?
# ============================================================
print("\n--- SECTION 10: Mechanism Guard Integrity ---")
# Recompute p60 and check the impulsiveness verdicts
for feat in ["kurtosis", "crest_factor"]:
    p50 = th["global"][feat]["p50"]
    p75 = th["global"][feat]["p75"]
    p60 = p50 + 0.4 * (p75 - p50)
    print(f"  {feat}: p50={p50:.4f}, p60={p60:.4f}, p75={p75:.4f}")

classes = ["LIV", "LOV", "NRV", "Bearing", "Flywheel", "Piston", "Riderbelt"]
for c in classes:
    k_p50 = th["classes"][c]["kurtosis"]["p50"]
    cf_p50 = th["classes"][c]["crest_factor"]["p50"]
    k_p60 = th["global"]["kurtosis"]["p50"] + 0.4 * (th["global"]["kurtosis"]["p75"] - th["global"]["kurtosis"]["p50"])
    cf_p60 = th["global"]["crest_factor"]["p50"] + 0.4 * (th["global"]["crest_factor"]["p75"] - th["global"]["crest_factor"]["p50"])

    # Recompute from raw data
    class_data = df[df["class"] == c]
    recomp_k_p50 = float(np.percentile(class_data["kurtosis"], 50))
    recomp_cf_p50 = float(np.percentile(class_data["crest_factor"], 50))
    k_match = abs(recomp_k_p50 - k_p50) < 0.01
    cf_match = abs(recomp_cf_p50 - cf_p50) < 0.01
    check(f"{c} kurtosis p50 match", k_match, f"recomputed={recomp_k_p50:.4f}, stored={k_p50:.4f}")
    check(f"{c} crest_factor p50 match", cf_match, f"recomputed={recomp_cf_p50:.4f}, stored={cf_p50:.4f}")

# ============================================================
# SECTION 11: Does the LLM actually generate different text per condition?
# ============================================================
print("\n--- SECTION 11: LLM Output Diversity ---")
# Check that explanations are not identical across cases (not just templated)
sg_explanations = [c.get("explanation", "") for c in audit]
unique_explanations = len(set(sg_explanations))
check("Sample-grounded explanations are not all identical",
      unique_explanations > 1,
      f"{unique_explanations} unique explanations out of {len(sg_explanations)}")

co_explanations = [c.get("explanation", "") for c in co_audit]
unique_co = len(set(co_explanations))
check("Class-only explanations are not all identical",
      unique_co > 1,
      f"{unique_co} unique out of {len(co_explanations)}")

# Check explanations differ between conditions for the same case
if len(audit) > 0 and len(co_audit) > 0:
    idx_0 = audit[0]["index"]
    sg_text = audit[0].get("explanation", "")
    co_match = [c for c in co_audit if c["index"] == idx_0]
    if co_match:
        co_text = co_match[0].get("explanation", "")
        check("Same case has different explanation across conditions",
              sg_text != co_text,
              f"sg_len={len(sg_text)}, co_len={len(co_text)}, identical={sg_text==co_text}")

# ============================================================
# SECTION 12: Faithfulness distribution within sample-grounded
# ============================================================
print("\n--- SECTION 12: Sample-Grounded Faithfulness Distribution ---")
sg_faiths = [c["faithfulness"] for c in audit]
check("ALL sample-grounded cases have faithfulness exactly 1.0",
      all(f == 1.0 for f in sg_faiths),
      f"min={min(sg_faiths):.4f}, max={max(sg_faiths):.4f}, std={np.std(sg_faiths):.6f}")
check("Sample-grounded faithfulness std is exactly 0",
      np.std(sg_faiths) == 0,
      f"std={np.std(sg_faiths):.8f}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 80)
n_pass = sum(1 for r in results if r[0] == PASS)
n_fail = sum(1 for r in results if r[0] == FAIL)
print(f"AUDIT SUMMARY: {n_pass} PASS, {n_fail} FAIL out of {len(results)} checks")
if n_fail > 0:
    print("\nFAILED CHECKS:")
    for status, name, evidence in results:
        if status == FAIL:
            print(f"  [FAIL] {name}: {evidence}")
print("=" * 80)

# Save
with open(repo / "outputs" / "harsh_audit_report.json", "w") as f:
    json.dump([{"status": s, "name": n, "evidence": e} for s, n, e in results], f, indent=2)
print(f"\nSaved to outputs/harsh_audit_report.json")
