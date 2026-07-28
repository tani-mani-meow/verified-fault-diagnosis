"""
Recompute faithfulness + gate for all 3 audit conditions after the vacuous-truth fix.
No API calls - just re-evaluate the gate logic on stored per_claim data.
"""

import json
from pathlib import Path
import numpy as np

repo = Path(__file__).resolve().parent.parent

FAITH_THRESHOLD = 0.80


def recompute_gate(case):
    """Recompute faithfulness and gate decision with the vacuous-truth fix."""
    per_claims = case.get("per_claim", [])
    n_claims = len(per_claims)

    # FIX: n_claims=0 -> faithfulness=0.0, ESCALATE
    if n_claims == 0:
        case["faithfulness"] = 0.0
        case["n_claims"] = 0
        case["n_sample_true"] = 0
        case["class_consistency"] = 0.0
        case["gate_decision"] = "ESCALATE"
        case["gate_reason"] = "no_verifiable_claims"
        return case

    # Normal computation (unchanged)
    n_true = sum(1 for c in per_claims if c.get("sample_true", False))
    n_class_consistent = sum(1 for c in per_claims if c.get("class_consistent", False))
    faith = n_true / n_claims
    class_cons = n_class_consistent / n_claims

    case["faithfulness"] = faith
    case["n_claims"] = n_claims
    case["n_sample_true"] = n_true
    case["class_consistency"] = class_cons

    # Count atypicals
    n_atypical = sum(1 for c in per_claims if c.get("atypical", False))
    case["n_atypical"] = n_atypical

    # Gate logic (must match pipeline.py order):
    # 1. no_verifiable_claims -> ESCALATE (handled above)
    # 2. action-critical claim not sample_true -> ESCALATE false_action_critical
    # 3. faithfulness < 0.80 -> ESCALATE low_faithfulness
    # 4. atypical action-critical -> ESCALATE atypical_action_critical
    # else RELEASE

    false_action_critical = any(
        c.get("action_critical", False) and not c.get("sample_true", False)
        for c in per_claims
    )
    atypical_action_critical = any(
        c.get("atypical", False) and c.get("action_critical", False)
        for c in per_claims
    )

    if false_action_critical:
        case["gate_decision"] = "ESCALATE"
        case["gate_reason"] = "false_action_critical"
    elif faith < FAITH_THRESHOLD:
        case["gate_decision"] = "ESCALATE"
        case["gate_reason"] = "low_faithfulness"
    elif atypical_action_critical:
        case["gate_decision"] = "ESCALATE"
        case["gate_reason"] = "atypical_action_critical"
    else:
        case["gate_decision"] = "RELEASE"
        case["gate_reason"] = "passed_verification"

    return case


def process_file(filepath, label):
    with open(filepath) as f:
        data = json.load(f)

    old_release = sum(1 for c in data if c["gate_decision"] == "RELEASE")
    old_escalate = sum(1 for c in data if c["gate_decision"] == "ESCALATE")
    old_mean_fid = np.mean([c["faithfulness"] for c in data])

    fixed = [recompute_gate(c) for c in data]

    new_release = sum(1 for c in fixed if c["gate_decision"] == "RELEASE")
    new_escalate = sum(1 for c in fixed if c["gate_decision"] == "ESCALATE")
    new_mean_fid = np.mean([c["faithfulness"] for c in fixed])

    vacuous = [c for c in fixed if c.get("gate_reason") == "no_verifiable_claims"]

    print(f"\n{'='*60}")
    print(f"{label} ({len(fixed)} cases)")
    print(f"{'='*60}")
    print(f"  BEFORE: faithfulness={old_mean_fid:.4f}, RELEASE={old_release}, ESCALATE={old_escalate}")
    print(f"  AFTER:  faithfulness={new_mean_fid:.4f}, RELEASE={new_release}, ESCALATE={new_escalate}")
    print(f"  Vacuous-truth cases fixed: {len(vacuous)}")
    for vc in vacuous:
        print(f"    index={vc['index']}, true={vc['true_label']}, pred={vc['predicted']}, "
              f"old_gate=RELEASE -> new_gate=ESCALATE (no_verifiable_claims)")

    reasons = {}
    for c in fixed:
        if c["gate_decision"] == "ESCALATE":
            r = c["gate_reason"]
            reasons[r] = reasons.get(r, 0) + 1
    print(f"  Escalation reasons: {reasons}")

    with open(filepath, "w") as f:
        json.dump(fixed, f, indent=2)
    print(f"  Saved to {filepath}")

    return fixed


# Process all 3 conditions
sg = process_file(repo / "outputs" / "faithfulness_audit.json", "Sample-grounded")
co = process_file(repo / "outputs" / "faithfulness_audit_classonly.json", "Class-only")
vo = process_file(repo / "outputs" / "faithfulness_audit_valuesonly.json", "Values-only")

# Updated 3-condition comparison
print(f"\n{'='*60}")
print("UPDATED 3-CONDITION COMPARISON")
print(f"{'='*60}")
print(f"{'Condition':<25} {'Mean Faith':<12} {'RELEASE':<18} {'ESCALATE':<18}")
print(f"{'-'*71}")
for label, data in [("Class-only", co), ("Values-only", vo), ("Sample-grounded", sg)]:
    fids = [c["faithfulness"] for c in data]
    rel = sum(1 for c in data if c["gate_decision"] == "RELEASE")
    esc = len(data) - rel
    print(f"{label:<25} {np.mean(fids):<12.4f} {rel}/{len(data)} ({100*rel/len(data):.0f}%)    {esc}/{len(data)} ({100*esc/len(data):.0f}%)")

sg_mean = np.mean([c["faithfulness"] for c in sg])
co_mean = np.mean([c["faithfulness"] for c in co])
vo_mean = np.mean([c["faithfulness"] for c in vo])
print(f"\nGradient: {co_mean:.4f} -> {vo_mean:.4f} -> {sg_mean:.4f}")
print(f"Monotonic: {co_mean < vo_mean < sg_mean}")

# Save updated summary
summary = {
    "sample_grounded": {
        "mean_faithfulness": float(sg_mean),
        "release_rate": float(sum(1 for c in sg if c["gate_decision"]=="RELEASE") / len(sg))
    },
    "class_only": {
        "mean_faithfulness": float(co_mean),
        "release_rate": float(sum(1 for c in co if c["gate_decision"]=="RELEASE") / len(co))
    },
    "values_only": {
        "mean_faithfulness": float(vo_mean),
        "release_rate": float(sum(1 for c in vo if c["gate_decision"]=="RELEASE") / len(vo))
    },
    "vacuous_truth_cases_fixed": len([c for cond in [sg, co, vo] for c in cond if c.get("gate_reason") == "no_verifiable_claims"]),
}
with open(repo / "outputs" / "condition_comparison_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nUpdated summary saved to outputs/condition_comparison_summary.json")