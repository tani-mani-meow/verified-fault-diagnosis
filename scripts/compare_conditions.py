# scripts/compare_conditions.py
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

repo = Path(__file__).resolve().parent.parent

with open(repo / "outputs/faithfulness_audit.json") as f:
    sg = json.load(f)
with open(repo / "outputs/faithfulness_audit_classonly.json") as f:
    co = json.load(f)
with open(repo / "outputs/faithfulness_audit_valuesonly.json") as f:
    vo = json.load(f)

def summary(cases, label):
    fids = [c["faithfulness"] for c in cases]
    rel = sum(1 for c in cases if c["gate_decision"] == "RELEASE")
    mw = sum(1 for c in cases if len(c.get("mechanism_warnings", [])) > 0)
    mean_val = np.mean(fids)
    median_val = np.median(fids)
    std_val = np.std(fids)
    min_val = min(fids)
    rel_pct = 100 * rel / len(cases)
    esc_pct = 100 * (len(cases) - rel) / len(cases)
    esc = len(cases) - rel
    print(f"| {label} | {mean_val:.4f} | {median_val:.4f} | {std_val:.4f} | {min_val:.4f} | {rel}/{len(cases)} ({rel_pct:.0f}%) | {esc}/{len(cases)} ({esc_pct:.0f}%) | {mw}/{len(cases)} |")
    return fids

print("| Condition | Mean | Median | Std | Min | RELEASE | ESCALATE | Mech Warnings |")
print("|---|---|---|---|---|---|---|---|")
sg_f = summary(sg, "Sample-grounded")
co_f = summary(co, "Class-only")
vo_f = summary(vo, "Values-only")

# Figure: overlaid histograms
fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(0, 1.05, 22)
ax.hist(co_f, bins=bins, alpha=0.6, label="Class-only (no sample info)", color="red")
ax.hist(vo_f, bins=bins, alpha=0.6, label="Values-only (raw values, no directions)", color="orange")
ax.hist(sg_f, bins=bins, alpha=0.6, label="Sample-grounded (directions given)", color="green")
ax.set_xlabel("Faithfulness Score")
ax.set_ylabel("Number of Cases")
ax.set_title("Certificate 3: Faithfulness Across Grounding Conditions")
ax.legend()
fig.savefig(repo / "outputs/audit_condition_comparison.png", dpi=150, bbox_inches="tight")
print(f"\nFigure saved to outputs/audit_condition_comparison.png")

# Save summary JSON
summary_json = {
    "sample_grounded": {
        "mean": float(np.mean(sg_f)),
        "median": float(np.median(sg_f)),
        "std": float(np.std(sg_f)),
        "min": float(min(sg_f)),
        "release_rate": float(sum(1 for c in sg if c["gate_decision"] == "RELEASE") / len(sg))
    },
    "class_only": {
        "mean": float(np.mean(co_f)),
        "median": float(np.median(co_f)),
        "std": float(np.std(co_f)),
        "min": float(min(co_f)),
        "release_rate": float(sum(1 for c in co if c["gate_decision"] == "RELEASE") / len(co))
    },
    "values_only": {
        "mean": float(np.mean(vo_f)),
        "median": float(np.median(vo_f)),
        "std": float(np.std(vo_f)),
        "min": float(min(vo_f)),
        "release_rate": float(sum(1 for c in vo if c["gate_decision"] == "RELEASE") / len(vo))
    },
}

with open(repo / "outputs/condition_comparison_summary.json", "w") as f:
    json.dump(summary_json, f, indent=2)
print("Saved summary JSON to outputs/condition_comparison_summary.json")
