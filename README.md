# verified-fault-diagnosis

A **Three-Certificate Trust Protocol** for acoustic-based predictive maintenance of
compressor systems. The core idea: an LLM may only deliver a fault explanation to a
technician if every factual claim it makes is **automatically verified against the
actual measured signal features** first.

## Research question
Can an LLM generate plain-English maintenance explanations for sensor-based fault
diagnosis — and can those explanations be **automatically verified as factually true**
before reaching the engineer?

## The three certificates
1. **Certificate 1 — Prediction accuracy.** A classical ML model (XGBoost on acoustic
   features) identifies the fault. Target: >95% accuracy.
2. **Certificate 2 — Explanation grounding (RAG).** The LLM explanation is grounded in
   a structured knowledge base of per-fault acoustic physics.
3. **Certificate 3 — Faithfulness verification (novel).** A *deterministic* checker
   extracts every numeric/feature claim from the explanation and verifies it against
   the recording's real feature values. Any failed claim raises an escalation flag and
   the explanation is withheld.

> Certificate 3 (output-side verification + gating) is the paper's core contribution.
> Certificates 1–2 use established techniques.

## Dataset (verified)
- **IIT Kanpur Air Compressor Acoustic Dataset** (Verma et al., *IEEE Trans.
  Reliability* 65(1):291–309, 2016, DOI 10.1109/TR.2015.2459684).
- 8 states: Healthy, LIV, LOV, NRV, Piston, Flywheel, Riderbelt, Bearing.
- 1800 recordings (225 per class), **16 kHz**, **3.125 s** (50,000 samples) each.
- Cross-modal validation planned on **MetroPT-3** (UCI #791).

## Current status

| Phase | Status |
|-------|--------|
| 0 — Setup & dataset verification | done |
| 1 — XGBoost baseline (Cert 1) | done — 95.56% test / 97.3% 5-fold CV |
| 1b — Leakage diagnostic | done — no train/test leakage confirmed |
| 2 — RAG grounding + faithfulness verifier (Cert 2 & 3) | done |
| 2b — Calibrate KB directions to data | done |
| 2c — Regenerate grounding features from data | done |
| 2d — Feature glossary + human-readable KB | done |
| 3 — explain→verify→gate pipeline | done |
| 3b — sample-grounded explanation prompt | done |
| 3c — Faithfulness audit + cost/latency figures | done — 3-condition ablation: SG=0.96, VO=0.63, CO=0.34 |

| Condition | Mean Faithfulness | Release Rate |
|-----------|:-:|:-:|
| Class-only (CO) | 0.34 | 17% |
| Values-only (VO) | 0.63 | 48% |
| Sample-grounded (SG) | 0.96 | 90% |

- **3d — Class-only ablation (proves verifier)** | done |
- **Audit fixing + vacuous-truth fix** | done — 34/34 repo audit passing |
- Phase 4 — Cross-modal + human study | not started |
- Phase 5 — IEEE paper | not started |

> The monotonic gradient CO < VO < SG confirms that more grounding context
> produces more faithful explanations — the central empirical claim of the paper.

## Repo layout
```
src/         feature extraction, training, SHAP, knowledge base, leakage check, verifier, pipeline

scripts/     faithfulness audits, ablation studies, comparison figures, repo audit

tests/       pytest suite with mock fixtures (16 tests, all passing)

outputs/     reports, plots, audit JSONs, feature matrix (data/ and model binaries are gitignored)
```

## Reproduce
```
pip install -r requirements.txt

# Set up API key
cp .env.example .env
# Edit .env with your NVIDIA NIM API key

# Phase 1: baseline model
python src/download_data.py
python src/inspect_data.py
python src/train_baseline.py
python src/explain_shap.py
python scripts/verify_phase1.py

# Phase 2-3: knowledge base + verifier
python src/compute_thresholds.py
python src/calibrate_kb.py
python src/build_grounding.py
python src/build_glossary.py

# Phase 3b-3d: faithfulness audits
python scripts/run_faithfulness_audit.py
python scripts/run_ablation_audit.py
python scripts/compare_conditions.py

# Full repo audit
python scripts/repo_audit.py
```

## Key results
- **Certificate 1:** 95.56% test accuracy, 97.3% 5-fold CV (XGBoost on 154 acoustic features)
- **Certificate 3 (sample-grounded):** 0.96 mean faithfulness, 90% release rate
- **Certificate 3 (class-only):** 0.34 mean faithfulness, 17% release rate — verifier correctly rejects unsupported claims
- **Vacuous-truth guard:** 0-claim explanations are correctly escalated (faithfulness=0.0)

## License / data
The dataset is not redistributed here; `download_data.py` fetches it. See the IIT
Kanpur / MathWorks sources for terms.