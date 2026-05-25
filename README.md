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
| 0 — Setup & dataset verification | ✅ done |
| 1 — XGBoost baseline (Cert 1) | ✅ 95.56% test / 97.3% 5-fold CV (provisional) |
| 1b — Leakage diagnostic | 🔄 in progress |
| 2 — RAG grounding + faithfulness verifier (Cert 2 & 3) | ⏳ |
| 3 — Cross-modal + human study | ⏳ |
| 4 — IEEE paper | ⏳ |

> ⚠️ **Provisional baseline:** the 95.56% figure uses a random file-level split. The
> clips are pre-segmented, so we are validating against session-level train/test
> leakage (Phase 1b) before treating any accuracy number as final.

## Repo layout
```
src/         feature extraction, training, SHAP, knowledge base, leakage check

scripts/     verification gates

outputs/     reports, plots, feature matrix (data/ and model binaries are gitignored)
```

## Reproduce
```
pip install -r requirements.txt

python src/download_data.py

python src/inspect_data.py

python src/train_baseline.py

python src/explain_shap.py

python scripts/verify_phase1.py
```

## License / data
The dataset is not redistributed here; `download_data.py` fetches it. See the IIT
Kanpur / MathWorks sources for terms.
