# Verified Fault Diagnosis — Complete Codebase & System Context Map

> **Repository Goal:** Trustworthy, verifiable AI-assisted industrial fault diagnosis for reciprocating air compressors using a Three-Certificate Trust Protocol. Combines machine learning (XGBoost), physical domain knowledge (Python Knowledge Base), Large Language Models (Llama 3.1 70B), and a deterministic sensor-grounded verifier.

---

## 1. Executive Summary & Core Philosophy

Standard Large Language Models (LLMs) used in automated industrial diagnostics often suffer from **hallucinations**, **unverifiable assertions**, and **uncritical overconfidence**.

This project resolves these failure modes by enforcing a **deterministic gating mechanism** between the LLM generator and the human technician:

1. **Classification is ML-driven**: Diagnostic labels are predicted by an XGBoost model trained on 84 acoustic features.
2. **Context is Grounded**: Prompts injected into the LLM contain empirical sensor readings, class percentiles ($P_{25}, P_{50}, P_{75}$), and physical fault mechanisms.
3. **Explanations are Audit-Checked**: An automated verifier parses the LLM output into structured claims, checks every claim against raw sensor feature values, and computes a **Faithfulness Score**.
4. **Reports are Gated**: If any action-critical claim fails or is ungrounded, the system **ESCALATES** (withholds the explanation from the technician).

---

## 2. System Architecture & 6-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          STAGE 1: AUDIO INPUT                           │
│      Acoustic WAV Recording (3.125s, 16kHz mono) → 84 Features        │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               STAGE 2: CLASSIFICATION (CERTIFICATE 1)                  │
│       StandardScaler + Trained XGBoost Model → Predicted Class         │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│             STAGE 3: GROUNDING RETRIEVAL (CERTIFICATE 2)                │
│    Python Knowledge Base + Feature Threshold Percentiles (P25/P50/P75) │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   STAGE 4: LLM EXPLANATION GENERATION                   │
│   NVIDIA NIM (Llama 3.1 70B) → Structured JSON (Explanation + Claims)   │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               STAGE 5: DETERMINISTIC VERIFICATION (CERT 3)              │
│    Check claims against sensor data percentiles & mechanism guards     │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         STAGE 6: GATE DECISION                          │
│          Faithfulness ≥ 0.8 & Clear Rules? → RELEASE                    │
│          Violation / Falsified Claims?     → ESCALATE                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Directory Structure & File Map

```
verified-fault-diagnosis/
├── CODEBASE_CONTEXT.md          # [THIS FILE] Comprehensive architecture map
├── README.md                    # Quickstart guide & installation instructions
├── presentation.html            # Standalone single-file interactive presentation
├── requirements.txt             # Python dependencies (xgboost, shap, scipy, etc.)
│
├── src/                         # Core Python Library Modules
│   ├── knowledge_base.py        # Domain knowledge dict: 7 fault classes & mechanisms
│   ├── verifier.py             # Deterministic verification math & threshold checking
│   ├── pipeline.py             # Orchestrator running Stages 1–6 end-to-end
│   ├── llm_client.py           # NVIDIA NIM API client for Llama 3.1 70B
│   ├── train_baseline.py       # XGBoost classifier training pipeline
│   ├── compute_thresholds.py   # Global and per-class feature percentile builder
│   ├── calibrate_kb.py         # Statistical calibration of grounding features
│   ├── build_glossary.py       # Feature glossary mapping code names to human labels
│   ├── explain_shap.py         # SHAP importance attribution calculation
│   ├── features.py             # Feature name utilities & groupings
│   ├── inspect_data.py         # Dataset loading and sanity verification
│   ├── download_data.py        # IITK dataset fetch script
│   └── leakage_check.py        # Data leakage audit across feature sets
│
├── scripts/                     # Execution & Audit Scripts
│   ├── live_demo.py            # CLI presentation demo tool (4 curated cases)
│   ├── repo_audit.py           # Ground-truth test pipeline (34/34 system checks)
│   ├── run_faithfulness_audit.py # 100-sample audit runner across fault classes
│   ├── run_ablation_audit.py   # Ablation study runner (SG vs VO vs CO)
│   ├── run_pipeline_demo.py    # Pipeline demonstration runner
│   ├── compare_conditions.py   # Statistical comparison of ablation conditions
│   ├── recompute_audits.py     # Batch audit recalculation utility
│   ├── audit_fixes.py          # Audit verification & repair utility
│   └── verify.py               # Quick validation script
│
├── tests/                       # Automated Test Suite (Pytest)
│   ├── test_verifier.py        # Unit tests for verifier thresholds & logic
│   ├── test_pipeline.py        # Integration tests for end-to-end pipeline
│   ├── test_audit.py           # Verification tests for audit results
│   └── test_ablation.py        # Tests for ablation condition generators
│
├── outputs/                     # Generated Artifacts & Model Data
│   ├── xgb_model.joblib        # Trained XGBoost classifier
│   ├── scaler.joblib           # Fitted StandardScaler
│   ├── features.csv            # Extracted dataset features (225 samples x 85 cols)
│   ├── feature_thresholds.json # Precomputed global & class percentiles (p10..p90)
│   ├── kb_calibration.json     # Calibrated feature directions per class
│   ├── feature_glossary.json   # Human-readable labels for all 84 features
│   ├── faithfulness_audit.json # 100-case LLM audit results
│   └── *.png                   # Generated SHAP plots & audit histograms
│
└── demo/                        # Legacy React/Vite web visualizer (optional)
```

---

## 4. Key Modules & Technical Specifications

### 4.1 `src/knowledge_base.py`
Defines `KNOWLEDGE_BASE`, a structured dictionary mapping 7 compressor fault states to their physical causal mechanisms and grounding features:

- **Classes**:
  1. `LIV`: Leakage in Inlet Valve (Suction reed valve failing to seal).
  2. `LOV`: Leakage in Outlet Valve (Discharge reed valve failing to seal).
  3. `NRV`: Non-Return Valve Fault (Backflow during operation).
  4. `Piston`: Piston ring wear or cylinder wall scuffing.
  5. `Bearing`: Rolling-element bearing defect (raceway/ball flaw).
  6. `Flywheel`: Drive belt looseness or pulley imbalance.
  7. `Riderbelt`: Rider belt degradation / mechanical slack.

### 4.2 `src/verifier.py`
Contains the core mathematical logic for **Certificate 3**:
- `load_thresholds()`: Loads global and per-class feature percentiles ($P_{10}, P_{25}, P_{50}, P_{75}, P_{90}$).
- `check_sample_faithfulness(sample_row, claims, pred_class, ...)`:
  - Validates feature claim direction (`low` vs `high`).
  - Evaluates claim against global sample bounds:
    - Claim `LOW` is verified true if $val \le P_{25}$ (or $P_{50}$).
    - Claim `HIGH` is verified true if $val \ge P_{75}$ (or $P_{50}$).
  - Evaluates **Class Consistency**: Checks if sample direction matches the expected class-level profile.
  - Flags **Atypical Action-Critical Claims**: Claims true for the specific sample, but opposite to the class training median.
  - Applies **Mechanism Guard**: Ensures physical keywords (e.g. "impulsive") are supported by acoustic evidence (e.g. kurtosis $\ge P_{50}$).
  - Calculates `Faithfulness = (number of sample-true claims) / (total claims)`.

### 4.3 `src/pipeline.py`
The master orchestrator (`run_pipeline()`):
1. Scales sample features via `scaler.joblib`.
2. Predicts class label and confidence probabilities via `xgb_model.joblib`.
3. Constructs Grounding Context from `KNOWLEDGE_BASE` and sample feature directions.
4. Generates prompt and requests structured JSON output from Llama 3.1 70B via `llm_client.py`.
5. Passes LLM response to `verifier.py`.
6. Evaluates Gating Logic:
   - **RELEASE**: Faithfulness $\ge 0.8$, zero false action-critical claims, zero atypical action-critical claims, no mechanism guard violations.
   - **ESCALATE**: Triggered if any violation occurs, withholding ungrounded reports from technicians.

### 4.4 `scripts/live_demo.py`
Interactive CLI demonstration tool for live presentations:
- Supports 4 curated cases:
  - `bearing`: **RELEASE** (Clean textbook pass, 2/2 claims verified).
  - `liv`: **RELEASE** (Suction valve leakage, 5/5 claims verified).
  - `lov`: **ESCALATE** (`atypical_action_critical`, sample direction disagrees with class norm).
  - `nrv`: **ESCALATE** (`no_verifiable_claims`, LLM produced zero verifiable claims, Faithfulness = 0.0).
- Accepts `--case` and `--slow` options for pacing.
- Operates live via NVIDIA NIM API key or falls back seamlessly to audit fixtures.

### 4.5 `presentation.html`
Standalone, zero-dependency interactive presentation application:
- Interactive 4-case verification checklist with animated claim processing.
- Pipeline stage diagram highlighting the Three-Certificate Trust Protocol.
- Faithfulness Ablation Study chart ($SG=96\% > VO=71\% > CO=44\%$).
- Honest limitations section and research background.

---

## 5. Gate Decision Logic & Reason Codes

| Gate Reason Code | Decision | Trigger Condition / Description |
|---|---|---|
| `passed_verification` | **RELEASE** | Faithfulness $\ge 0.8$, all action-critical claims true, zero atypical critical claims, mechanism guard clear. |
| `no_verifiable_claims` | **ESCALATE** | LLM produced 0 extractable directional claims (Faithfulness = 0.0). |
| `atypical_action_critical` | **ESCALATE** | An action-critical claim is sample-true, but contradicts the class training median. |
| `false_action_critical` | **ESCALATE** | An action-critical claim is directly contradicted by actual sensor readings. |
| `mechanism_guard_flagged` | **ESCALATE** | LLM asserted physical mechanism terms (e.g. "impulsive") unsupported by sensor data. |

---

## 6. Monotonic Faithfulness Gradient (Ablation Benchmark)

Ablation experiments prove that explicit grounding context in the prompt directly controls explanation truthfulness:

| Grounding Condition | Description | Audit Faithfulness |
|---|---|---|
| **Sample + Class Grounding (SG)** | Full prompt: sample feature directions + class priors | **96%** (Textbook accuracy) |
| **Values-Only Grounding (VO)** | Prompt provides raw sensor values without directional hints | **71%** (LLM must infer bounds) |
| **Class-Only Grounding (CO)** | Prompt provides class mechanism only, omitting sample values | **44%** (Severe hallucination) |

*Monotonic Relationship*: $SG (96\%) > VO (71\%) > CO (44\%)$ across all 7 fault classes.

---

## 7. Operational Commands Quick Reference

### 7.1 Running the Live CLI Presentation
```powershell
# Navigate to project root
cd "C:\Users\Atharva Patil\Documents\projects\research\new"

# Run curated demo cases (PowerShell syntax)
& .venv\Scripts\python.exe scripts/live_demo.py --case bearing
& .venv\Scripts\python.exe scripts/live_demo.py --case liv
& .venv\Scripts\python.exe scripts/live_demo.py --case lov
& .venv\Scripts\python.exe scripts/live_demo.py --case nrv

# Run with 3-second pacing between stages
& .venv\Scripts\python.exe scripts/live_demo.py --case bearing --slow
```

### 7.2 Running Ground-Truth System Audit
```powershell
& .venv\Scripts\python.exe scripts/repo_audit.py
```
*(Verifies 34/34 system integrity, data purity, and verification checks).*

### 7.3 Running Test Suite
```powershell
& .venv\Scripts\python.exe -m pytest tests/
```

### 7.4 Viewing the Interactive Presentation
Open `presentation.html` in any web browser.
