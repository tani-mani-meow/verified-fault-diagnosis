---
title: Architecture
date: 2026-08-04
focus: arch
---

<!-- refreshed: 2026-08-04 -->
# Architecture

**Analysis Date:** 2026-08-04

## System Overview

A research pipeline implementing the "Three-Certificate Trust Protocol" for
acoustic fault diagnosis of air compressors. Raw audio recordings flow through
classical ML classification (Certificate 1), knowledge-base-grounded LLM
explanation (Certificate 2), and a deterministic faithfulness verifier with a
release gate (Certificate 3, the paper's core contribution).

The codebase is a flat, script-oriented Python project: `src/` holds both the
reusable modules and the offline pipeline stages, `scripts/` holds
experiment/audit/demo drivers, and all artifacts land in `outputs/`.

```text
┌────────────────────────────────────────────────────────────────────┐
│                     DATA & SIGNAL LAYER (Cert 1 input)             │
│  `src/download_data.py` → `src/inspect_data.py` → `src/features.py`│
└───────────────────────────────┬────────────────────────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                ML BASELINE LAYER (Certificate 1)                   │
│  `src/train_baseline.py`  `src/explain_shap.py`  `src/leakage_check.py` │
│  `src/compute_thresholds.py` (percentile thresholds from TRAIN set) │
└───────────────┬────────────────────────────────┬───────────────────┘
                ▼                                ▼
┌─────────────────────────────────┐  ┌───────────────────────────────┐
│  KNOWLEDGE BASE (Cert 2)        │  │  `outputs/` artifacts:        │
│  `src/knowledge_base.py`        │  │  xgb_model.joblib, scaler,    │
│  `src/calibrate_kb.py`          │  │  feature_thresholds.json,     │
│  `src/build_grounding.py`       │  │  kb_calibration.json,         │
│  `src/build_glossary.py`        │  │  feature_glossary.json        │
└───────────────┬─────────────────┘  └──────────────┬────────────────┘
                ▼                                   ▼
┌────────────────────────────────────────────────────────────────────┐
│          EXPLAIN → VERIFY → GATE LAYER (Certificate 3)             │
│  `src/pipeline.py`  (orchestrator)                                 │
│  `src/llm_client.py` (NVIDIA NIM, temperature=0)                   │
│  `src/verifier.py`   (deterministic claim checker)                 │
└───────────────────────────────┬────────────────────────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│  OUTPUT: gate decision RELEASE (explanation delivered) or          │
│          ESCALATE (explanation withheld) + audit JSONs in outputs/ │
└────────────────────────────────────────────────────────────────────┘
```

Audit/QA scripts (`scripts/*`) and the pytest suite (`tests/*`) sit beside the
pipeline and consume `src/` modules plus `outputs/` artifacts. The `demo/`
React app is a standalone showcase with hardcoded results — it does not call
the Python pipeline.

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| `extract_features` | Compute 154 acoustic features from a raw signal (time-domain stats, spectral stats, 8 band-energy ratios, 64 WPT log-energies) | `src/features.py` |
| Dataset download/verify | Fetch IIT Kanpur zip from MathWorks, confirm 8 classes / 1800 files / 16 kHz | `src/download_data.py`, `src/inspect_data.py` |
| Baseline training | XGBoost 8-class classifier, scaler, feature list, 5-fold CV, confusion matrix | `src/train_baseline.py` |
| Threshold computation | TRAIN-only global/per-class percentiles (p10..p90) + distinctive top/bottom-2 features | `src/compute_thresholds.py` |
| Leakage diagnostic | Adjacent-similarity test, contiguous-split stress test, GroupKFold CV | `src/leakage_check.py` |
| SHAP analysis | Per-class mean \|SHAP\| top-10 + beeswarm plots | `src/explain_shap.py` |
| Knowledge base | 7 fault classes: mechanism text, grounding features, human labels, confusable-pair notes | `src/knowledge_base.py` |
| KB calibration | Verify KB directions against data-derived class medians (data_direction) | `src/calibrate_kb.py` |
| Grounding selection | Data-driven pick of 6 grounding features per class (separation score, strong/moderate tiers) | `src/build_grounding.py` |
| Glossary build | Human-readable labels + frequency bands for every feature; WPT tiling sanity checks | `src/build_glossary.py` |
| LLM client | NVIDIA NIM chat completions, exponential backoff, thread-local token usage | `src/llm_client.py` |
| Pipeline orchestrator | predict → ground → prompt LLM → parse claims → verify → gate | `src/pipeline.py` |
| Verifier | Per-claim sample-truth / class-consistency / atypical checks against percentiles | `src/verifier.py` |
| Audit drivers | 100-case faithfulness audits per grounding condition, aggregation, plots | `scripts/run_faithfulness_audit.py`, `scripts/run_ablation_audit.py`, `scripts/run_valuesonly_audit.py` |
| Repo QA | Full repo audit (security, staleness, consistency, reproducibility, tests) | `scripts/repo_audit.py` |
| Tests | 16 pytest tests using real `outputs/features.csv` + mock claims | `tests/test_verifier.py`, `tests/test_pipeline.py`, `tests/test_ablation.py`, `tests/test_audit.py` |
| Showcase site | Static 3D React marketing/paper-showcase page | `demo/src/App.jsx`, `presentation.html` |

## Pattern Overview

**Overall:** Linear research pipeline (pipeline-of-stages) executed by
top-level scripts, with shared module state in `src/` reached via `sys.path`
manipulation. Offline artifacts in `outputs/` (JSON + joblib) are the only
persistent state; there is no database and no web service.

**Key Characteristics:**
- Every stage is a standalone script with `main()` and an
  `if __name__ == "__main__"` guard (`src/train_baseline.py:167`).
- Determinism by convention: `random_state=42` everywhere
  (`src/train_baseline.py:76`, `src/compute_thresholds.py:26`,
  `scripts/run_faithfulness_audit.py:113`).
- **No hardcoded thresholds in the verifier**: all numeric bounds come from
  `outputs/feature_thresholds.json`, which is computed from TRAIN-set
  percentiles only (stated as a rule in `src/knowledge_base.py:3-5`).
- LLM output is structured as strict JSON `{claims, explanation}` and parsed
  with `json.loads` → `ast.literal_eval` fallback (`src/pipeline.py:260-271`).
- The verifier is deliberately deterministic (no LLM in the check loop) —
  this is the paper's design: "explain → verify → gate".

## Layers

**Data & Signal Layer:**
- Purpose: Acquire the dataset and turn raw WAV into a numeric feature matrix.
- Location: `src/download_data.py`, `src/inspect_data.py`, `src/features.py`
- Contains: URL fetch + zip extraction (zip-slip guarded, `src/download_data.py:45-61`), dataset verification report, and the single feature-extraction function.
- Depends on: librosa, scipy, pywt, numpy; raw files under `data/AirCompressorDataset/` (gitignored).
- Used by: `src/train_baseline.py:19` (imports `extract_features`).

**ML Baseline Layer (Certificate 1):**
- Purpose: Train/evaluate the XGBoost classifier and produce the threshold + interpretability artifacts the rest of the pipeline consumes.
- Location: `src/train_baseline.py`, `src/explain_shap.py`, `src/leakage_check.py`, `src/compute_thresholds.py`
- Contains: model/scaler training, SHAP analysis, leakage diagnostics, percentile computation.
- Depends on: `src/features.py`, `outputs/features.csv`.
- Used by: every downstream stage (model, scaler, feature names, thresholds).

**Knowledge Base Layer (Certificate 2):**
- Purpose: Structured RAG grounding — a Python dict (explicitly *not* a vector DB, see `src/knowledge_base.py:2`) plus the data-calibration artifacts that make KB directions trustworthy.
- Location: `src/knowledge_base.py`, `src/calibrate_kb.py`, `src/build_grounding.py`, `src/build_glossary.py`
- Contains: per-class mechanism text, grounding feature lists, human-readable labels, calibration JSON.
- Depends on: `outputs/feature_thresholds.json`, `outputs/feature_names.json`.
- Used by: `src/pipeline.py`, `src/verifier.py`, `scripts/live_demo.py`.

**Explain → Verify → Gate Layer (Certificate 3):**
- Purpose: Generate the technician explanation from grounded context, extract structured claims, verify each claim against the recording's real feature values, and gate delivery.
- Location: `src/pipeline.py`, `src/llm_client.py`, `src/verifier.py`
- Contains: the orchestrator `run_pipeline()`, the NVIDIA NIM client, and the deterministic verifier.
- Depends on: KB layer + all `outputs/` artifacts.
- Used by: `scripts/run_*_audit.py`, `scripts/run_pipeline_demo.py`, `tests/test_pipeline.py`.

**Audit & QA Layer:**
- Purpose: Empirical faithfulness audits, ablations, comparisons, and repo hygiene checks; also the pytest suite.
- Location: `scripts/`, `tests/`
- Depends on: `src/` modules + `outputs/` artifacts (audit JSONs, features.csv).

**Showcase Layer:**
- Purpose: Paper/marketing presentation artifacts; no code dependency on the Python pipeline.
- Location: `demo/` (Vite + React 3D site), `presentation.html` (self-contained).

## Data Flow

### Primary Request Path (single sample through the pipeline)

1. Entry: `run_pipeline(sample_row, true_label, mock, grounding_mode)` — `src/pipeline.py:64`
2. Load artifacts: `load_thresholds()` / `load_calibration()` (`src/verifier.py:9-29`), then model binaries `outputs/xgb_model.joblib`, `outputs/scaler.joblib`, `outputs/feature_names.json` — `src/pipeline.py:67-80`
3. Predict: scale the row, `clf.predict_proba`, argmax → `predicted_class`, confidence — `src/pipeline.py:82-91`
4. KB lookup + glossary: `KNOWLEDGE_BASE[predicted_class]`, `outputs/feature_glossary.json` — `src/pipeline.py:93-112`; missing class short-circuits to `ESCALATE` with reason `no grounding entry` — `src/pipeline.py:94-108`
5. Grounding context: build a table per `grounding_mode` (`sample` / `values_only` / `class_only`) combining feature labels, this sample's values, global p25/p50/p75, and derived `sample_dir` — `src/pipeline.py:114-147`
6. LLM call: `llm_client.generate(prompt, system=...)` with `temperature=0.0`, strict JSON schema prompt — `src/pipeline.py:252`, `src/llm_client.py:40-98`; in `mock` mode, claims come from `MOCK_FIXTURES` — `src/pipeline.py:152-154`
7. Parse: strip Markdown fences, `json.loads` with `ast.literal_eval` fallback; unparseable → `ESCALATE` reason `unparseable_llm_output` — `src/pipeline.py:254-286`
8. Verify: map `action_critical` → `critical`, call `check_sample_faithfulness(sample_row, predicted_class, verifier_claims, ...)` — `src/pipeline.py:288-295`, `src/verifier.py:160-261` (per-claim `sample_true`, `class_consistent`, `atypical`)
9. Mechanism guard: if the class is not impulsiveness-supported (kurtosis/crest-factor check vs interpolated p60), scan the explanation for banned impulse terms → `mechanism_warnings` — `src/pipeline.py:301-317`
10. Gate decision (`src/pipeline.py:323-348`), priority order:
    - 0 verifiable claims → `ESCALATE` (`no_verifiable_claims`) — vacuous-truth guard
    - any false action-critical claim → `ESCALATE` (`false_action_critical`)
    - faithfulness < 0.80 → `ESCALATE` (`low_faithfulness`)
    - atypical action-critical claim → `ESCALATE` (`atypical_action_critical`)
    - otherwise → `RELEASE` (`passed_verification`); atypical non-critical features are attached as `atypical_features`
11. Output: report dict `{predicted, confidence, true_label, claims, per_claim, faithfulness, class_consistency, gate_decision, reason, explanation, mechanism_warnings[, atypical_features]}` — `src/pipeline.py:350-366`

### Offline Artifact Build Flow (run once per pipeline phase)

1. `python src/download_data.py` → `data/AirCompressorDataset/` (gitignored)
2. `python src/inspect_data.py` → `outputs/dataset_report.txt`, `outputs/healthy_vs_liv.png`
3. `python src/train_baseline.py` → `outputs/features.csv`, `xgb_model.joblib`, `scaler.joblib`, `feature_names.json`, `baseline_report.txt`, `confusion_matrix.png`
4. `python src/compute_thresholds.py` → `outputs/feature_thresholds.json`
5. `python src/calibrate_kb.py` → `outputs/kb_calibration.json`
6. `python src/build_grounding.py` → `outputs/grounding_features.json`
7. `python src/build_glossary.py` → `outputs/feature_glossary.json`
8. Audit runs (`scripts/run_faithfulness_audit.py` etc.) → per-condition audit JSONs + summary JSONs + PNG figures

### Audit Flow (Certificate 3 measurement)

1. Load `outputs/features.csv`, reproduce the exact 80/20 split (`random_state=42`), exclude Healthy, stratified-sample 100 rows — `scripts/run_faithfulness_audit.py:100-123`
2. Run `run_pipeline(..., mock=False)` per row across 3 worker threads; queue with retries; thread-safe append + per-case JSON save (resume capability) — `scripts/run_faithfulness_audit.py:28-95, 161-186`
3. Record latency + token usage via `llm_client.get_last_usage()` (thread-local) — `scripts/run_faithfulness_audit.py:41-43`, `src/llm_client.py:24-38`
4. Aggregate faithfulness stats, gate outcomes, safety cross-tab (release vs correctness), per-class stats, cost/latency → `outputs/faithfulness_audit_summary.json` — `scripts/run_faithfulness_audit.py:190-284`
5. Plot histogram / release-rate-by-class / gate-vs-correctness PNGs — `scripts/run_faithfulness_audit.py:287-327`

**State Management:**
- Persistent state lives only in `outputs/` (JSON thresholds/calibration/audits/glossary, joblib model+scaler, CSV feature matrix). `data/` raw audio and `*.joblib` are gitignored (`src/../.gitignore`).
- In-memory shared state: module-level constants `KNOWLEDGE_BASE`, `ALL_CLASSES`, `FAULT_CLASSES` in `src/knowledge_base.py:7-293`; thread-local `LAST_USAGE` in `src/llm_client.py:25`.
- Concurrency: audit scripts use `threading.Lock` + shared lists (`scripts/run_faithfulness_audit.py:24-25`); LLM calls are thread-safe via thread-local token tracking.

## Key Abstractions

**KNOWLEDGE_BASE (dict):**
- Purpose: The structured RAG store — one entry per fault class (7 fault classes; Healthy has no entry).
- Examples: `src/knowledge_base.py:7-289`
- Pattern: Plain Python dict with per-class keys: `name`, `mechanism` (verbatim causal text the LLM must reuse), `acoustic_signature`, `grounding_features` (canonical feature → direction), `grounding_features_human`, `distinguishing_from`, `references`.
- Exports: `FAULT_CLASSES`, `ALL_CLASSES`, `normalize_direction()` — `src/knowledge_base.py:292-304`

**Claim (dict schema):**
- Purpose: The structured contract between LLM and verifier.
- Shape: `{"feature": "<canonical_name>", "direction": "high|low", "action_critical": bool}` — `src/pipeline.py:178-182`
- Verifier additionally supports `"critical"` and `"context_dependent"`/`"distinctive"` variants — `src/verifier.py:105-116, 141`

**Report (dict schema):**
- Purpose: The uniform output of `run_pipeline()`, consumed by demos, audits, and tests.
- Shape: `{predicted, confidence, true_label, claims, per_claim, faithfulness, class_consistency, gate_decision, reason, explanation, mechanism_warnings, [atypical_features]}` — `src/pipeline.py:350-366`

**Percentile Thresholds JSON:**
- Purpose: The single source of numeric truth for verification; global (all classes) + per-class p10–p90 + distinctive top/bottom-2 lists.
- Producer: `src/compute_thresholds.py` → `outputs/feature_thresholds.json`

**Calibration JSON:**
- Purpose: Records whether each KB grounding direction matches data (`data_direction`, `kb_direction`, `match`) so the verifier only enforces directions the data supports.
- Producer: `src/calibrate_kb.py` → `outputs/kb_calibration.json`

**MOCK_FIXTURES:**
- Purpose: Deterministic stand-ins for LLM output, used to unit-test gate logic and demos without API keys.
- Location: `src/pipeline.py:15-62` (8 fixtures: correct, flipped, critical, atypical_non_critical, atypical_critical, impulsive_unsupported, impulsive_supported)

**CACHED_DEMOS:**
- Purpose: Real audit-derived LLM outputs replayed in `scripts/live_demo.py` when the NIM API is unavailable.
- Location: `scripts/live_demo.py:40-` (keyed by fault, each mapping to a row index in `outputs/features.csv`)

## Entry Points

**CLI pipeline stages (Python scripts):**
- Location: `src/*.py` and `scripts/*.py`, each with `main()` + `if __name__ == "__main__"` guard.
- Triggers: manual `python src/...` / `python scripts/...` per README reproduction flow (`README.md:73-101`).
- Responsibilities: `src/` stages build artifacts; `scripts/` stages run experiments and QA.

**Programmatic API:**
- `run_pipeline(sample_row, true_label=None, mock=False, grounding_mode="sample")` — `src/pipeline.py:64`; used by `scripts/run_pipeline_demo.py:57`, all audit scripts, and `tests/test_pipeline.py:38`.
- `verify_claim`, `verify_explanation`, `check_sample_faithfulness` — `src/verifier.py:31, 129, 160`; used by tests directly.
- `generate(prompt, system)` / `generate_with_usage(...)` — `src/llm_client.py:40, 100`; used by `src/pipeline.py:252` and `scripts/test_llm.py`.

**Frontend entry points:**
- `demo/index.html` → `demo/src/main.jsx` → `demo/src/App.jsx` (Vite dev server: `npm run dev`).
- `presentation.html` — fully self-contained single-file slide deck.

**Test entry points:**
- `pytest tests/` — 16 tests (README `tests/` note; `scripts/repo_audit.py:256` runs the same command).

## Architectural Constraints

- **Threading:** Audits run 3 concurrent worker threads (`ThreadPoolExecutor`, `scripts/run_faithfulness_audit.py:179`); all shared-state writes are guarded by a mutex. The LLM client is thread-safe by design (thread-local `LAST_USAGE`, `src/llm_client.py:25`).
- **Global state:** `KNOWLEDGE_BASE`/`ALL_CLASSES`/`FAULT_CLASSES` are module-level singletons in `src/knowledge_base.py`; imported by `src/pipeline.py:11`, `src/verifier.py:7`, `src/calibrate_kb.py:9`, `scripts/live_demo.py:22`. Mutating them affects all consumers.
- **Determinism rule:** all splits/randomness use `random_state=42`; verification thresholds must come from TRAIN-set percentiles only (enforced by code comment convention in `src/knowledge_base.py:3-5` and the code itself in `src/compute_thresholds.py:22-30`).
- **No hardcoded numeric facts in code:** threshold values live exclusively in `outputs/feature_thresholds.json`; `src/knowledge_base.py` contains directions only.
- **Path resolution convention:** every script/tests resolves the repo root as `Path(__file__).resolve().parent.parent` and injects `src/` via `sys.path.insert(0, ...)` — no installed package, no `__init__.py` in `src/`.
- **LLM determinism:** `temperature=0.0` always (`src/llm_client.py:59`); the model defaults to `meta/llama-3.1-70b-instruct` via NVIDIA NIM, overridable with `NVIDIA_MODEL` env var (`src/llm_client.py:17-18`).
- **Secrets:** API key is read from `.env` at repo root (gitignored); `.env.example` is the committed template. `src/llm_client.py:13` loads `.env` with `override=True` on import — note this means importing `llm_client` mutates the process environment.

## Anti-Patterns

### sys.path hacking instead of a package

**What happens:** Every script and test file duplicates
`sys.path.insert(0, str(repo_dir / "src"))` before importing (`scripts/run_pipeline_demo.py:11-13`, `scripts/run_faithfulness_audit.py:16-18`, `tests/test_verifier.py:8-10`).
**Why it's wrong:** Fragile — imports work only when the CWD/script layout matches expectations; makes IDE support and packaging awkward.
**Do this instead:** Convert `src/` to a proper package with `__init__.py` and install it (`pip install -e .`) so all modules are importable as `from vfd import pipeline`.

### Duplicated train/test split logic

**What happens:** The identical split snippet (load `outputs/features.csv`, sort classes, `train_test_split(test_size=0.2, random_state=42, stratify=y)`) is copy-pasted into `src/train_baseline.py:67-78`, `src/compute_thresholds.py:24-29`, `src/explain_shap.py:34-45`, `scripts/run_pipeline_demo.py:27-37`, `scripts/run_faithfulness_audit.py:106-115`, `tests/test_pipeline.py:19-31`.
**Why it's wrong:** Any divergence silently corrupts experimental comparisons (e.g., auditing different rows than training).
**Do this instead:** Extract one `load_split_features()` helper (e.g., in `src/features.py` or a new `src/data.py`) and import it everywhere.

### Near-duplicate prompt templates per grounding mode

**What happens:** `src/pipeline.py:160-250` contains three almost-identical system prompts (class_only / values_only / sample) differing only in a few sentences.
**Why it's wrong:** Prompt drift between modes is possible and hard to review; changes must be made three times.
**Do this instead:** Build the shared rules once and append a small per-mode section, or generate the prompt from a template dict.

### Test/mock data embedded in production code

**What happens:** `MOCK_FIXTURES` (8 fixture claim sets) lives inside `src/pipeline.py:15-62`, and gate-threshold constants like `0.80` appear inline in `src/pipeline.py:335`, `src/verifier.py:151`, and `scripts/recompute_audits.py` (`FAITH_THRESHOLD = 0.80`).
**Why it's wrong:** Production module carries test data; the faithfulness gate threshold is scattered magic numbers that could drift apart.
**Do this instead:** Move fixtures to `tests/fixtures/` and define the gate threshold as a module-level constant imported by all consumers.

### Hardcoded expected results in the repo audit

**What happens:** `scripts/repo_audit.py:133-141` asserts the summary means are ~0.96 / ~0.3377 / ~0.63 and `scripts/leakage_check.py:172` hardcodes `baseline_acc = 0.955556` as a fallback.
**Why it's wrong:** Re-running experiments that legitimately change numbers will fail the audit, conflating "regression" with "result updated".
**Do this instead:** Read expected values from the summary JSON itself or accept a documented result file.

## Error Handling

**Strategy:** Fail-fast with explicit `sys.exit(1)` in CLI stages when required artifacts are missing (e.g., `src/train_baseline.py:29-31`, `src/compute_thresholds.py:15-18`, `src/verifier.py:15, 25` raise `FileNotFoundError` for thresholds/calibration). The runtime pipeline raises `ValueError` when the API key is missing in non-mock mode (`src/pipeline.py:156-158`).

**Patterns:**
- Upstream-artifact guards: every stage checks its input file exists before loading, with a message naming the producing script (`"Run compute_thresholds.py first"` — `src/verifier.py:15`).
- LLM errors: `src/llm_client.py:63-97` retries transient HTTP errors (429/503/504) and connection timeouts with exponential backoff (2s base, `delay *= 2`), up to 10 attempts; non-transient errors re-raise immediately.
- Unparseable LLM output: `src/pipeline.py:260-271` falls back to `ast.literal_eval`, then converts failure into an explicit `ESCALATE` with `reason="unparseable_llm_output"` — never a crash.
- Per-case resilience in audits: worker threads catch exceptions per row, log, and re-queue with a 5s sleep (`scripts/run_faithfulness_audit.py:169-176`); JSON is saved per-case so runs can resume.
- Verification never raises on data problems: missing features/calibration become failed claims with `reason` strings (`src/verifier.py:36-73`).

## Cross-Cutting Concerns

**Logging:** Plain `print()` to stdout everywhere; no logging module, no log files. Audit scripts print per-case progress lines and summaries.
**Validation:** Artifact-level validation in dedicated scripts — `scripts/verify_phase1.py` (Phase 1 outputs), `scripts/verify.py` (dataset), `scripts/repo_audit.py` (whole repo), `src/build_glossary.py:114-170` (WPT tiling + Gray-code ordering assertions raising `ValueError` on inconsistency).
**Authentication:** Single bearer token `NVIDIA_API_KEY` read from `.env` (`src/llm_client.py:16`); placeholder value detection in `src/pipeline.py:157` and `scripts/run_pipeline_demo.py:119`.
**Reproducibility:** Pinned `requirements.txt` (UTF-16 encoded), fixed seeds, and a documented phase-by-phase reproduction flow in `README.md:73-101`.

---

*Architecture analysis: 2026-08-04*
