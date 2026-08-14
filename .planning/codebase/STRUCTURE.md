---
title: Codebase Structure
date: 2026-08-04
focus: arch
---

# Codebase Structure

**Analysis Date:** 2026-08-04

## Directory Layout

```
verified-fault-diagnosis/
├── src/               # Pipeline modules + offline stage scripts (Python, flat, no package)
│   ├── features.py            # extract_features() — 154 acoustic features
│   ├── download_data.py       # Dataset fetch/unzip (MathWorks)
│   ├── inspect_data.py        # Dataset verification report + waveforms
│   ├── train_baseline.py      # XGBoost baseline (Cert 1) + features.csv
│   ├── compute_thresholds.py  # Global/per-class percentiles JSON
│   ├── leakage_check.py       # Train/test leakage diagnostic
│   ├── explain_shap.py        # Per-class SHAP importance + plots
│   ├── knowledge_base.py      # KNOWLEDGE_BASE dict (Cert 2 RAG store)
│   ├── calibrate_kb.py        # KB-vs-data direction calibration
│   ├── build_grounding.py     # Data-driven grounding feature selection
│   ├── build_glossary.py      # Human-readable feature glossary
│   ├── llm_client.py          # NVIDIA NIM client (temperature 0, retries)
│   ├── pipeline.py            # run_pipeline() — explain→verify→gate (Cert 3)
│   └── verifier.py            # Deterministic claim verifier + artifact loaders
├── scripts/          # Experiment, audit, demo, and QA drivers
│   ├── run_faithfulness_audit.py      # 100-case SG audit + plots
│   ├── run_ablation_audit.py          # class-only (CO) ablation audit
│   ├── run_valuesonly_audit.py        # values-only (VO) audit
│   ├── run_valuesonly_pilot.py        # VO pilot run
│   ├── compare_conditions.py          # SG/CO/VO comparison figure + summary JSON
│   ├── recompute_audits.py            # Re-gate stored audits offline (no API)
│   ├── audit_fixes.py                 # Investigate audit FAILs (post-mortem)
│   ├── harsh_audit.py                 # Chain-of-evidence audit vs raw data
│   ├── repo_audit.py                  # Full repo hygiene audit (34 checks)
│   ├── verify.py                      # Dataset verification (phase 0)
│   ├── verify_phase1.py               # Phase 1 output verification
│   ├── run_pipeline_demo.py           # Mock/real gate demo with assertions
│   ├── live_demo.py                   # Interactive terminal demo (cached LLM outputs)
│   ├── run_feature_ablation.py        # Feature-family × classifier 5-fold CV
│   ├── test_llm.py                    # Smoke test for NIM API key
├── tests/            # pytest suite (16 tests, mock claims + real features.csv)
│   ├── test_verifier.py
│   ├── test_pipeline.py
│   ├── test_ablation.py
│   └── test_audit.py
├── outputs/          # All generated artifacts (committed except *.joblib)
│   ├── features.csv                  # 1800-row feature matrix (3 MB)
│   ├── feature_thresholds.json       # global/classes/distinctive percentiles
│   ├── kb_calibration.json           # KB direction calibration
│   ├── grounding_features.json       # data-driven grounding per class
│   ├── feature_glossary.json         # human labels + freq bands
│   ├── feature_names.json            # canonical feature order
│   ├── faithfulness_audit.json       # sample-grounded 100-case audit
│   ├── faithfulness_audit_classonly.json
│   ├── faithfulness_audit_valuesonly.json
│   ├── faithfulness_audit_*_summary.json
│   ├── condition_comparison_summary.json
│   ├── harsh_audit_report.json, atypical_analysis.json,
│   ├── pipeline_demo_report.json, shap_per_class.json
│   ├── baseline_report.txt, dataset_report.txt, leakage_report.txt
│   └── *.png                        # confusion matrix, SHAP beeswarms, audit figures
├── demo/             # Standalone React 3D showcase (Vite + Tailwind + R3F)
│   ├── index.html
│   ├── package.json / vite.config.js / tailwind.config.js / postcss.config.js
│   └── src/
│       ├── main.jsx / App.jsx / index.css
│       └── components/  # Nav, HeroScene, StatsGrid, PipelineScene, GradientScene,
│                        # VerificationDemo, MechanismGuard, AblationHeatmap,
│                        # NoveltySection, HonestLimits
├── data/             # GITIGNORED — downloaded dataset (8 class folders of WAVs)
├── .planning/        # GSD planning artifacts
│   └── codebase/     #   ← codebase map documents (this file lives here)
├── presentation.html # Self-contained single-file slide deck
├── README.md         # Project overview, certificates, reproduction flow
├── requirements.txt  # Pinned dependencies (UTF-16 encoded)
├── .env.example      # Committed env template (NVIDIA NIM key placeholder)
└── .gitignore        # data/, *.joblib, .env, __pycache__/, scratch/, .venv/
```

## Directory Purposes

**`src/` — Pipeline source (14 flat modules):**
- Purpose: Reusable core modules AND the offline pipeline stage entry points. There is no `__init__.py`; modules are imported by scripts/tests that inject `src/` into `sys.path`.
- Contains: feature extraction, ML training, thresholds, KB, calibration, LLM client, pipeline orchestrator, verifier.
- Key files: `src/pipeline.py` (orchestrator), `src/verifier.py` (deterministic check), `src/knowledge_base.py` (RAG store), `src/features.py` (signal → features).

**`scripts/` — Drivers (15 scripts):**
- Purpose: Experiments, audits, ablations, demos, and QA verification. Each resolves `repo_dir = Path(__file__).resolve().parent.parent` and does `sys.path.insert(0, str(repo_dir / "src"))` before importing (`scripts/run_pipeline_demo.py:11-15`).
- Contains: three per-condition audit drivers, comparison/aggregation, repo audit, live demo, verification scripts.
- Key files: `scripts/run_faithfulness_audit.py`, `scripts/compare_conditions.py`, `scripts/repo_audit.py`, `scripts/live_demo.py`.

**`tests/` — pytest suite (4 files, 16 tests):**
- Purpose: Regression tests for the verifier and pipeline gate logic, plus artifact-consistency tests over `outputs/`.
- Contains: `tests/test_verifier.py` (7 tests — claim truth/neutral/ungrounded, escalation, zero-claims guard, glossary completeness), `tests/test_pipeline.py` (7 tests — gate decisions per mock fixture), `tests/test_ablation.py` (1), `tests/test_audit.py` (1).
- Note: tests read the real `outputs/features.csv` and mock claim fixtures; they require trained artifacts to exist (see `tests/test_pipeline.py:18-19`).

**`outputs/` — Generated artifacts:**
- Purpose: Single sink for everything produced: feature matrix, thresholds, calibration, glossary, audit JSONs, summaries, reports, plots. Committed to git except `*.joblib` model binaries (gitignored).
- Contains: JSON artifacts (9+), CSVs, txt reports, PNG figures.
- Key files: `outputs/feature_thresholds.json`, `outputs/kb_calibration.json`, `outputs/faithfulness_audit*.json`, `outputs/condition_comparison_summary.json`.

**`demo/` — React showcase site:**
- Purpose: Interactive 3D paper showcase (React 18 + Vite + Tailwind + React Three Fiber + framer-motion). Static: results are hardcoded, no backend calls.
- Contains: `demo/src/App.jsx` assembling 10 section components in `demo/src/components/`.
- Run: `npm run dev` / `npm run build` / `npm run preview` (`demo/package.json:6-10`).

**`data/` — Raw dataset (gitignored):**
- Purpose: `data/AirCompressorDataset/` with 8 class folders of 16 kHz WAV files, populated by `src/download_data.py`; never committed.

**Root files:**
- `README.md` — research summary, three-certificate protocol, phase status table, reproduction commands.
- `requirements.txt` — fully pinned environment (note: UTF-16 encoded with BOM; includes xgboost, shap, librosa, scikit-learn, pandas, langchain, google-genai, ollama, pytest, streamlit, faiss-cpu, boto3, etc.).
- `.env.example` — template for `NVIDIA_API_KEY` / `NVIDIA_MODEL` / `NVIDIA_BASE_URL`; copy to `.env` (gitignored).
- `presentation.html` — standalone presentation deck (hero/pipeline/demo/gradient/limits sections).
- `.planning/codebase/` — GSD codebase map documents.

## Key File Locations

**Entry Points:**
- `src/pipeline.py:64` — `run_pipeline()`, the core programmatic entry point used by demos, audits, and tests.
- `src/llm_client.py:40,100` — `generate_with_usage()` / `generate()`, the LLM entry point.
- `scripts/run_pipeline_demo.py:153` — `main()`, demo entry (mock by default, `--real` flag for live API).
- `scripts/live_demo.py` — interactive terminal demo with cached real outputs.
- `demo/index.html` → `demo/src/main.jsx` — frontend entry.
- Each `src/*.py` and `scripts/*.py` stage script is itself an entry point (`python src/xxx.py`).

**Configuration:**
- `.env` (gitignored) / `.env.example` — NVIDIA NIM credentials and model override.
- `src/llm_client.py:16-18` — env var names: `NVIDIA_API_KEY`, `NVIDIA_MODEL` (default `meta/llama-3.1-70b-instruct`), `NVIDIA_BASE_URL` (default `https://integrate.api.nvidia.com/v1`).
- `.gitignore` — ignores `data/`, `*.joblib`, `.env`, `__pycache__/`, `scratch/`, `.venv/`.
- `demo/package.json`, `demo/vite.config.js`, `demo/tailwind.config.js` — frontend config.
- No CI config, no linter/formatter config, no `pyproject.toml`/`setup.py` present.

**Core Logic:**
- `src/features.py:10` — `extract_features(signal, sr)` → OrderedDict of 154 features (time-domain stats, spectral stats, 8 `band_energy_ratio_*`, 64 `wpt_energy_*` via 6-level db4 wavelet packet).
- `src/train_baseline.py:92-97` — XGBoost `multi:softprob`, 8 classes, `random_state=42`.
- `src/knowledge_base.py:7-289` — per-class KB entries.
- `src/verifier.py:160` — `check_sample_faithfulness()` (per-claim sample truth vs p25/p75).
- `src/pipeline.py:323-348` — gate decision priority chain.

**Testing:**
- `tests/test_verifier.py` — verifier unit tests with `MOCK_THRESHOLDS`/`MOCK_CALIBRATION`/`MOCK_LIV_RECORDING` fixtures (`tests/test_verifier.py:16-80`).
- `tests/test_pipeline.py` — pipeline gate tests using `MOCK_FIXTURES` from `src/pipeline.py`.
- Run: `python -m pytest tests/` (also invoked by `scripts/repo_audit.py:256`).

## Naming Conventions

**Files:**
- Python modules: `snake_case.py` (e.g., `compute_thresholds.py`, `train_baseline.py`).
- Stage/experiment scripts: `run_*` prefix for experiment drivers (`scripts/run_faithfulness_audit.py`), `verify*.py` for QA checks (`scripts/verify_phase1.py`, `scripts/verify.py`).
- Tests: `test_<module>.py` in `tests/` (e.g., `test_verifier.py`).
- JSON artifacts: `snake_case` descriptive names of what they hold (`feature_thresholds.json`, `kb_calibration.json`, `faithfulness_audit_classonly.json`).
- Audit JSON convention: `faithfulness_audit{_<condition>}.json` where condition ∈ {`classonly`, `valuesonly`, `valuesonly_pilot`}; plain `faithfulness_audit.json` is the sample-grounded run. Summary files append `_summary.json`.
- Plots: `<subject>[_<condition>].png` (e.g., `audit_condition_comparison.png`, `shap_LIV.png`).

**Directories:**
- Flat, single-level, descriptive: `src/`, `scripts/`, `tests/`, `outputs/`, `demo/`, `data/`.
- No subpackage hierarchy inside `src/`.

**Code:**
- Functions: `snake_case` (`extract_features`, `run_pipeline`, `check_sample_faithfulness`).
- Constants: `UPPER_SNAKE` (`KNOWLEDGE_BASE`, `ALL_CLASSES`, `FAULT_CLASSES`, `MOCK_FIXTURES`, `MIN_ABS_SEP`, `FAITH_THRESHOLD`).
- Class names: standard research class labels (`LIV`, `LOV`, `NRV`, `Bearing`, `Flywheel`, `Piston`, `Riderbelt`, `Healthy`) — also the KB keys and dataset folder names.
- Feature names: canonical `snake_case` with numeric suffixes (`spectral_rolloff_85`, `band_energy_ratio_4`, `wpt_energy_15`).

## Where to Add New Code

**New Feature:**
- Primary code: extend `src/features.py` `extract_features()`; register human labels in `src/build_glossary.py` `scalar_map` or the WPT/band branches.
- Tests: extend `tests/test_verifier.py` fixtures and `tests/test_pipeline.py` if gate logic changes.

**New Pipeline Stage / Artifact Producer:**
- Implementation: new module in `src/` following the pattern: resolve `repo_dir = Path(__file__).resolve().parent.parent`, `outputs_dir = repo_dir / "outputs"`, guard inputs with `sys.exit(1)` + "Run X first" messages, write JSON artifact to `outputs/`.
- Consumers load it with a matching `load_*()` function in `src/verifier.py` (e.g., `load_thresholds()`).

**New Experiment / Audit:**
- Script: `scripts/run_*.py` importing `src` via `sys.path.insert(0, str(repo_dir / "src"))`; follow `scripts/run_faithfulness_audit.py` for the concurrent/resumable audit pattern (queue + 3 workers + mutex + per-case JSON).
- Data: write outputs to `outputs/` with the `faithfulness_audit_<condition>.json` naming convention.

**New Knowledge Base Entry / Class:**
- Edit `src/knowledge_base.py` (dict entry + `FAULT_CLASSES`/`ALL_CLASSES` update; the `assert len(KNOWLEDGE_BASE) == 7` at `src/knowledge_base.py:307` must be updated too).
- Re-run `src/calibrate_kb.py` and `src/build_grounding.py` if grounding changes.

**New Component/Module (shared helper):**
- Implementation: `src/<name>.py`, imported by others via the `sys.path` convention. Note the codebase currently has no package structure — if adding many modules, prefer introducing a package (`__init__.py`) and `pip install -e .`.

**Utilities:**
- Shared helpers currently live at the top of the modules that use them (`interpolate_p40`/`interpolate_p60` duplicated in `src/calibrate_kb.py:11-15` and `src/build_grounding.py:9-13`; split logic duplicated across six files) — a shared `src/data.py` / `src/stats.py` would be the natural home for new utilities.

**Frontend section:**
- Add a section component in `demo/src/components/` and mount it in `demo/src/App.jsx` (import + JSX tag), matching existing component style.

## Special Directories

**`outputs/`:**
- Purpose: All generated artifacts — feature matrix, thresholds, calibration, glossary, audit JSONs, summaries, reports, figures.
- Generated: Yes (by `src/` and `scripts/` stages).
- Committed: Yes, except `*.joblib` binaries (gitignored via `.gitignore`).

**`data/`:**
- Purpose: Raw IIT Kanpur dataset (8 class folders of WAVs).
- Generated: Downloaded by `src/download_data.py` from MathWorks.
- Committed: No (gitignored).

**`demo/`:**
- Purpose: Standalone React showcase; `node_modules` not present, `package-lock.json` committed.
- Generated: Dev artifacts only during `npm run build` (dist).
- Committed: Source yes; `dist/` not observed.

**`.planning/`:**
- Purpose: GSD planning artifacts; `codebase/` holds the map documents (STACK/ARCHITECTURE/etc.).
- Generated: Yes (by GSD commands).
- Committed: Yes.

**`scratch/` (referenced in `.gitignore` and `scripts/repo_audit.py:87-89`):**
- Purpose: Temporary scratch space; must never be tracked.

---

*Structure analysis: 2026-08-04*
