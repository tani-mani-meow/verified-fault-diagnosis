---
title: Technology Stack
date: 2026-08-04
focus: tech
---

# Technology Stack

**Analysis Date:** 2026-08-04

## Languages

**Primary:**
- Python 3.12 — all research code: feature extraction, ML training, RAG grounding, faithfulness verifier, audit scripts (`src/`, `scripts/`, `tests/`). Code uses `str | None` style type hints (requires ≥3.10), developed against Python 3.12.9 (no `.python-version` or `runtime.txt` pins the version).

**Secondary:**
- JavaScript / JSX (React 18) — static 3D research-showcase website only, in `demo/`. It is presentation material (hardcoded results), not part of the research pipeline.

## Runtime

**Environment:**
- Python 3.12 (no virtualenv/conda config committed; no `pyproject.toml`, `setup.py`, or `setup.cfg` exists — this is a script-based project, not an installable package)

**Package Manager:**
- pip — single frozen `requirements.txt` (pip-freeze style, ~170 packages, all `==`-pinned). Installed via `pip install -r requirements.txt` per `README.md`.
- npm — for `demo/`, with `demo/package-lock.json` committed.

## Frameworks

**Core (ML/data, no web framework):**
- [XGBoost 3.3.0](https://github.com/dmlc/xgboost) — Certificate 1 classifier (`XGBClassifier`, `multi:softprob`, 8 classes) in `src/train_baseline.py:92` and the leakage check `src/leakage_check.py`
- [scikit-learn 1.7.2](https://scikit-learn.org) — `train_test_split`, `StratifiedKFold`/`GroupKFold`, `StandardScaler`, `Pipeline`, metrics (`src/train_baseline.py`, `src/leakage_check.py`, `scripts/run_feature_ablation.py`)
- [librosa 0.11.0](https://librosa.org) — audio loading and low-level audio features in `src/features.py:18` and `src/train_baseline.py:45`
- [PyWavelets 1.9.0](https://pywavelets.readthedocs.io) — `pywt.WaveletPacket` db4 level-6 → 64 `wpt_energy_*` features in `src/features.py:60` and `src/build_glossary.py:154`
- [SHAP 0.52.0](https://shap.readthedocs.io) — per-class feature importance in `src/explain_shap.py`

**Testing:**
- [pytest 8.3.1](https://docs.pytest.org) — 16 tests in `tests/`, run with `python -m pytest tests/` (no pytest config file, no `conftest.py`; each test file injects `src/` into `sys.path` manually, e.g. `tests/test_pipeline.py:12`)

**Build/Dev (demo only):**
- [Vite 5](https://vitejs.dev) — `demo/vite.config.js` (dev server port 3000, `open: true`)
- [Tailwind CSS 3.4](https://tailwindcss.com) — `demo/tailwind.config.js` + `demo/postcss.config.js` (GitHub-dark palette: `#0d1117`, `#58a6ff`, etc.)
- [three.js 0.162](https://threejs.org) with [@react-three/fiber 8](https://docs.pmnd.rs/react-three-fiber) + [@react-three/drei 9](https://github.com/pmndrs/drei) — 3D scenes in `demo/src/components/HeroScene.jsx`, `PipelineScene.jsx`, `VerificationDemo.jsx`, etc.
- [framer-motion 11](https://www.framer.com/motion/) — animations; [lucide-react 0.359](https://lucide.dev) — icons (`demo/src/App.jsx:12`)

## Key Dependencies

**Critical (actually imported by the research pipeline):**

| Package | Version | Why it matters | Where used |
|---|---|---|---|
| `xgboost` | 3.3.0 | The Certificate 1 classifier | `src/train_baseline.py`, `src/leakage_check.py` |
| `librosa` | 0.11.0 | Audio load + audio feature extraction | `src/features.py`, `src/train_baseline.py`, `src/inspect_data.py` |
| `PyWavelets` (`pywt`) | 1.9.0 | 64 WPT band-energy features | `src/features.py:60`, `src/build_glossary.py` |
| `numpy` / `scipy` / `pandas` | 2.3.1 / 1.16.3 / 2.3.0 | Core numerics, `scipy.stats` features | everywhere in `src/`, `scripts/` |
| `scikit-learn` | 1.7.2 | Splits, scaler, CV, metrics | `src/train_baseline.py`, `src/leakage_check.py` |
| `shap` | 0.52.0 | Feature-attribution plots | `src/explain_shap.py` |
| `joblib` | 1.5.2 | Model/scaler persistence (`outputs/xgb_model.joblib`, `outputs/scaler.joblib`) | `src/train_baseline.py`, `src/pipeline.py:77` |
| `soundfile` | 0.14.0 | Dataset integrity validation | `src/inspect_data.py:9`, `scripts/verify.py:3` |
| `requests` | 2.32.4 | NVIDIA NIM LLM API calls (OpenAI-compatible REST) | `src/llm_client.py:65` |
| `python-dotenv` | 1.1.1 | Loads `.env` from repo root | `src/llm_client.py:8`, `scripts/live_demo.py`, `scripts/test_llm.py` |
| `matplotlib` | 3.10.5 | Confusion matrix, SHAP, audit plots | `src/train_baseline.py`, `scripts/compare_conditions.py`, audit scripts |
| `pytest` | 8.3.1 | Test runner | `tests/` |
| `tqdm` | 4.68.3 | Progress bars during feature extraction | `src/train_baseline.py` |

**Infrastructure / auxiliary:**
- `numba 0.65.1` + `llvmlite 0.47.0` — runtime deps of librosa
- `pooch 1.9.0` — librosa's sample-data loader (not used directly)
- `PyMuPDF`, `pypdf`, `PyPDF2`, `python-docx`, `python-pptx`, `pytesseract`, `opencv-python`, `lxml`, `openpyxl`, `xlsxwriter` — document/spreadsheet libs pinned in `requirements.txt` but **never imported anywhere** in `src/`, `scripts/`, or `tests/`
- `boto3`, `botocore`, `s3transfer`, `localstack`, `localstack-client`, `awscli-local`, `terraform-local`, `docker`, `dnslib`, `pyotp`, `git-filter-repo`, `GitPython` — dev/infra tools in the freeze, **unused by the project**

**Notable unused-but-pinned LLM/vector packages:** `langchain 0.3.26`, `langchain-community`, `langchain-core`, `faiss-cpu 1.11.0`, `streamlit 1.46.1`, `ollama 0.5.1`, `google-genai 2.10.0`, `tenacity 9.1.2`, `altair`, `pydeck`, `SQLAlchemy`. The project's own RAG is a plain Python dict (`src/knowledge_base.py:7`, header comment: "Structured RAG knowledge base (Python dict, not a vector DB)") and its LLM client is hand-rolled `requests` — no LangChain/FAISS code exists in the repo. `requirements.txt` behaves as a full environment freeze rather than a curated dependency list.

## Configuration

**Environment:**
- `.env.example` committed at repo root; user copies to `.env` (`cp .env.example .env` per `README.md`). `.env` is gitignored (`.gitignore:17`).
- Loaded via `load_dotenv(dotenv_path=repo_dir / ".env", override=True)` in `src/llm_client.py:13` (also `scripts/live_demo.py:223`, `scripts/test_llm.py:17`).

**Build:**
- No build system for Python (scripts run directly: `python src/xxx.py`). `demo/` uses Vite (`npm run dev` / `npm run build` / `npm run preview`).
- No linter/formatter config (no `.eslintrc`, `.prettierrc`, `ruff`, `black`, `flake8`), no CI config (no `.github/`), no `pytest.ini`/`pyproject.toml`.

**Artifact conventions (file-based "database"):**
- `outputs/` holds all reports: `features.csv` (154-column feature matrix: 84 features per the committed `outputs/feature_names.json` plus `class`/`file_path` — README reports 154 acoustic features), `feature_thresholds.json`, `kb_calibration.json`, `feature_glossary.json`, `grounding_features.json`, audit JSONs, PNG plots
- Model binaries `outputs/xgb_model.joblib` / `outputs/scaler.joblib` are gitignored (`*.joblib`); dataset `data/` is gitignored and fetched by `src/download_data.py`

## Platform Requirements

**Development:**
- Python 3.10+ (type-hint syntax), tested on 3.12.9; Windows-specific packages pinned (`pywin32 311`, `windows-curses 2.4.2`), and `scripts/live_demo.py` contains `os.system('')` ANSI-color enabling for Windows terminals
- Node.js ≥18 for `demo/` (Vite 5 requirement)
- Network access to `integrate.api.nvidia.com` (LLM API) and `mathworks.com` (dataset download)

**Production:**
- No deployment target — research/CLI project. The only "frontend" is a static Vite build in `demo/` for presentations.

---

*Stack analysis: 2026-08-04*
