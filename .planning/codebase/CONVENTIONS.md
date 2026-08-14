---
title: Coding Conventions
date: 2026-08-04
focus: quality
---

# Coding Conventions

**Analysis Date:** 2026-08-04

## Naming Patterns

**Files:**
- Python modules: `snake_case.py` — e.g. `src/verifier.py`, `src/knowledge_base.py`, `src/compute_thresholds.py`
- Test files: `test_<module>.py` in `tests/` — e.g. `tests/test_verifier.py`, `tests/test_pipeline.py`
- Scripts: `snake_case.py` under `scripts/` — e.g. `scripts/run_faithfulness_audit.py`, `scripts/repo_audit.py`
- React components: `PascalCase.jsx` in `demo/src/components/` — e.g. `demo/src/components/VerificationDemo.jsx`
- Data/output artifacts: `snake_case.json|.csv|.png` in `outputs/`

**Functions:**
- Python: `snake_case` with a descriptive verb prefix — `verify_claim`, `check_sample_faithfulness`, `load_thresholds`, `extract_features`, `run_pipeline` (`src/verifier.py:31,160`, `src/features.py:10`, `src/pipeline.py:64`)
- Entry point is always `main()` guarded by `if __name__ == "__main__":` — `src/train_baseline.py:21,167`, `src/compute_thresholds.py:9,113`
- JS: `camelCase` for local functions and hooks — `selectedCard`, `setActiveCard` (`demo/src/components/VerificationDemo.jsx:50-54`)

**Variables:**
- Python: `snake_case` — `faithfulness_score`, `class_grounding`, `gate_decision`, `per_claim`
- Single-letter locals only in tight loops/`numpy` math: `y, sr` in `src/features.py:13`, `X, y` in `src/train_baseline.py:67`
- JS: `camelCase` — `activeCard`, `visibleClaims`, `hovered`

**Types:**
- `PascalCase` only in the React demo (`FloatingCard` in `demo/src/components/VerificationDemo.jsx:8`); Python has no classes — data flows as plain `dict`s with string keys
- Claim schema is a dict: `{"feature": <canonical_name>, "direction": "high|low|unspecified", "critical"/"action_critical": bool}` — see `tests/test_verifier.py:87` and `src/pipeline.py:18`

**Constants:**
- Module-level `UPPER_SNAKE_CASE` for config-like data:
  - `KNOWLEDGE_BASE`, `FAULT_CLASSES`, `ALL_CLASSES`, `normalize_direction` (`src/knowledge_base.py:7,292-296`)
  - `MOCK_FIXTURES` (`src/pipeline.py:15`)
  - `API_KEY`, `MODEL`, `BASE_URL`, `LAST_USAGE` (`src/llm_client.py:16-25`)
  - Test mocks: `MOCK_THRESHOLDS`, `MOCK_CALIBRATION`, `MOCK_LIV_RECORDING` (`tests/test_verifier.py:16-80`)
- Private module globals use a leading underscore: `_feature_count_printed` (`src/features.py:8`)

## Code Style

**Formatting:**
- No formatter or linter config detected (no `pyproject.toml`, `.prettierrc`, `eslint.config.*`, `.ruff.toml`, `.flake8`)
- Python: 4-space indentation, double-quoted strings by convention (`src/verifier.py`, `src/pipeline.py`), though single quotes appear in dict keys (`src/features.py:14-16`) and JSX files use single quotes consistently
- JSX: 2-space indentation, no semicolons, single quotes (`demo/src/App.jsx`)
- Blank lines separate logical blocks; comments above the code they explain, rarely trailing
- f-strings are used everywhere, always with explicit format specifiers: `f"{val:.4f}"`, `f"[{idx+1}/100]"`, `f"mean={actual_mean:.4f}"` (`src/verifier.py:85`, `scripts/repo_audit.py:179`)
- Long prompt strings use implicit string concatenation of adjacent literals (`src/pipeline.py:161-183`)
- A prior commit explicitly removed verbose comments: `2eff2f5 Refactor codebase to remove AI fingerprints and hyper-verbose comments` — keep comments terse

**Linting:**
- None configured; the closest thing is the runtime PASS/FAIL audit style used by `scripts/repo_audit.py` and `scripts/harsh_audit.py`

## Import Organization

**Order:**
1. stdlib: `os`, `sys`, `json`, `re`, `time`, `queue`, `threading`, `pathlib.Path`, `concurrent.futures`
2. third-party: `numpy as np`, `pandas as pd`, `matplotlib.pyplot as plt`, `sklearn.*`, `joblib`, `librosa`, `pywt`, `requests`, `dotenv`
3. local `src` modules — always after a manual `sys.path` insert

**Path Aliases:**
- No package installation. `src/` is imported via a repeated path hack in every importer (see Module Design):
  ```python
  # Ensure we can import from src
  script_dir = Path(__file__).resolve().parent
  repo_dir = script_dir.parent
  sys.path.insert(0, str(repo_dir / "src"))
  from pipeline import run_pipeline
  import llm_client
  ```
  (`scripts/run_faithfulness_audit.py:16-21`, identical in `src/verifier.py:6-7`, `src/pipeline.py:10-13`, `tests/test_verifier.py:8-10`, `tests/test_pipeline.py:10-12`)
- Importing modules from `src` uses bare module names (`from pipeline import ...`), not `from src.pipeline import ...`

## Error Handling

**Patterns:**
- Loaders validate artifact existence and raise `FileNotFoundError` with actionable guidance — always "Run X first":
  ```python
  raise FileNotFoundError(f"Thresholds file not found at {thresholds_path}. Run compute_thresholds.py first.")
  ```
  (`src/verifier.py:15`, `src/verifier.py:26`, `src/pipeline.py:75`)
- Required env config raises `ValueError`: `raise ValueError("NVIDIA_API_KEY is not set. Run in mock mode or update .env file.")` (`src/pipeline.py:158`)
- Scripts degrade gracefully: `print(f"Error: ...")` then `sys.exit(1)` (`src/compute_thresholds.py:15-17`, `src/train_baseline.py:29-31`, `scripts/run_faithfulness_audit.py:102-104`)
- LLM calls: exponential backoff retry loop, re-raise on final attempt; retry only on 429/503/504 and connection/timeout errors (`src/llm_client.py:63-96`)
- LLM JSON parsing uses a fallback chain: `json.loads` → `ast.literal_eval` → `unparseable = True` flag handled by the caller (`src/pipeline.py:260-271`, result path at `src/pipeline.py:273-286`)
- Broad `except Exception` is used deliberately for non-fatal branches: per-file feature extraction abort (`src/train_baseline.py:55-57`), audit case failures with retry (`scripts/run_faithfulness_audit.py:93-95`)
- Return-instead-of-raise: functions return result dicts with a `"reason"` string field rather than throwing — `verify_claim` returns `{"passed": False, "reason": "feature not grounded for this class"}` (`src/verifier.py:45-52`); `run_pipeline` returns a full report with `"gate_decision": "RELEASE"|"ESCALATE"` and `"reason"` codes like `"passed_verification"`, `"low_faithfulness"`, `"false_action_critical"`, `"atypical_action_critical"`, `"no_verifiable_claims"` (`src/pipeline.py:323-349`)

**Gating convention (domain-specific):** the pipeline never throws for bad explanations — it returns an ESCALATE report. Write new pipeline/verifier logic the same way; only loading/config errors raise.

## Logging

**Framework:** `print()` only — the `logging` module is not used anywhere.

**Patterns:**
- Step markers and progress: `print("=== STARTING CERTIFICATE 3 FAITHFULNESS AUDIT (CONCURRENT/RESUME) ===")` (`scripts/run_faithfulness_audit.py:98`)
- Per-case progress: `print(f"[{idx+1}/100] Auditing Index {original_idx} (True Class: {true_class})...")` (`scripts/run_faithfulness_audit.py:30`)
- Audit scripts print `[{status}] name: evidence` lines (PASS/FAIL) and accumulate results in a list (`scripts/repo_audit.py:12-15`, `scripts/harsh_audit.py:21-24`)
- tqdm progress bars for long loops: `for f in tqdm(files):` (`src/train_baseline.py:43`)
- Keep `print` output greppable: one line per event, values formatted with fixed precision

## Comments

**When to Comment:**
- Explain the *why* (data provenance, split consistency, thread-safety), not the *what*:
  - `# Stratified 80/20 train/test split, same as train_baseline.py` (`src/compute_thresholds.py:22`)
  - `# Mutex to protect writes to audit_results and the JSON file` (`scripts/run_faithfulness_audit.py:23`)
  - `# pacing pacing pacing` (`scripts/run_faithfulness_audit.py:175`)
- Section banners with `# ====` for multi-part scripts (`scripts/repo_audit.py:27-29`)
- A comment marks deliberate domain rules in data files: `# RULE: no hardcoded numeric thresholds are invented here...` (`src/knowledge_base.py:3-5`)

**JSDoc/TSDoc:** None in the demo; no comment docstrings in JSX.

**Docstrings:**
- Module-level docstrings for audit scripts explaining scope: `scripts/repo_audit.py:1-4`, `scripts/harsh_audit.py:1-5`
- Function docstrings with a one-line summary in `src/llm_client.py` (all five functions, e.g. `src/llm_client.py:40-43`)
- Test functions use numbered scenario docstrings describing the case under test — see `tests/test_verifier.py:82-86`

## Function Design

**Size:** Functions are small-to-medium; the largest are the prompt-builder branches in `run_pipeline` (`src/pipeline.py:160-250`). Audit aggregation lives in one long `main()` (`scripts/run_faithfulness_audit.py:97-329`).

**Parameters:**
- Optional heavy inputs default to `None` and are lazily loaded:
  ```python
  def verify_explanation(recording_features, predicted_class, claims, action_critical_idx=None, thresholds=None, calibration=None):
      if thresholds is None:
          thresholds = load_thresholds()
  ```
  (`src/verifier.py:129-133`) — this is what makes the unit tests able to inject `MOCK_THRESHOLDS`/`MOCK_CALIBRATION`
- `run_pipeline(sample_row, true_label=None, mock=False, grounding_mode="sample")` — `mock` accepts a fixture key string or `True` (`src/pipeline.py:64,152-154`)

**Return Values:**
- Plain dicts with stable, documented keys; never tuples except `generate_with_usage` which returns `(content, prompt_tokens, completion_tokens)` (`src/llm_client.py:40`)
- `generate` wraps `generate_with_usage` discarding usage via `content, _, _ = ...` (`src/llm_client.py:104`)
- Reports always include the same core keys (`predicted`, `confidence`, `gate_decision`, `reason`, `explanation`, `per_claim`) — see the report dict at `src/pipeline.py:350-362`

## Module Design

**Exports:** Modules are either single-purpose scripts (`main()` + `if __name__ == "__main__"`) or importable function libraries (`src/verifier.py`, `src/features.py`, `src/knowledge_base.py`). No classes in Python.

**Barrel Files:** None. `src/` is not a package (no `__init__.py`); imports work only via the `sys.path.insert(0, ...)` hack repeated in `src/verifier.py:6`, `src/pipeline.py:10`, all `scripts/*`, and all `tests/*`.

**Config & data files:**
- All artifacts are read/written from `outputs/` relative to the repo root; every module re-derives `repo_dir` with `Path(__file__).resolve().parent.parent` (e.g. `src/pipeline.py:65`)
- JSON is written with `open(path, "w", encoding="utf-8")` + `json.dump(..., indent=2)` (audit files) or `indent=4` (`src/compute_thresholds.py:101-102`, `src/train_baseline.py:108-109`); Python 3.5+ f-strings and `pathlib` everywhere
- Model binaries are `joblib.dump`/`joblib.load` from `outputs/` (`src/train_baseline.py:87,102`, `src/pipeline.py:77-78`)

## Determinism & Research Hygiene

- Fixed seed convention: `random_state=42` in every split/model — `src/train_baseline.py:77,95,124,131`, `src/compute_thresholds.py:27`, `src/leakage_check.py:132,157`, `src/explain_shap.py:44`, and the test fixture `tests/test_pipeline.py:28-30`
- Thread-safety when writing shared state: `threading.Lock()` around list append + JSON save (`scripts/run_faithfulness_audit.py:84-89`), thread-local token usage via `threading.local()` (`src/llm_client.py:25`)
- Secrets live only in `.env` (gitignored — `.gitignore:13`); `src/llm_client.py:13` loads it with `load_dotenv(dotenv_path=repo_dir / ".env", override=True)`. `.env.example` exists as a template. Never hardcode API keys; `src/llm_client.py:21-22` substitutes `"dummy_key"` when missing so imports don't crash
- Directory hygiene enforced by `.gitignore`: `data/`, `*.joblib`, `scratch/`, `.env`, `__pycache__/` are never committed

---

*Convention analysis: 2026-08-04*
