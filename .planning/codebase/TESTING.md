---
title: Testing Patterns
date: 2026-08-04
focus: quality
---

# Testing Patterns

**Analysis Date:** 2026-08-04

## Test Framework

**Runner:**
- pytest 8.3.1 — pinned in `requirements.txt` (`pytest==8.3.1`)
- No config file: no `pytest.ini`, `pyproject.toml`, `setup.cfg`, or `tox.ini` — pytest runs with pure defaults
- No `conftest.py`; no fixtures are shared across test modules
- No CI pipeline (no `.github/`), no pre-commit config

**Assertion Library:**
- Plain Python `assert` statements — no `assertpy`/`hamcrest`
- Assertion messages are f-string expressions of the actual value, e.g. `assert res["passed"] is True, f"Claim failed: {res['reason']}"` (`tests/test_verifier.py:89`)

**Run Commands:**
```bash
python -m pytest tests/ -v          # Run all tests (verbosely — used by scripts/repo_audit.py:256)
python -m pytest tests/test_verifier.py   # Single file
```
- `scripts/repo_audit.py` runs the suite automatically as audit check #8 (`scripts/repo_audit.py:255-262`) and fails the audit if any test fails
- No watch mode, no coverage target, no markers defined

## Test File Organization

**Location:**
- All tests live in a flat `tests/` directory at the repo root; there is no `tests/unit/` vs `tests/integration/` split
- No `__init__.py` in `tests/` — pytest's rootdir discovery picks up `test_*.py` files

**Naming:**
- `test_<module_under_test>.py`:
  - `tests/test_verifier.py` → `src/verifier.py`
  - `tests/test_pipeline.py` → `src/pipeline.py`
  - `tests/test_audit.py` → `outputs/faithfulness_audit_summary.json` (artifact test)
  - `tests/test_ablation.py` → `outputs/faithfulness_audit_classonly.json` (artifact test)

**Structure:**
```
tests/
├── test_verifier.py    # Unit tests: 9 tests, module-level mock fixtures
├── test_pipeline.py    # Pipeline tests: 7 tests, @pytest.fixture + real features.csv
├── test_audit.py       # Output artifact contract test (1 test)
└── test_ablation.py    # Output artifact contract test (1 test)
```

**Import pattern (no conftest):** every test file repeats the path hack before importing `src`:
```python
# tests/test_verifier.py:7-12
script_dir = Path(__file__).resolve().parent
repo_dir = script_dir.parent
sys.path.insert(0, str(repo_dir / "src"))

from verifier import verify_claim, verify_explanation
from knowledge_base import KNOWLEDGE_BASE
```

## Test Structure

**Suite Organization:** plain module-level `def test_*()` functions. No test classes, no parametrize, no markers.

```python
# tests/test_verifier.py:82-90
def test_true_claim():
    """
    1. TRUE claim: pick a real LIV recording, assert a claim that its spectral_rolloff_85
       is 'low' -> should PASS.
    """
    claim = {"feature": "spectral_rolloff_85", "direction": "low"}
    res = verify_claim(MOCK_LIV_RECORDING, "LIV", claim, MOCK_THRESHOLDS, MOCK_CALIBRATION)
    assert res["passed"] is True, f"Claim failed: {res['reason']}"
    assert res["actual_value"] == MOCK_LIV_RECORDING["spectral_rolloff_85"]
```

**Patterns:**
- Scenario docstrings: `tests/test_verifier.py` documents each case as a numbered scenario ("1. TRUE claim ... 4. MIXED explanation") — keep this style when adding verifier tests
- Multi-assert tests that walk a single scenario end-to-end (e.g. `test_false_claim` checks claim level AND explanation level, `tests/test_verifier.py:92-105`)
- Grouping related scenarios in one test function with inline section comments (`# Test explanation level escalation...` `tests/test_verifier.py:102`)
- Float equality uses tolerance: `assert abs((release_pct + escalate_pct) - 100.0) < 1e-5` (`tests/test_audit.py:26`); `abs(sg_mean - 0.96) < 0.02` in `scripts/repo_audit.py:133`
- Tests read real artifact files from `outputs/` and assert on required keys: `required_keys = ["faithfulness", "gate_outcomes", ...]` loop (`tests/test_audit.py:17-19`)

## Fixtures

**Fixture pattern (real-data fixture):**
```python
# tests/test_pipeline.py:16-33
@pytest.fixture
def base_liv_sample():
    features_csv = repo_dir / "outputs" / "features.csv"
    assert features_csv.exists(), f"features.csv not found at {features_csv}"

    df = pd.read_csv(features_csv)
    y_str = df['class']
    class_names = sorted(y_str.unique())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    y = y_str.map(class_to_idx).values

    _, df_test, _, _ = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )
    liv_subset = df_test[df_test['class'] == 'LIV']
    return liv_subset.iloc[0].to_dict()
```
- The fixture re-derives the exact 80/20 stratified split used at training time (`random_state=42`) so it samples from the true test set — keep this split consistent with `src/train_baseline.py:76-78`
- Only ONE fixture exists in the whole suite; `test_pipeline_impulsive_*` inline-reads `features.csv` instead (`tests/test_pipeline.py:77-90`)

## Mocking

**Framework:** No mocking library at all — no `unittest.mock`, `pytest-mock`, or `monkeypatch` anywhere in `tests/`. Mocking is done with (1) hand-written data fixtures and (2) a `mock=` mode flag in the pipeline.

**Pattern 1 — module-level mock data dicts** (`tests/test_verifier.py:16-80`):
```python
MOCK_THRESHOLDS = {
    "global": {
        "spectral_rolloff_85": {"p10": 1000.0, "p25": 1200.0, "p50": 1600.0, "p75": 2000.0, "p90": 2200.0},
        ...
    },
    "classes": {...},
    "distinctive": {...}
}
MOCK_LIV_RECORDING = {"spectral_rolloff_85": 1000.0, "skewness": -0.1, ...}
```
- These are injected through the `thresholds=`/`calibration=` keyword parameters of `verify_claim`/`verify_explanation`/`check_sample_faithfulness`, which default to `None` and lazily load the real files (`src/verifier.py:129-133`)

**Pattern 2 — mock fixture mode in the pipeline** (`src/pipeline.py:15-62`):
```python
MOCK_FIXTURES = {
    "correct": {
        "claims": [
            {"feature": "spectral_rolloff_85", "direction": "low", "action_critical": False},
            {"feature": "spectral_centroid", "direction": "low", "action_critical": False}
        ],
        "explanation": "suction valve leakage (LIV) is confirmed as ..."
    },
    "flipped": {...}, "critical": {...}, "atypical_non_critical": {...},
    "atypical_critical": {...}, "impulsive_unsupported": {...}, "impulsive_supported": {...}
}
```
- `run_pipeline(sample_row, true_label="LIV", mock="correct")` skips the LLM call entirely (`src/pipeline.py:152-154`) — the pipeline tests drive every gate path (`RELEASE`/`ESCALATE` + each `reason` code) without a network call or API key

**What to Mock:**
- The LLM call (via `mock=` fixtures) — never hit `NVIDIA_API_KEY` endpoints in tests
- Threshold/calibration data (via kwargs) when the test targets verifier logic in isolation

**What NOT to Mock:**
- Real pipeline artifacts: `outputs/features.csv` is read directly by `tests/test_pipeline.py`; the audit JSONs are read by `tests/test_audit.py` and `tests/test_ablation.py`
- `KNOWLEDGE_BASE` from `src/knowledge_base.py` — tests import and use the real KB (`tests/test_verifier.py:13`, `tests/test_verifier.py:181-184`)

## Fixtures and Factories

**Test Data:**
- Verifier test data is module-level dict constants with a comment tying them to reality: `# Mock calibration aligned with thresholds and new grounding features` (`tests/test_verifier.py:41`); `# Mock LIV recording features matching the low directions` (`tests/test_verifier.py:72`)
- Sample rows for the pipeline come from the real feature matrix (see fixture above) with targeted mutations to force conditions:
  ```python
  # tests/test_pipeline.py:59-61
  sample = dict(base_liv_sample)
  sample["skewness"] = 1.0  # Force it to be high (global_p75 is around -0.0005)
  ```

**Location:**
- Mock data lives inline in the test files (module scope) or in `src/pipeline.py` `MOCK_FIXTURES` — there is no `tests/fixtures/` directory and no JSON fixture files

## Coverage

**Requirements:** None enforced. No coverage tool installed, no `--cov` config, no `pytest-cov` in `requirements.txt`.

**View Coverage:** Not available. If added, the natural command would be `python -m pytest tests/ --cov=src`.

## Test Types

**Unit Tests (verifier logic):** `tests/test_verifier.py` — exercises `verify_claim`, `verify_explanation`, `check_sample_faithfulness` with mock thresholds/calibration. Covers: true claims, false claims, ungrounded features, neutral features, mixed explanations with score thresholds (0.8 gate), critical-claim escalation, glossary completeness, and the vacuous-truth guard (`test_zero_claims_escalates`, `tests/test_verifier.py:187-207`).

**Integration Tests (pipeline + artifacts):** `tests/test_pipeline.py` runs the full explain→verify→gate chain with mocked LLM output against real model artifacts and a real feature sample — asserts `gate_decision` and `reason` for each fixture. `tests/test_audit.py` and `tests/test_ablation.py` are artifact contract tests validating committed experiment outputs (file exists, required keys, percentages sum to 100, 100 cases present, verifier actually fires in class-only mode).

**E2E Tests:** Not used. LLM-dependent behaviors are covered by mock-mode integration tests; ad-hoc live checks exist as plain scripts instead of pytest tests:
- `scripts/test_llm.py` — smoke-tests the NVIDIA NIM API key/endpoint
- `scripts/verify.py`, `scripts/verify_phase1.py` — dataset/model artifact verification scripts
- `scripts/repo_audit.py` — the 34-check "full repo audit" that also runs the pytest suite

## Current Suite Status (2026-08-04)

- **9 of 16 tests pass** on a fresh checkout: all of `tests/test_verifier.py` (9) pass without any artifacts beyond `outputs/feature_glossary.json`
- **7 tests fail** — every test in `tests/test_pipeline.py` raises `FileNotFoundError: Baseline model binaries or feature names not found. Run training first.` (`src/pipeline.py:74-75`) because `outputs/xgb_model.joblib`, `outputs/scaler.joblib` are gitignored (`*.joblib` in `.gitignore:11`) and not committed
- The README claims "16 tests, all passing" (`README.md:68`); that state holds only after `python src/train_baseline.py` has produced the model artifacts locally
- This means `scripts/repo_audit.py` check #8 (all tests pass) also fails on a fresh clone until training artifacts exist

## Common Patterns

**Async Testing:** Not applicable — no async code in the pipeline; concurrency is thread-based (`ThreadPoolExecutor` in `scripts/run_faithfulness_audit.py:179`) and is not unit-tested.

**Error Testing:**
```python
# tests/test_verifier.py:113-115 — asserting the reason string of a failed claim
res = verify_claim(MOCK_LIV_RECORDING, "LIV", claim, MOCK_THRESHOLDS, MOCK_CALIBRATION)
assert res["passed"] is False
assert res["reason"] == "feature not grounded for this class"
```
- Failure is modeled as a dict field (`passed: False` + `reason:`) rather than exceptions — assert on the `reason` string
- Escalation paths are asserted via `escalate`/`gate_decision` flags: `assert exp_res["escalate"] is True` (`tests/test_verifier.py:104`)

**Vacuous-truth guard test (regression test):** `test_zero_claims_escalates` verifies 0-claim explanations produce `faithfulness == 0.0` and `escalate == True` for both `verify_explanation` and `check_sample_faithfulness` (`tests/test_verifier.py:187-207`) — added as the regression test for commit `67495b7` ("Fix vacuous-truth bug").

## Demo Frontend

- The React demo (`demo/`) has **no tests at all** — no test script in `demo/package.json` (only `dev`, `build`, `preview`), no jest/vitest/playwright dependencies
- Note: `demo/src/components/*.jsx` import shared data from `../data/content` (e.g. `demo/src/components/VerificationDemo.jsx:4`), but that module is absent from the repo, so the demo cannot currently build

---

*Testing analysis: 2026-08-04*
