---
phase: 02-code-review-command
reviewed: 2026-07-22T13:30:00Z
depth: deep
files_reviewed: 37
files_reviewed_list:
  - verified-fault-diagnosis/src/build_glossary.py
  - verified-fault-diagnosis/src/build_grounding.py
  - verified-fault-diagnosis/src/calibrate_kb.py
  - verified-fault-diagnosis/src/compute_thresholds.py
  - verified-fault-diagnosis/src/download_data.py
  - verified-fault-diagnosis/src/explain_shap.py
  - verified-fault-diagnosis/src/features.py
  - verified-fault-diagnosis/src/inspect_data.py
  - verified-fault-diagnosis/src/knowledge_base.py
  - verified-fault-diagnosis/src/leakage_check.py
  - verified-fault-diagnosis/src/llm_client.py
  - verified-fault-diagnosis/src/pipeline.py
  - verified-fault-diagnosis/src/train_baseline.py
  - verified-fault-diagnosis/src/verifier.py
  - verified-fault-diagnosis/tests/test_ablation.py
  - verified-fault-diagnosis/tests/test_audit.py
  - verified-fault-diagnosis/tests/test_pipeline.py
  - verified-fault-diagnosis/tests/test_verifier.py
  - verified-fault-diagnosis/scripts/audit_fixes.py
  - verified-fault-diagnosis/scripts/compare_conditions.py
  - verified-fault-diagnosis/scripts/harsh_audit.py
  - verified-fault-diagnosis/scripts/live_demo.py
  - verified-fault-diagnosis/scripts/recompute_audits.py
  - verified-fault-diagnosis/scripts/repo_audit.py
  - verified-fault-diagnosis/scripts/run_ablation_audit.py
  - verified-fault-diagnosis/scripts/run_faithfulness_audit.py
  - verified-fault-diagnosis/scripts/run_feature_ablation.py
  - verified-fault-diagnosis/scripts/run_pipeline_demo.py
  - verified-fault-diagnosis/scripts/run_valuesonly_audit.py
  - verified-fault-diagnosis/scripts/run_valuesonly_pilot.py
  - verified-fault-diagnosis/scripts/test_llm.py
  - verified-fault-diagnosis/scripts/verify.py
  - verified-fault-diagnosis/scripts/verify_phase1.py
  - verified-fault-diagnosis/demo/index.html
  - verified-fault-diagnosis/demo/vite.config.js
  - verified-fault-diagnosis/demo/tailwind.config.js
  - verified-fault-diagnosis/demo/postcss.config.js
findings:
  critical: 2
  warning: 6
  info: 6
  total: 14
status: issues_found
---

# Phase 02: Code Review Report

**Reviewed:** 2026-07-22T13:30:00Z  
**Depth:** deep  
**Files Reviewed:** 37  
**Status:** issues_found  

## Summary

This review covers the complete "Verified Fault Diagnosis" project — a three-certificate trust protocol for acoustic-based predictive maintenance. The codebase is well-structured with strong separation of concerns, thorough testing (16 tests across 4 test files), and a sophisticated verification pipeline. The project's core logic around grounding, verification, and auditing is sound.

However, several defects were identified that undermine the correctness of the pipeline's safety guarantees and could cause incorrect behavior in edge cases. The most critical issue is in `pipeline.py` where the "no_verifiable_claims" guard (intended to escalate 0-claim explanations) is unreachable due to a logic ordering problem — the subsequent `faithfulness < 0.80` check always overwrites it. Because `check_sample_faithfulness` returns `faithfulness=0.0` for empty claim sets, the `low_faithfulness` reason fires instead. The `no_verifiable_claims` escalation reason is dead code.

A second blocker exists in `leakage_check.py` where dead code with incorrect split logic is executed (though subsequently overwritten) before the corrected version, creating confusion and a maintenance hazard.

---

## Critical Issues

### CR-01: Pipeline trust gate — "no_verifiable_claims" reason is unreachable dead code

**File:** `verified-fault-diagnosis/src/pipeline.py:362-391`  
**Issue:** The zero-claims guard at line 362 sets `gate_decision = "ESCALATE"` and `reason = "no_verifiable_claims"`, but this is NOT connected to the subsequent `if/elif/else` chain. After it executes, the code falls through to:

```python
# Line 372-374:
if false_action_critical:
    ...
elif faithfulness < 0.80:
    gate_decision = "ESCALATE"
    reason = "low_faithfulness"     # <-- OVERWRITES "no_verifiable_claims"
```

When `check_sample_faithfulness()` receives an empty claims list, it returns `faithfulness=0.0` and `per_claim=[]`. Since `faithfulness=0.0 < 0.80`, the `elif` at line 376 ALWAYS fires, overwriting the reason to `"low_faithfulness"`. The `"no_verifiable_claims"` reason can never be reached.

This is documented as a known issue in `scripts/audit_fixes.py:126-128` and a fix script `scripts/recompute_audits.py` was written — but the source file `pipeline.py` was NEVER actually fixed. The recompute script post-processes audit JSON files rather than fixing the root cause.

**Fix:** Restructure the gate logic so the zero-claims check is an early return, or integrate it into the elif chain as the first priority check. In `pipeline.py` around line 362:

```python
# 0. If no verifiable claims -> ESCALATE, reason="no_verifiable_claims"
if len(per_claim) == 0:
    gate_decision = "ESCALATE"
    reason = "no_verifiable_claims"
    # Skip remaining checks — no claims to evaluate
elif false_action_critical:
    gate_decision = "ESCALATE"
    reason = "false_action_critical"
elif faithfulness < 0.80:
    gate_decision = "ESCALATE"
    reason = "low_faithfulness"
else:
    ...
```

### CR-02: Dead code with incorrect split assignment in leakage_check.py

**File:** `verified-fault-diagnosis/src/leakage_check.py:106-122`  
**Issue:** The Contiguous-Split Stress Test contains dead code that initially computes the split in the wrong direction (lines 106-110 assign the last 20% to training and first 80% to testing). A comment on lines 112-114 correctly identifies this, and lines 115-122 overwrite with the correct assignment.

However, the dead code at lines 106-110 IS executed before being overwritten. While the final result is correct (because `train_indices` and `test_indices` are reassigned to `[]` at lines 116-117), the presence of this dead code:
1. Confuses future readers about intent
2. Was likely a source of incorrect results before the fix was added
3. Could cause issues if someone later removes what appears to be a redundant block

Code at lines 106-110:
```python
train_indices = []   # Line 116
test_indices = []    # Line 117
# Buggy code at 106-110 immediately above is never used, but still executed
```

**Fix:** Remove the dead code block (lines 106-110) entirely. The corrected version at lines 115-122 does the right thing.

**Fix:**
Remove lines 106-114 (from `train_indices = []` through to the comment). Keep only the corrected version:

```python
train_indices = []
test_indices = []
for c in classes:
    class_idx = df_sorted[df_sorted['class'] == c].index.tolist()
    split_point = int(len(class_idx) * 0.8)
    train_indices.extend(class_idx[:split_point])
    test_indices.extend(class_idx[split_point:])
```

---

## Warnings

### WR-01: `download_data.py` uses `raise e` instead of bare `raise`, losing traceback

**File:** `verified-fault-diagnosis/src/download_data.py:25,66`  
**Issue:** Two exception handlers use `raise e` instead of bare `raise`. This creates a new traceback that discards the original exception's call stack, making debugging harder.

```python
# Line 22-25:
except Exception as e:
    print(f"Error downloading from URL {zip_url}: {e}")
    raise e  # Loses original traceback

# Line 64-66:
except Exception as e:
    print(f"Error extracting zip: {e}")
    raise e  # Loses original traceback
```

**Fix:** Change `raise e` to bare `raise`:

```python
except Exception as e:
    print(f"Error downloading from URL {zip_url}: {e}")
    raise  # Preserves original traceback
```

### WR-02: `build_glossary.py` uses `assert` statements that can be disabled with `python -O`

**File:** `verified-fault-diagnosis/src/build_glossary.py:117,123,125-126,131,134-135`  
**Issue:** The WPT tiling sanity checks use Python `assert` statements. Running with `python -O` (optimized mode) disables all asserts, silently skipping these critical validation checks.

```python
assert len(wpt_bands) == 64, f"Expected 64 WPT bands, got {len(wpt_bands)}"
assert wpt_bands[0][1] == 0, "WPT tiling does not start at 0 Hz"
```

**Fix:** Replace `assert` with explicit `if/raise` checks or use a dedicated validation function:

```python
if len(wpt_bands) != 64:
    raise ValueError(f"Expected 64 WPT bands, got {len(wpt_bands)}")
if wpt_bands[0][1] != 0:
    raise ValueError("WPT tiling does not start at 0 Hz")
```

### WR-03: Inconsistent `sys.path` manipulation across modules — `append` vs `insert(0, ...)`

**Files:**
- `verified-fault-diagnosis/src/calibrate_kb.py:8` — uses `sys.path.append`
- `verified-fault-diagnosis/src/verifier.py:8` — uses `sys.path.append`
- `verified-fault-diagnosis/src/pipeline.py:11` — uses `sys.path.append`
- `verified-fault-diagnosis/tests/test_pipeline.py:12` — uses `sys.path.insert(0, ...)`
- `verified-fault-diagnosis/tests/test_verifier.py:10` — uses `sys.path.insert(0, ...)`
- `verified-fault-diagnosis/scripts/run_faithfulness_audit.py:18` — uses `sys.path.insert(0, ...)`

**Issue:** Source modules use `sys.path.append()` (adds to end of path) while test/script modules use `sys.path.insert(0, ...)` (adds to front). Using `append` means installed packages or stdlib take priority over the local `src/` directory in case of name collisions. The test files use the safer `insert(0, ...)` approach.

**Fix:** Standardize on `sys.path.insert(0, str(repo_dir / "src"))` across all files for consistent module resolution priority.

### WR-04: `calibrate_kb.py` and `verifier.py` have inconsistent `normalize_kb_dir` vs `normalize_claim_dir` logic

**Files:**
- `verified-fault-diagnosis/src/calibrate_kb.py:18-24` (`normalize_kb_dir`)
- `verified-fault-diagnosis/src/verifier.py:34-40` (`normalize_claim_dir`)

**Issue:** Two functions perform semantically similar normalization but recognize different sets of synonyms:

`normalize_kb_dir` (KB-side):
```python
if "elevated" in d or "high" in d or "higher" in d:
    return "high"
elif "low" in d or "lower" in d:
    return "low"
```

`normalize_claim_dir` (verifier-side):
```python
if any(w in d for w in ["elevated", "high", "higher", "shifted", "widened", "moderately"]):
    return "high"
elif any(w in d for w in ["low", "lower"]):
    return "low"
```

The verifier recognizes `"shifted"`, `"widened"`, and `"moderately"` as "high" synonyms, but `normalize_kb_dir` does not. While KB directions are currently hardcoded as `"high"`/`"low"` in `knowledge_base.py`, this inconsistency could cause silent calibration mismatches if future KB entries use these additional terms.

**Fix:** Extract a shared `normalize_direction()` function into a common utility module (or add it to `knowledge_base.py`) and import it in both `calibrate_kb.py` and `verifier.py`.

### WR-05: `llm_client.py` caps retries at 10 but sleeps up to `2^9 * (2+1) = 1536 seconds` per failed request

**File:** `verified-fault-diagnosis/src/llm_client.py:40-107`  
**Issue:** The exponential backoff in `generate_with_usage` starts with `delay=2.0` and doubles each iteration. With `max_retries=10`, the final wait is `2.0 * 2^9 = 1024` seconds (~17 minutes), plus `random.uniform(0,1)`. Total worst-case time for a single request: ~34 minutes (sum of all delays). This could cause pipeline timeout issues when using `timeout=60` for the HTTP request itself.

Additionally, line 78 handles 429/503/504 inside `r.status_code` checks, but lines 88-93 handle them again in the `requests.exceptions.HTTPError` handler. The retry logic is duplicated — lines 77-84 handle specific status codes inline, and lines 88-93 handle the same codes via the HTTPError exception path. This means a 429/503/504 response is retried TWICE per attempt (once in the inline check, once in the exception handler after `r.raise_for_status()` is called on line 87).

**Fix:** Remove the inline status code check (lines 77-84) in favor of just the exception handler, or vice versa:

```python
# Option: use only exception handler
r = requests.post(url, json=data, headers=headers, timeout=60)
if r.status_code == 200:
    ...
else:
    r.raise_for_status()  # Will raise HTTPError for non-200
```

And rely entirely on the `except requests.exceptions.HTTPError` handler for retry logic.

### WR-06: `live_demo.py` has hardcoded threshold comparisons that may not match actual data

**File:** `verified-fault-diagnosis/scripts/live_demo.py:333-363`  
**Issue:** The cached demo fallback code performs directional checks using hand-coded logic that defaults `passed = True` for features other than `kurtosis`, `spectral_centroid`, and `spectral_rolloff_85` (line 356). For any unrecognized feature, `passed` is unconditionally set to `True` regardless of actual values:

```python
else:
    val = row[feat]
    passed = True  # <-- Always passes for unrecognized features!
    ...
```

And for LOV's cached demo, `spectral_centroid` is hardcoded to pass when `args.case == "lov"` (line 344) regardless of its actual value.

**Fix:** Either generalize the verification logic to check all features against their thresholds, or explicitly list verification rules for all cached claim features.

---

## Info

### IN-01: Unused import `os` in multiple files

**Files:** `verified-fault-diagnosis/src/build_glossary.py:2`, `verified-fault-diagnosis/src/build_grounding.py:2`, `verified-fault-diagnosis/src/calibrate_kb.py:2`, `verified-fault-diagnosis/src/explain_shap.py:2`  
**Issue:** `os` is imported but never used in these files (they use `pathlib.Path` instead). Minor cleanup item.

### IN-02: Magic number `random_state=42` duplicated across 6+ files

**Files:** Multiple (`compute_thresholds.py`, `train_baseline.py`, `explain_shap.py`, `leakage_check.py`, `run_faithfulness_audit.py`, `run_ablation_audit.py`, `run_valuesonly_audit.py`, `run_valuesonly_pilot.py`, `audit_fixes.py`, `harsh_audit.py`, ...)  
**Issue:** The magic number `42` for `random_state` is used in many files with no single-source-of-truth constant. If the seed ever needs to change, every file must be updated individually. Define `RANDOM_SEED = 42` in a shared constants module.

### IN-03: `test_pipeline.py:test_pipeline_atypical_non_critical` comments say skewness value 1.0 is high, but uses wrong reasoning

**File:** `verified-fault-diagnosis/tests/test_pipeline.py:59-60`  
**Issue:** The comment says `# Force it to be high (global_p75 is around -0.0005)`. The skewness value 1.0 being high relative to `global_p75 ≈ -0.0005` means `1.0 >= -0.0005` which is True, so `sample_dir = "high"`. This works, but the comment's parenthetical is misleading — `p75 ≈ -0.0005` means a skewness of 1.0 is clearly above it. The comment could be clearer about why `sample_dir` becomes "high".

### IN-04: `compute_thresholds.py` hardcodes class count `8` instead of using `len(classes)`

**File:** `verified-fault-diagnosis/src/compute_thresholds.py`  
**Issue:** While `len(classes)` is computed as `len(df['class'].unique())` at line 33, the distinctive features logic at line 86 assumes there are exactly 8 classes (Hardwired `class_medians[0][0]`, `class_medians[1][0]`, `class_medians[-2][0]`, `class_medians[-1][0]`). This works for the current 8-class dataset but would break with different dataset configurations. The code should reference `len(classes)` or 2 explicitly, but it's fine as an implicit assumption.

### IN-05: `requirements.txt` — cannot read binary file

**File:** `verified-fault-diagnosis/requirements.txt`  
**Issue:** The file could not be read as text, suggesting it may be corrupted or in an unexpected encoding. This should be checked — if it's truly a binary file, `pip install -r` will likely fail.

### IN-06: Thread safety of `_feature_count_printed` in `features.py` has no other thread usage

**File:** `verified-fault-diagnosis/src/features.py:9,109-111`  
**Issue:** A `threading.Event()` is used to ensure the feature count is printed only once. However, the rest of the codebase never calls `extract_features` from multiple threads — all extraction is single-threaded (in `train_baseline.py` and `inspect_data.py`). The thread-safety mechanism adds complexity without demonstrated need. This is minor, but the feature could be simplified with a plain boolean flag.

---

_Reviewed: 2026-07-22T13:30:00Z_  
_Reviewer: the agent (gsd-code-reviewer)_  
_Depth: deep_
