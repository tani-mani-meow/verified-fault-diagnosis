---
phase: 02-code-review-command
fixed_at: 2026-07-22T15:00:00Z
review_path: verified-fault-diagnosis/REVIEW.md
iteration: 1
findings_in_scope: 8
fixed: 8
skipped: 0
status: all_fixed
---

# Phase 02: Code Review Fix Report

**Fixed at:** 2026-07-22T15:00:00Z
**Source review:** verified-fault-diagnosis/REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 8 (2 critical + 6 warning)
- Fixed: 8
- Skipped: 0

## Fixed Issues

### CR-01: Pipeline trust gate — "no_verifiable_claims" reason is unreachable dead code

**Files modified:** `src/pipeline.py`
**Applied fix:** Restructured the trust gate if/elif chain so the zero-claims check (`if len(per_claim) == 0`) is the first condition in a proper `if/elif/elif/else` chain. The `false_action_critical` computation is now done before the chain begins, and all subsequent checks use `elif`. This ensures that when `per_claim` is empty, the `"no_verifiable_claims"` reason is preserved and not overwritten by the `faithfulness < 0.80` check.

### CR-02: Dead code with incorrect split assignment in leakage_check.py

**Files modified:** `src/leakage_check.py`
**Applied fix:** Removed the buggy Contiguous-Split code block (the first `for c in classes` loop that assigned the last 20% to train and first 80% to test) along with the self-correcting comment block. Only the corrected loop (first 80% → train, last 20% → test) remains.

### WR-01: `download_data.py` uses `raise e` instead of bare `raise`, losing traceback

**Files modified:** `src/download_data.py`
**Applied fix:** Changed both `raise e` statements in exception handlers (lines 25 and 66) to bare `raise`, preserving the original exception's traceback for debugging.

### WR-02: `build_glossary.py` uses `assert` statements that can be disabled with `python -O`

**Files modified:** `src/build_glossary.py`
**Applied fix:** Replaced all 7 `assert` statements with explicit `if/raise ValueError(...)` checks:
- Total count check (line 117)
- WPT node width check (line 123)
- WPT node start frequency check (line 125)
- WPT node end frequency check (line 126)
- Contiguous tiling gap/overlap check (line 131)
- WPT start at 0 Hz bound check (line 134)
- WPT end at 8000 Hz bound check (line 135)
- Gray code index mismatch check (line 162)

### WR-03: Inconsistent `sys.path.append` vs `insert(0, ...)` across modules

**Files modified:** `src/pipeline.py`, `src/calibrate_kb.py`, `src/verifier.py`
**Applied fix:** Changed `sys.path.append(...)` to `sys.path.insert(0, ...)` in all three source modules to match the test/script convention of prioritizing the local `src/` directory in module resolution.

### WR-04: `calibrate_kb.py` and `verifier.py` have inconsistent `normalize_kb_dir` vs `normalize_claim_dir` logic

**Files modified:** `src/knowledge_base.py`, `src/calibrate_kb.py`, `src/verifier.py`
**Applied fix:** Extracted a shared `normalize_direction()` function into `knowledge_base.py` with the more comprehensive synonym recognition (including "shifted", "widened", "moderately"). Both `calibrate_kb.py` and `verifier.py` now import and call `normalize_direction` from `knowledge_base.py`. The old `normalize_kb_dir` and `normalize_claim_dir` functions have been removed.

### WR-05: Duplicated retry logic in `llm_client.py`

**Files modified:** `src/llm_client.py`
**Applied fix:** Removed the inline status code check (`elif r.status_code in [429, 503, 504]`) with embedded retry logic. The code now only checks for success (status 200) and calls `r.raise_for_status()` for all other responses. The retry logic is handled exclusively by the `except requests.exceptions.HTTPError` handler, eliminating the double-retry issue where 429/503/504 responses were retried twice per attempt.

### WR-06: `live_demo.py` hardcoded fallback with `passed = True`

**Files modified:** `scripts/live_demo.py`
**Applied fix:** 
- **spectral_centroid:** Removed the hardcoded `passed = True` override for LOV case. Now uses `global_p25`/`global_p75` comparisons matching the verifier's logic (`val <= p25` for LOW, `val >= p75` for HIGH). Case-specific branches only customize the display string.
- **spectral_rolloff_85:** Changed from `p50`-based comparison to `p25`-based comparison, matching the verifier's LOW claim logic.
- **else branch (general case):** Replaced unconditional `passed = True` with proper threshold-based verification: `val <= global_p25` for LOW claims, `val >= global_p75` for HIGH claims. This matches the `check_sample_faithfulness` function's logic.

---

_Fixed: 2026-07-22T15:00:00Z_
_Fixer: the agent (gsd-code-fixer)_
_Iteration: 1_
