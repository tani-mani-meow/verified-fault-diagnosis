---
title: Codebase Concerns
date: 2026-08-04
focus: concerns
---

# Codebase Concerns

**Analysis Date:** 2026-08-04

Scope: full repo. Findings are based on static analysis of `src/`, `scripts/`, `tests/`, `demo/`, `outputs/` and committed audit artifacts. No API keys/secrets were found in tracked files (grep for `nvapi-`, `sk-`, `AIza` patterns returned nothing).

## Tech Debt

**1. Three near-identical audit scripts (copy-paste duplication):**
- `scripts/run_faithfulness_audit.py`, `scripts/run_ablation_audit.py`, `scripts/run_valuesonly_audit.py` are ~90% identical (~230–330 lines each): same worker/queue/retry structure, same 100-case stratified selection (`train_test_split(fault_test, train_size=100, stratify=..., random_state=42)`), same case-schema dict, same summary aggregation. The scripts differ only in `grounding_mode` and output filename.
- Impact: any schema or gate-logic change must be applied in 3+ places; they have already drifted (e.g., `run_ablation_audit.py:57` reads `action_critical` while `scripts/harsh_audit.py:158` reads `critical`).
- Fix approach: extract a shared `scripts/audit_lib.py` with `run_audit(grounding_mode, out_file)` and a shared case-schema builder.

**2. Two parallel verification implementations in `src/verifier.py`:**
- `verify_claim()`/`verify_explanation()` (lines 31–158, thresholds-based) vs `check_sample_faithfulness()` (lines 160–261, sample-direction-based). Production code (`src/pipeline.py:295`) uses only `check_sample_faithfulness`; the older path is exercised only by `tests/test_verifier.py` and `scripts/harsh_audit.py:116`.
- Impact: two definitions of "truth" that can drift; the escalation threshold `0.80` is duplicated at `src/verifier.py:151` and `src/pipeline.py:335`.
- Fix approach: delete or clearly deprecate `verify_claim`/`verify_explanation`, or make `verify_explanation` delegate to `check_sample_faithfulness`; centralize the 0.80 threshold in one constant.

**3. Gate decision logic implemented three times:**
- `src/pipeline.py:327-348`, `src/verifier.py:150-152`, `scripts/recompute_audits.py:15-74` each re-implement the ESCALATE decision ordering (no_verifiable_claims → false_action_critical → low_faithfulness → atypical_action_critical).
- Fix approach: single `compute_gate(per_claim, faithfulness)` in `src/verifier.py`, imported by all three.

**4. Prompt construction duplicated 3x in `src/pipeline.py`:**
- The `class_only`/`values_only`/`sample` branches (lines 160–250) are three ~60-line copies of the system prompt differing only in context-table format and one instruction line.
- Impact: prompt-engineering changes must be applied 3×; risk of divergence (already visible: only the `values_only` prompt at line 204 says "do NOT mention any feature not in this table").
- Fix approach: template with f-string/`string.Template` keyed by `grounding_mode`.

**5. `interpolate_p40`/`interpolate_p60` duplicated:**
- Defined in both `src/calibrate_kb.py:11-15` and `src/build_grounding.py:9-13`. The "p60" pseudo-percentile (`p50 + 0.4*(p75-p50)`) also appears inline at `src/pipeline.py:303-304` and `scripts/harsh_audit.py:303,310-311`. Centralize in one module.

**6. Test fixtures embedded in production code:**
- `MOCK_FIXTURES` lives in `src/pipeline.py:15-62`, and `CACHED_DEMOS` in `scripts/live_demo.py:40-127`. Test/demo-only data pollutes the runtime module and bloats `pipeline.py` with `mock=...` branches.
- Fix approach: move mock fixtures to `tests/` and cached demo data to `demo_data.json` loaded by the script.

**7. `requirements.txt` is a full `pip freeze` dump:**
- Contains ~230 pinned transitive packages including unrelated bloat (`localstack`, `awscli-local`, `terraform-local`, `docker`, `git-filter-repo`, `PyMuPDF`, `pytesseract`, `streamlit`, `langchain`, `ollama`) that the project never imports. `src/` actually needs: numpy, scipy, pandas, scikit-learn, xgboost, librosa, pywt, shap, joblib, matplotlib, python-dotenv, requests, pytest, soundfile, tqdm.
- Impact: heavy/conflicting installs (e.g., `numpy==2.3.1` with `numba==0.65.1` and `opencv-python==4.13.0.92` risk wheel-resolution failures on some platforms), and no `requirements-dev.txt` split.
- Fix approach: regenerate a minimal `requirements.txt` from actual imports; pin only direct deps with `>=`.

**8. Uncommitted working-tree deletions:**
- `git status` shows `CODEBASE-REVIEW.md`, `CODEBASE_CONTEXT.md`, `REVIEW-FIX.md`, `REVIEW.md` deleted in the working tree but the deletions are not committed.

## Known Bugs

**1. `scripts/harsh_audit.py` Section 4 reads the wrong field name (6 FAILs in committed report):**
- `harsh_audit.py:158` checks `c.get("critical", False)`, but `check_sample_faithfulness` stores `action_critical` (`src/verifier.py:242-250`). Result: all 6 "Escalation ... reason=atypical_action_critical" checks FAIL with `n_critical_atypical=0` in `outputs/harsh_audit_report.json`, even though the escalations are genuine (gate logic in `src/pipeline.py:339-348` reads `action_critical` correctly).
- The bug was diagnosed in `scripts/audit_fixes.py:170-175` ("field 'critical' values: MISSING... field 'action_critical' values: ...") but the audit script was never fixed.
- Fix approach: read `pc.get("action_critical", False)` in `scripts/harsh_audit.py:158`, re-run, commit a passing report.

**2. `scripts/harsh_audit.py` Sections 1 & 10 validate against the wrong population (13 FAILs):**
- Stored thresholds are computed on the 80% TRAIN split (`src/compute_thresholds.py:24-29`), but `harsh_audit.py:45-46` recomputes percentiles on the FULL `features.csv`. All 5 global-threshold spot checks and 11 class-p50 checks fail on 0.001–0.01 differences (e.g., kurtosis recomputed=1.2854 stored=1.2729).
- `scripts/audit_fixes.py:77-81` confirmed this ("15 threshold FAILs were audit-script errors — checked wrong subset") — but again the script was not corrected.
- Fix approach: replicate the train split (`train_test_split(test_size=0.2, random_state=42, stratify=...)`) inside `harsh_audit.py` before recomputing.

**3. `scripts/harsh_audit.py` Section 12 is now guaranteed to FAIL:**
- Line 355 asserts ALL sample-grounded cases have faithfulness exactly 1.0. The vacuous-truth fix (commit `67495b7`) intentionally made 0-claim cases 0.0 — current `outputs/faithfulness_audit.json` has `min=0.0`, mean 0.96. The committed `outputs/harsh_audit_report.json` shows `min=1.0000` and therefore predates the fix; it is stale and its "19 FAILs" contradict the README's "34/34 repo audit passing" claim (that refers to `scripts/repo_audit.py`, a different tool, but the stale report is misleading).
- Fix approach: re-run `scripts/harsh_audit.py` against current outputs after fixing bugs 1 and 2, and regenerate the report.

**4. Infinite retry loop in audit workers:**
- `scripts/run_faithfulness_audit.py:162-176` (and the identical copies in `run_ablation_audit.py:201-215`, `run_valuesonly_audit.py:162-176`): a case that fails permanently (e.g., non-retryable 4xx from the LLM API after `llm_client`'s own 10 retries) is requeued forever — `q.put(item)` then immediately `q.get_nowait()` fetches it again. `q.join()` at line 181 never returns; the audit hangs indefinitely with no retry cap.
- Fix approach: track per-index retry counts and give up after N attempts, recording the case as failed in the JSON.

**5. `outputs/harsh_audit_report.json` committed with 19 FAILs:**
- The committed report records FAILs for the checks above. Anyone reading the repo sees an audit report that contradicts the README ("34/34 repo audit passing") and the paper numbers.

**6. Mislabeled cached demo case:**
- `scripts/live_demo.py:109-127`: the cache key `"nrv"` holds a recording whose `true_label`/`predicted_class` are `"LOV"` (index 993, confidence 0.9906). The presenter note (line 175-179) explains it as an LOV recording, but the `--case nrv` name is misleading.

**7. `normalize_direction` mis-maps "moderately":**
- `src/knowledge_base.py:299` maps any direction containing "moderately" to `"high"` even when unqualified (e.g., "moderately changed" → high). Since `verify_claim` (line 105-116) auto-passes claims whose normalized direction is `unspecified` unless "distinctive" is present, direction-normalization mistakes silently affect verification outcomes.

**8. `verify_claim` vacuous-pass loophole (unaddressed by the vacuous-truth fix):**
- `src/verifier.py:105-116`: an LLM claim with direction `unspecified` (e.g., "normal", "within range", "context-dependent") and no "distinctive" keyword passes unconditionally without checking the sample value, inflating `faithfulness_score`. The vacuous-truth fix (commit `67495b7`) only handled `n_claims == 0`.
- The `test_verifier.py:135` fixture `"context_dependent"` deliberately exercises this pass-through behavior.

## Security Considerations

**API key handling (`src/llm_client.py`):**
- Risk: when `NVIDIA_API_KEY` is missing, `llm_client.py:21-22` substitutes the literal string `"dummy_key"` and sends it as a Bearer token to the NVIDIA endpoint. Harmless (request fails auth) but masks misconfiguration; `scripts/run_pipeline_demo.py:119` and `scripts/live_demo.py:225` detect this with hardcoded string lists that must stay in sync.
- Current mitigation: key is loaded from gitignored `.env` (`llm_client.py:13`), never logged, sent over HTTPS; `scripts/repo_audit.py:39-45` greps tracked files for `nvapi-` patterns. Verified clean as of this analysis.
- Recommendations: raise early instead of sending `"dummy_key"`; centralize the placeholder-check in one helper (currently duplicated at `llm_client.py:21`, `scripts/run_pipeline_demo.py:119`, `scripts/live_demo.py:225`).

**Data supply chain (`src/download_data.py`):**
- Risk: the MathWorks zip is downloaded from a hardcoded URL (`download_data.py:13`) with no checksum/size verification; a compromised or changed archive would be silently extracted.
- Current mitigation: zip-slip path validation is present (`download_data.py:46-49,57-60`) — good.
- Recommendations: pin a SHA-256 of the archive.

**Command execution:**
- `scripts/repo_audit.py:19` uses `subprocess` with `shell=True`; commands are static strings so injection risk is low, but `2>nul` makes the script Windows-only (broken on POSIX).
- `scripts/live_demo.py:15` calls `os.system('')` on Windows purely to enable ANSI colors — benign but odd.

**Secrets hygiene:** `.gitignore` covers `.env` and `*.joblib`; `.env.example` (checked) contains only placeholder keys; no secrets found in tracked files. No `credentials.*`, `*.pem`, or key files exist in the repo.

## Performance Bottlenecks

**1. Feature extraction is slow and not resumable (`src/train_baseline.py`):**
- Problem: extracting 6-level WPT features (db4, `src/features.py:60-63`) for 1800 × 50k-sample recordings takes a long time, and any error aborts the whole run via `sys.exit(1)` (`train_baseline.py:57`) — previously extracted rows are lost since `features.csv` is only written after all 1800 files succeed (line 63).
- Files: `src/train_baseline.py:43-63`, `src/features.py`.
- Improvement path: write rows incrementally to CSV, or skip already-processed files by `file_path` on re-run.

**2. O(n²) audit file rewrites:**
- Problem: each successful case rewrites the entire audit JSON under a lock (`scripts/run_faithfulness_audit.py:84-89`, same in both sibling scripts) — 100 full serializations + disk writes per audit run, serializing threads on the lock.
- Improvement path: append incrementally or flush at the end/checkpoint every 10 cases.

**3. LLM retry backoff can stall audits for ~17 minutes:**
- `src/llm_client.py:62-96`: 10 retries, `delay` doubling from 2s with no cap (delays 2,4,8,...,512s), plus a 60s request timeout per attempt. On an API outage, one case can block ~17 min; with 3 workers, the audits stall. Combined with bug 4 above (no retry cap at the worker level) a failing case never terminates.
- Improvement path: cap `delay` at 30s, reduce `max_retries`, and add an overall deadline.

**4. `leakage_check.py` per-class random sampling:**
- `src/leakage_check.py:82-87` runs 1000 `np.random.choice` pairs per class (8000 samples total) — negligible, but the loop is un-vectorized; trivial to improve if runtime ever matters.

## Fragile Areas

**1. Tests are not hermetic — they require gitignored training artifacts:**
- `tests/test_pipeline.py` (fixture at lines 17-33) requires `outputs/features.csv`, and `run_pipeline` (`src/pipeline.py:70-80`) requires `outputs/xgb_model.joblib`, `outputs/scaler.joblib`, `outputs/feature_names.json`. The `.joblib` files are gitignored (`/.gitignore:11`) and absent in this working tree; `data/` is also absent, so a fresh clone cannot run the suite or reproduce the README's "16 tests, all passing".
- Files: `tests/test_pipeline.py`, `src/pipeline.py:70-80`, `.gitignore:1,11`.
- Safe modification: keep a fixtures-based mock path (`mock=True`) that skips model loading, or commit a tiny holdout model for tests.

**2. Tests assert data-dependent properties:**
- `tests/test_pipeline.py:35-41` asserts `faithfulness == 1.0` for the `"correct"` fixture, which holds only if the first test-split LIV row's `spectral_rolloff_85` and `spectral_centroid` are below p25 — true today, silently false if `features.csv` is regenerated from a differently-processed dataset.
- `tests/test_audit.py:17-27` and `tests/test_ablation.py:17` hardcode 100-case counts and exact summary keys; they fail if audits are re-run with a different sample size.

**3. Hardcoded result numbers in presentation assets:**
- `demo/src/components/HeroScene.jsx:58` ("95.56% Accuracy"), `demo/src/components/GradientScene.jsx:89-91` (0.34/0.63/0.96 curve heights), `presentation.html:278,314`, and `scripts/live_demo.py:283-294` (hardcoded SHAP importance values per class, lines 284-294) will silently go stale if models/audits are re-run.
- `scripts/live_demo.py:248-261` prints "84 features extracted" — currently accurate, but the count is hardcoded in prose.

**4. `scripts/repo_audit.py` bakes in specific results and history:**
- Asserts `sample_grounded mean ~0.96`, `class_only ~0.3377`, `values_only ~0.63` (lines 133-138, 168-179) with ±0.02 tolerance; requires exactly 100 cases per audit (line 157); and inspects a specific fix commit hash `67495b7` (line 119). Any legitimate re-run that shifts means past the tolerance (e.g., different LLM version) turns the "repo audit" red.
- Also Windows-specific `2>nul` shell redirections (lines 39-49, 66) break on POSIX.

**5. `src/knowledge_base.py` import-time side effects:**
- Module-level `print` (line 308) and `assert` (line 307) run on every import by any consumer (e.g., `src/pipeline.py:11`), polluting stdout of library users. The assert hard-codes `7` classes.
- Also contains a citation typo at line 96: "Veret al. (2016)" (LOV references).

**6. Working tree state vs README claims:**
- README claims completed phases and 34/34 repo audit passing, but `outputs/harsh_audit_report.json` (committed) records 19 FAILs and `git status` shows 4 uncommitted file deletions. The repo's audit story is inconsistent across artifacts.

## Scaling Limits

**Dataset/model:**
- Current capacity: 1800 recordings, 8 classes, 154-84 features (README says 154; `features.py` produces 84 — README §"Key results" says "154 acoustic features", inconsistent with `outputs/feature_ablation.json`'s 84 and `live_demo.py:261`'s "84 features").
- Limit: `num_class=8` hardcoded in `src/train_baseline.py:94,123`; `KNOWLEDGE_BASE` asserts 7 fault classes (`src/knowledge_base.py:307`); the stratified 100-case audit selection (`train_size=100`) is hardcoded in all three audit scripts. Any new class or dataset size breaks these.
- Scaling path: derive `num_class` from data, drive audit sample size from a constant/CLI flag, and make KB entries data-driven (`outputs/grounding_features.json`) instead of hand-maintained in `src/knowledge_base.py` (the KB's `grounding_features` dicts currently duplicate what `build_grounding.py` computes — two sources of truth; `calibrate_kb.py` reconciles them at build time).

**Concurrency:**
- Audit scripts use 3 worker threads against a single API key; rate limits are handled by sleeping 1s (`run_faithfulness_audit.py:175`) and exponential backoff in `llm_client.py`. No parallel-safe concerns beyond the shared `LAST_USAGE` thread-local (correctly used).

## Dependencies at Risk

- **`requirements.txt` full-freeze bloat** (see Tech Debt 7): pins like `numpy==2.3.1` + `numba==0.65.1` + `shap==0.52.0` + `xgboost==3.3.0` + `librosa==0.11.0` are mutually version-locked; a single upgrade cascade is likely. Unrelated heavy packages (localstack, docker, terraform-local, awscli-local, git-filter-repo, pytesseract, PyMuPDF, langchain, ollama) create install failure surface for no benefit.
- **LLM endpoint dependency:** the entire Certificate 2/3 evaluation depends on the NVIDIA NIM API (`NVIDIA_BASE_URL` default `https://integrate.api.nvidia.com/v1`, `llm_client.py:18`) and model `meta/llama-3.1-70b-instruct` (default at `llm_client.py:17`). All audit numbers (0.96/0.63/0.34) were generated against that specific model; switching providers/models invalidates the paper numbers. No provider abstraction exists beyond one client.
- **Model binaries are gitignored but required:** `*.joblib` in `.gitignore:11` while `outputs/*.json/png` are committed — the committed JSON reports cannot be regenerated without the absent binaries + absent `data/`, making results non-reproducible from the repo alone.

## Missing Critical Features

**Phase 4 / 5 (acknowledged in README):** cross-modal validation on MetroPT-3 and the IEEE paper are not started. The whole 100-case audit ran only on the IITK compressor dataset with a single LLM — the core claim (gradient CO < VO < SG) rests on one model, one dataset.

**No confidence-aware gating:** `predicted_class` confidence is computed (`src/pipeline.py:86-88`) and reported but never used in the gate decision; a 51% confidence prediction gets the same RELEASE path as a 99% one (`src/pipeline.py:323-348`).

**Healthy-class gap:** `KNOWLEDGE_BASE` has no "Healthy" entry, so any Healthy prediction short-circuits to ESCALATE with a canned message (`src/pipeline.py:93-108`) — no explanation generated for the most common real-world case.

**Mechanism guard is a substring blacklist:** the impulsiveness guard (`src/pipeline.py:299-317`) bans 8 English words ("impulsive", "impact", "spike", ...) via substring match on the lowercase explanation — trivial to bypass with synonyms ("thumping", "knocking", "pulsing") and prone to false positives (e.g., "impact wrench" in a maintenance context).

**No model-drift monitoring:** nothing re-validates the model or thresholds against newly acquired recordings; the thresholds are frozen at `outputs/feature_thresholds.json` from the original training split.

## Test Coverage Gaps

**Untested areas (High priority):**
- `src/llm_client.py` — retry/backoff logic, `LAST_USAGE` accounting, `dummy_key` fallback: no tests.
- `src/verifier.py::check_sample_faithfulness` — the PRODUCTION verifier path. Only the 0-claims edge is tested (`tests/test_verifier.py:187-207`). Untested: unspecified/`"distinctive"` direction handling (`verifier.py:105-116`), neutral sample_dir vs claim direction, `atypical`/`class_consistent` computation, per-claim `action_critical` propagation.
- `src/leakage_check.py`, `src/compute_thresholds.py`, `src/calibrate_kb.py`, `src/build_grounding.py` — no tests at all (only `tests/test_verifier.py::test_glossary_completeness` covers glossary completeness against the KB).
- Gate decision ordering — covered indirectly through `run_pipeline` mock fixtures (`tests/test_pipeline.py`), but there is no unit test for the gate function itself (it isn't factored out — see Tech Debt 3).
- `demo/` frontend — no tests; hardcoded result numbers (Fragile Areas 3).

**Risk:** regressions in the verifier's edge-case logic (the safety-critical core of the paper) would go unnoticed — the current suite would still pass if, e.g., the `unspecified`-direction auto-pass behavior changed.

**Priority:** High for `check_sample_faithfulness` edge cases; Medium for `llm_client` retry behavior (a change there affects audit cost/latency figures); Low for analysis scripts.

---

*Concerns audit: 2026-08-04*
