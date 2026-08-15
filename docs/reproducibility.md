# Local reproduction and verification contract

This project verifies **structured feature-direction claims**, not all factual
statements in prose. It does not certify the classifier's diagnosis, causal
reasoning, maintenance recommendations, or the LLM's assignment of criticality.

## Rebuild a local baseline

Use Python 3.12 and install `requirements-research.txt` in your virtual environment. Download the audio with `python src/download_data.py`. From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\reproduce_baseline.py --output-dir outputs\reproducibility\my-baseline --verify-audio all
```

The output directory must be new. The command makes no network requests and does
not overwrite `outputs/features.csv`, historical models, audits, plots, or reports.
It hashes all 1,800 local WAV files, checks sample identities and feature order,
compares extracted features with the saved CSV, and trains a new scaler and XGBoost
classifier in the running environment. It verifies saved training percentiles and
calibration directions, and repeats the original five-fold CV with preprocessing
fitted separately inside each fold. The held-out split and full-data CV are
separate estimates; the CV is not a tuning step for the held-out model.

The run records packages, Python/platform, Git revision and dirty status, source
and input hashes, a source snapshot, explicit train/test indices and portable recording IDs, CV fold
indices, predictions, metrics, model parameters, and output hashes. `output_sha256.json`
is written last: an incomplete run must not be treated as a completed baseline.
`environment.freeze.txt` is an inventory of the existing environment, including
unrelated packages; it is not a tested cross-platform dependency lockfile. The public replay and test dependencies are checked separately by CI.

## Use the compatible model

Historical `outputs/scaler.joblib` records scikit-learn 1.9.0 while the available
environment has 1.7.2. The pipeline now refuses version-mismatched estimators.
It does not suppress the warning or rewrite the old pickle. Select a newly trained
artifact directory explicitly:

```powershell
$env:VFD_ARTIFACT_DIR = (Resolve-Path outputs\reproducibility\my-baseline).Path
.\.venv\Scripts\python.exe scripts\run_pipeline_demo.py
```

Python callers can instead pass `artifact_dir=...` to `run_pipeline`. The default
demo is offline and uses mock responses. Without an explicit path/environment
variable, the pipeline still looks in the historical `outputs/` directory and
reports the compatibility error. The newly trained models reproduce the reported
95.56% test accuracy and 97.28% mean five-fold CV accuracy in this environment.

## Strict gate, version `strict-v1`

- Generated responses require strict JSON with exactly `claims` and `explanation`.
  Complete JSON code fences are accepted. Duplicate JSON keys, unsupported claim
  fields, duplicate features, missing fields, and non-boolean criticality fail.
- Each generated claim must state a grounded feature and `high` or `low`.
  The lower-level verifier also supports explicit `neutral`/`normal` and `elevated`
  aliases for compatibility; arbitrary or vague directions fail.
- Values and percentile bounds must be finite numbers, with p25 < p75. Missing,
  degenerate or reversed bounds and missing calibration fail closed. Values equal
  to p25 are low; values equal to p75 are high.
- Release requires at least one claim, every claim passing sample verification,
  no atypical action-critical claim, and no mechanism warning. Mechanism checking
  remains a conservative class-level lexical heuristic, not a proof of causal
  correctness. Atypical non-critical observations can still be released.
- On escalation, `explanation` is empty. `candidate_explanation` retains the
  unverified candidate for audit/debugging and must not be shown as verified text.
- The older `verify_claim`/`verify_explanation` APIs require class consistency as
  well as sample truth. The production pipeline uses `check_sample_faithfulness`
  and the shared gate to distinguish those two concepts.

`legacy-0.8` exists only as an explicit historical comparison policy. It is not
selectable as the production pipeline's gate. Historical audit scripts and result
files describe the old experiment; do not overwrite them with strict-policy runs.

## Replay saved LLM outputs without API calls

```powershell
.\.venv\Scripts\python.exe scripts\replay_audits.py --output-dir outputs\reproducibility\my-replay
```

The script verifies the three conditions use the same held-out sample IDs and
rechecks each stored claim against the feature matrix. It records hashes, changed
decisions, and paired recording-level bootstrap intervals (10,000 resamples, seed
42). These intervals assume independent recordings; recording-session dependence
is not resolved by this analysis.

| Condition | Historical mean faithfulness | Historical releases | Strict replay releases |
|---|---:|---:|---:|
| Sample-grounded | 0.9600 | 90/100 | 85/100 |
| Values-only | 0.6300 | 48/100 | 48/100 |
| Class-only | 0.3377 | 17/100 | 17/100 |

The five changed sample-grounded row indices are 226, 233, 353, 411 and 419;
each has a mechanism warning. Their structured claims still pass. No historical
gate decisions disagree with replay of the old policy. The stored records retain
per-claim fields and prose, but not complete raw model response envelopes or exact
prompts. Reconstructed response validation therefore cannot establish whether the
original response omitted fields. Historical latency/token counts do not measure
the corrected pipeline. No new LLM generations were performed.

## Tests and manuscript readiness

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Tests forbid HTTP calls and train temporary integration artifacts in the running
environment. If the system temp directory is unavailable, create a writable parent
and use a **new, disposable** `--basetemp` directory (pytest clears that directory).

The verified feature count is **84 = 7 time-domain + 5 spectral + 8 band-energy +
64 wavelet packet features**. Historical references to 154 or 85 are incorrect.
The current spectral roll-off is computed from cumulative FFT magnitude, not
cumulative squared magnitude; descriptions must match that definition.

The local manuscript distinguishes
the historical 80% gate from the corrected all-claims-pass gate. Strict replay
is reported separately from the original generation study's saved audit records. Claims that these experiments confirm absence of leakage or
guarantee safe deployment are unsupported. Session provenance, prose coverage,
independent-dataset validation, human usefulness, and clean-environment installation
remain open publication-readiness work. Archived PDFs and historical plots remain unchanged. The revised manuscript PDF
is regenerated separately from those experimental artifacts.

## Public artifact identities

The public feature matrix uses `class/filename.wav` in `file_path`. Numerical values and row order are preserved. This changes the CSV file hash relative to historical local manifests. Fresh replays record the current hash. Local manuscript builds, source snapshots, and old environment inventories are excluded from the public tree.
