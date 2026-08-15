# Verified Fault Diagnosis

**Checking an LLM's maintenance explanations against the machine's acoustic measurements.**

A fault classifier can predict a leaking valve. An LLM can turn that prediction into a convincing explanation. The harder question is whether the recording actually supports the explanation's claims.

This project follows that question through a working pipeline. XGBoost classifies compressor recordings, a knowledge base supplies fault context, and an LLM produces an explanation with structured feature claims. A deterministic verifier checks those claims against the recording and decides whether to release the explanation or escalate it for review.

The interesting part is the boundary between generation and verification. A claim can be true for a recording and still be unusual for the predicted fault. The gate keeps track of both.

**1,800 recordings · 84 acoustic features · 8 machine states · 64 passing tests**

## Try it locally

The quickest way to inspect the project is to replay the saved LLM responses through the verifier. Everything needed for replay is included in the repository. No API key or audio download is required.

Use Python 3.12.

```sh
git clone https://github.com/atharvoid/verified-fault-diagnosis.git
cd verified-fault-diagnosis
python -m venv .venv
```

Activate the environment with `.\.venv\Scripts\Activate.ps1` on Windows PowerShell, or `source .venv/bin/activate` on macOS/Linux. Then run

```sh
python -m pip install -r requirements.txt
python scripts/replay_audits.py --output-dir outputs/replay-first-run
```

The command prints a comparison of the historical and current gates, then saves a summary and individual claim decisions in the output directory. Use a new directory for each run. Dependency installation needs an internet connection. Replay runs offline.

## What the experiment shows

The explanation experiment compares three prompts on the same 100 held-out recordings. Each prompt receives a different amount of grounding information.

| Grounding supplied to the LLM | Mean claim faithfulness | Historical releases | Strict replay releases |
|---|---:|---:|---:|
| Class context only | 0.3377 | 17/100 | 17/100 |
| Feature values only | 0.6300 | 48/100 | 48/100 |
| Sample and class grounding | **0.9600** | 90/100 | **85/100** |

Faithfulness is the fraction of structured claims that agree with the recording's feature directions, averaged across recordings. An explanation with zero verifiable claims receives a score of zero.

More grounding was associated with higher faithfulness in this experiment. The strict results come from rechecking the saved responses under the current policy. They are separate from the original LLM generation experiment.

The classifier achieved **95.56% held-out accuracy** and **97.28% mean five-fold cross-validation accuracy** on the IIT Kanpur dataset. Recording-session independence remains unverified, so these figures need further validation on independently acquired recordings.

## How the gate works

Each generated claim names a feature, a direction, and whether the LLM considers it action-critical. For example

```json
{
  "feature": "spectral_centroid",
  "direction": "low",
  "action_critical": true
}
```

The verifier compares the measured value with training-set percentiles. Values at or below p25 count as low, values at or above p75 count as high, and values in between are neutral. It also checks whether the claimed direction agrees with the calibrated fault-class pattern.

The `strict-v1` policy releases an explanation only when

- At least one structured claim is present.
- Every claim passes sample verification.
- No action-critical claim is atypical for the predicted class.
- The mechanism-word heuristic raises no warning.

Otherwise, the pipeline returns `ESCALATE` with a reason. It clears the released explanation field and retains the candidate separately for inspection.

### Why the stricter policy matters

The earlier gate allowed an overall faithfulness score of 0.8. Four true claims could therefore let one false, non-critical claim through. The current gate requires every claim to pass, and a regression test covers that exact case.

Mechanism warnings now block release too. That change accounts for the five additional sample-grounded escalations in the replay. All their structured feature claims pass, but their prose triggers the class-level wording heuristic.

These are useful distinctions when reviewing an explanation. Passing feature checks establishes a narrow, inspectable property. Causal reasoning, diagnostic correctness, maintenance advice, and the LLM's choice of criticality still require separate evaluation.

## Engineering details worth a look

- **Validation fails closed.** Duplicate JSON keys, repeated features, unsupported fields, non-boolean criticality, non-finite values, and missing or degenerate thresholds are rejected. The response parser and release policy live together in a small contract module. [Verification contract](src/verification_contract.py)
- **Sample truth and class consistency are separate.** An unusual measurement can be real. The verifier records both properties so the gate can treat an atypical action-critical claim differently. [Verifier](src/verifier.py)
- **Replay is inspectable.** Each run records input and source hashes, changed decisions, and paired recording-level bootstrap intervals. Recording identities are portable across machines. [Replay script](scripts/replay_audits.py)
- **Tests cover failure paths.** The suite blocks HTTP requests, trains temporary compatible model artifacts, checks malformed responses and percentile boundaries, and checks the demos against the saved evidence. [Tests](tests/)

## Run the tests

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The public source tree was checked in a fresh Python environment with **64 tests passing**. The CI workflow is configured to run replay and tests on Windows and Linux, plus a production build of the React demo. [CI workflow](.github/workflows/ci.yml)

## Explore the demo

Open `presentation.html` in a browser for the standalone presentation. For the React and Three.js showcase, use

```sh
cd demo
npm ci
npm run build
npm run preview
```

The demos animate recorded examples and show their strict replay decisions. Candidate prose is labeled as unverified. Use the Python replay command to execute the verifier yourself. The React showcase requires WebGL and loads fonts externally.

## Scope and next steps

The current evidence comes from one compressor dataset. The next research steps are validation across independent recording sessions and machines, evaluation of claims expressed only in prose, and a study of whether the explanations help maintenance engineers.

The mechanism guard is a lexical heuristic based on class-level statistics. The saved experiments also lack complete raw response envelopes and exact prompts, which limits retrospective schema checks and exact regeneration.

Full audio verification, model rebuilding, and API-backed experiments use `requirements-research.txt`. The reproduction guide documents those workflows, artifact compatibility, and the distinction between historical experiments and strict replay. [Reproduction guide](docs/reproducibility.md)

## License and data

Project code is MIT licensed. Dataset-derived artifacts retain the original dataset license and required citation. Raw audio is fetched separately by `src/download_data.py`. [Code license](LICENSE) · [Dataset license](DATASET_LICENSE.txt)

Nishchal K. Verma, R. K. Sevakula, S. Dixit and A. Salour, *Intelligent Condition Based Monitoring using Acoustic Signals for Air Compressors*, IEEE Transactions on Reliability, vol. 65, no. 1, pp. 291–309, 2016. [Dataset source](https://www.mathworks.com/help/signal/ug/data-sets-for-signal-processing.html).
