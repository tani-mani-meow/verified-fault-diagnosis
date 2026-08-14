---
title: External Integrations
date: 2026-08-04
focus: tech
---

# External Integrations

**Analysis Date:** 2026-08-04

## APIs & External Services

**LLM Inference (NVIDIA NIM — primary external integration):**
- [NVIDIA NIM API](https://integrate.api.nvidia.com) — OpenAI-compatible `/chat/completions` endpoint used for explanation generation (Certificates 2 & 3). This is the project's only live external API.
  - Client: hand-rolled `requests` wrapper, no SDK — `src/llm_client.py`
  - Endpoint: `{BASE_URL}/chat/completions` (`src/llm_client.py:44`)
  - Auth: `Authorization: Bearer {NVIDIA_API_KEY}` header (`src/llm_client.py:46`); key read from env var `NVIDIA_API_KEY`
  - Model: `NVIDIA_MODEL` env var, default `meta/llama-3.1-70b-instruct` (`src/llm_client.py:17`)
  - Base URL: `NVIDIA_BASE_URL` env var, default `https://integrate.api.nvidia.com/v1` (`src/llm_client.py:18`)
  - Request params: `temperature: 0.0`, 60s timeout, request body `{model, messages: [system?, user]}` (`src/llm_client.py:56-60`)
  - Retry behavior: up to 10 attempts, exponential backoff starting at 2s (+jitter) on HTTP 429/503/504 and connection/timeout errors (`src/llm_client.py:62-96`)
  - Token usage: recorded per-thread via `threading.local()` (`LAST_USAGE`) and surfaced with `llm_client.get_last_usage()` (`src/llm_client.py:25-38`) — used by audit scripts to compute cost/latency stats
  - Rate limiting: `politeness_delay(seconds=1.0)` helper (`src/llm_client.py:107`); audit scripts additionally sleep 1s between cases and run only 3 concurrent workers (`scripts/run_faithfulness_audit.py:175-180`)
  - No-key fallback: missing/placeholder key is replaced with `"dummy_key"` so imports don't crash; `src/pipeline.py:156-158` raises `ValueError` if the key is unset when real mode is requested

  **Consumers of `llm_client`:**
  - `src/pipeline.py:252` — the explain step of the explain→verify→gate pipeline (3 prompt modes: `class_only`, `values_only`, `sample` — see `src/pipeline.py:160-250`)
  - `scripts/run_faithfulness_audit.py` — 100-case sample-grounded audit
  - `scripts/run_valuesonly_audit.py` / `scripts/run_valuesonly_pilot.py` — values-only ablation audits
  - `scripts/run_ablation_audit.py` — class-only ablation audit
  - `scripts/live_demo.py` — live demo mode (falls back to hardcoded `CACHED_DEMOS` taken from real audit outputs when the API is unavailable)
  - `scripts/test_llm.py` — API smoke test ("Reply with exactly the word: OK")

**Dataset Download (one-time fetch):**
- [MathWorks Air Compressor Dataset](https://www.mathworks.com/supportfiles/audio/AirCompressorDataset/AirCompressorDataset.zip) — IIT Kanpur acoustic dataset (Verma et al., IEEE Trans. Reliability 65(1):291-309, 2016). Downloaded with stdlib `urllib.request.urlretrieve` in `src/download_data.py:13-25`, extracted with zip-slip validation (`src/download_data.py:45-61`) into gitignored `data/AirCompressorDataset/` (8 class folders × 225 WAV files, 16 kHz). No auth required.

## Data Storage

**Databases:**
- None. No DBMS, ORM, or vector store. The RAG knowledge base is a static Python dict (`src/knowledge_base.py:7`). Persistence is file-based JSON under `outputs/`:
  - `outputs/feature_thresholds.json` — global/class percentiles (p10–p90) loaded by `src/verifier.py:9-18`
  - `outputs/kb_calibration.json` — per-class calibrated feature directions loaded by `src/verifier.py:20-29`
  - `outputs/feature_glossary.json` — human-readable feature labels (`src/pipeline.py:110-112`)
  - `outputs/features.csv` — the 154-feature matrix consumed by tests and audits
  - Audit artifacts: `outputs/faithfulness_audit*.json`, `outputs/condition_comparison_summary.json`, `outputs/harsh_audit_report.json`, etc.

**File Storage:**
- Local filesystem only. Model artifacts `outputs/xgb_model.joblib` / `outputs/scaler.joblib` (joblib, gitignored). No cloud storage (the `boto3`/`localstack`/`awscli-local` packages pinned in `requirements.txt` are unused).

**Caching:**
- None beyond the local dataset zip (`data/AirCompressorDataset.zip` skip-if-exists logic in `src/download_data.py:18-27`) and `scripts/live_demo.py`'s `CACHED_DEMOS` (offline fallback demo outputs).

## Authentication & Identity

**Auth Provider:**
- None. Only a single API key: `NVIDIA_API_KEY` from `.env` (gitignored), sent as a Bearer token to NVIDIA NIM. No OAuth, no user accounts, no JWT handling (PyJWT is pinned but unused).
- Dummy-key guardrail: `src/llm_client.py:20-22` substitutes `"dummy_key"` for missing/placeholder keys; `src/pipeline.py:156-158` refuses real-mode runs without a valid key.
- `scripts/repo_audit.py:39-45` greps git history/tracked files for `nvapi-` key patterns to prevent accidental key commits, and checks `.env.example` references NIM rather than Gemini (`scripts/repo_audit.py:199-204`).

## Monitoring & Observability

**Error Tracking:**
- None (no Sentry etc.).

**Logs:**
- Console `print` logging throughout; audit scripts persist per-case metrics (latency, prompt/completion tokens, gate decision) to `outputs/faithfulness_audit.json` and aggregate cost/latency stats to `outputs/faithfulness_audit_summary.json` (`scripts/run_faithfulness_audit.py:238-278`). Token-usage tracking lives in `llm_client.get_last_usage()`.

## CI/CD & Deployment

**Hosting:**
- None. Research repo; the only deployable is the static `demo/` Vite build (presentation site).

**CI Pipeline:**
- None (`no .github/` directory). Quality gate is manual: `python scripts/repo_audit.py`, which runs security checks, staleness checks, output-consistency checks (e.g. audit means ≈ 0.96/0.34/0.63), reproducibility checks (`.env.example`, `random_state=42`), and executes the pytest suite (`scripts/repo_audit.py:256-262`).

## Environment Configuration

**Required env vars (from `.env.example`):**

| Variable | Default | Purpose |
|---|---|---|
| `NVIDIA_API_KEY` | (required, placeholder `your_nvidia_key_here`) | Bearer token for NVIDIA NIM |
| `NVIDIA_MODEL` | `meta/llama-3.1-70b-instruct` | Chat model id |
| `NVIDIA_BASE_URL` | `https://integrate.api.nvidia.com/v1` | OpenAI-compatible endpoint root |

**Secrets location:**
- `.env` at repo root (gitignored via `.gitignore:17`); `.env.example` contains only placeholders. No secrets in any tracked file (verified by `scripts/repo_audit.py:39-45`).

## Webhooks & Callbacks

**Incoming:**
- None.

**Outgoing:**
- None (outbound calls only: NIM API and the one-time MathWorks zip download).

---

*Integration audit: 2026-08-04*
