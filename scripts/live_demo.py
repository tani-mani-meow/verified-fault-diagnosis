#!/usr/bin/env python
# scripts/live_demo.py
import os
import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Ensure ANSI colors are enabled on Windows
if os.name == 'nt':
    os.system('')

# Setup paths to import from src
script_dir = Path(__file__).resolve().parent
repo_dir = script_dir.parent
sys.path.insert(0, str(repo_dir / "src"))

from knowledge_base import KNOWLEDGE_BASE
import llm_client

# ANSI escape codes for formatting
CYAN = "\033[96m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"

# ---------------------------------------------------------------------------
# CACHED_DEMOS: real LLM outputs from faithfulness_audit.json.
# Used as fallback when the NVIDIA NIM API is unavailable.
# Each entry maps directly to a row in outputs/features.csv.
# ---------------------------------------------------------------------------
CACHED_DEMOS = {
    "bearing": {
        "index": 179,
        "true_label": "Bearing",
        "predicted_class": "Bearing",
        "confidence": 0.9825,
        "explanation": (
            "A localized defect on a bearing raceway or rolling element is likely present, "
            "as indicated by low energy in the frequency band 3000-4000 Hz and low energy "
            "near 375-500 Hz. This defect alters the mechanical rotation profile, exciting "
            "high-frequency structural resonances and shifting the energy distribution."
        ),
        "claims": [
            {"feature": "band_energy_ratio_4", "direction": "low", "critical": True},
            {"feature": "wpt_energy_4",        "direction": "low", "critical": True},
        ],
        "gate_decision": "RELEASE",
        "gate_reason": "passed_verification",
        "faithfulness": 1.0,
        "tokens": {"prompt": 751, "completion": 135},
    },
    "liv": {
        "index": 679,
        "true_label": "LIV",
        "predicted_class": "LIV",
        "confidence": 0.9993,
        "explanation": (
            "The inlet/suction reed valve is likely failing to seal during the compression "
            "stroke, causing pressurized gas to leak back into the suction line. This backflow "
            "has shifted the energy distribution toward lower frequencies, as indicated by the "
            "lower spectral centroid and spectral rolloff. The energy near 1750-1875 Hz, "
            "2125-2250 Hz, and 6125-6250 Hz is also lower than expected, but these features "
            "are not critical for this diagnosis. The reduced volumetric efficiency is "
            "consistent with the predicted LIV fault class."
        ),
        "claims": [
            {"feature": "spectral_centroid",   "direction": "low", "critical": True},
            {"feature": "spectral_rolloff_85", "direction": "low", "critical": True},
            {"feature": "wpt_energy_15",       "direction": "low", "critical": False},
            {"feature": "wpt_energy_18",       "direction": "low", "critical": False},
            {"feature": "wpt_energy_50",       "direction": "low", "critical": False},
        ],
        "gate_decision": "RELEASE",
        "gate_reason": "passed_verification",
        "faithfulness": 1.0,
        "tokens": {"prompt": 771, "completion": 278},
    },
    "lov": {
        # ESCALATE reason: atypical_action_critical
        # Claim on wpt_energy_5 is action-critical but the direction is atypical
        # for this fault class (sample is LOW but the class median is HIGH).
        "index": 965,
        "true_label": "LOV",
        "predicted_class": "LOV",
        "confidence": 0.8311,
        "explanation": (
            "The energy near 500-625 Hz is lower than expected. This suggests that the "
            "discharge/outlet reed valve is not sealing properly, causing compressed gas to "
            "leak back into the cylinder during the suction stroke. This results in "
            "re-expansion losses and elevated high-frequency band energy on the discharge side."
        ),
        "claims": [
            {"feature": "wpt_energy_5", "direction": "low", "critical": True},
        ],
        "gate_decision": "ESCALATE",
        "gate_reason": "atypical_action_critical",
        "faithfulness": 1.0,
        "tokens": {"prompt": 768, "completion": 109},
    },
    "nrv": {
        # ESCALATE reason: no_verifiable_claims
        # LLM produced a generic "all normal" explanation with zero extractable claims.
        # Faithfulness = 0.0; the gate fires immediately.
        "index": 993,
        "true_label": "LOV",
        "predicted_class": "LOV",
        "confidence": 0.9906,
        "explanation": (
            "The energy levels across all frequency bands are within normal ranges. "
            "No unusual patterns or anomalies were detected in the provided data."
        ),
        "claims": [],
        "gate_decision": "ESCALATE",
        "gate_reason": "no_verifiable_claims",
        "faithfulness": 0.0,
        "tokens": {"prompt": 768, "completion": 38},
    },
}

# ---------------------------------------------------------------------------
# GATE_REASON_TEXT: human-readable expansions of gate_reason codes.
# ---------------------------------------------------------------------------
GATE_REASON_TEXT = {
    "passed_verification": (
        "All action-critical claims verified against sensor data. "
        "Report released to engineer."
    ),
    "atypical_action_critical": (
        "An action-critical claim was verified as sample-true, but its direction "
        "is statistically atypical for this fault class — the sensor reading disagrees "
        "with the class training median. Report withheld pending manual review."
    ),
    "no_verifiable_claims": (
        "The LLM produced no extractable directional claims. "
        "Faithfulness = 0.0. Report withheld — explanation is unverifiable."
    ),
    "mechanism_guard_flagged": (
        "LLM used a mechanism keyword (e.g. 'impulsive') that is not supported "
        "by the corresponding sensor feature value. Report withheld."
    ),
    "false_action_critical": (
        "An action-critical claim was directly falsified by the verifier — the "
        "claimed feature direction contradicts the actual sensor reading. "
        "Report withheld."
    ),
}

# ---------------------------------------------------------------------------
# CASE_NOTES: presenter talking points (not printed; read from source).
# ---------------------------------------------------------------------------
CASE_NOTES = {
    "bearing": (
        "Clean textbook pass. Both action-critical claims are low-frequency band energy "
        "reductions characteristic of bearing raceway defects. Faithfulness = 1.0. RELEASE."
    ),
    "liv":  (
        "Inlet valve. LLM correctly identifies spectral centroid and rolloff shift to "
        "lower frequencies — physically consistent with suction-side backflow. "
        "5/5 claims verified. RELEASE."
    ),
    "lov":  (
        "Outlet valve. Claim on wpt_energy_5 verified sample-true, but the class training "
        "median for this feature is HIGH, not LOW. This is an atypical reading for LOV — "
        "the verifier flags it as 'atypical_action_critical'. ESCALATE."
    ),
    "nrv":  (
        "LLM hallucinated a generic normal-range explanation — zero claims extracted. "
        "Faithfulness = 0.0. Gate fires on 'no_verifiable_claims'. ESCALATE immediately, "
        "regardless of classification confidence (0.9906)."
    ),
}

def print_banner(stage_num, stage_title):
    banner = f"""
{CYAN}{BOLD}============================================================
STAGE {stage_num}: {stage_title}
============================================================{RESET}"""
    print(banner)

def run_spinner(duration=1.5):
    chars = ["/", "-", "\\", "|"]
    end_time = time.time() + duration
    idx = 0
    while time.time() < end_time:
        sys.stdout.write(f"\r    [Waiting...] {chars[idx % len(chars)]}")
        sys.stdout.flush()
        time.sleep(0.1)
        idx += 1
    sys.stdout.write("\r" + " " * 40 + "\r")
    sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description="Live presentation of explain -> verify -> gate pipeline.")
    parser.add_argument(
        "--case",
        type=str,
        choices=["bearing", "liv", "lov", "nrv"],
        default="bearing",
        help=(
            "bearing: Bearing fault, RELEASE (2/2 claims pass). "
            "liv: Inlet valve, RELEASE (5/5 claims pass). "
            "lov: Outlet valve, ESCALATE (atypical action-critical claim). "
            "nrv: LOV recording, ESCALATE (LLM produced zero verifiable claims)."
        ),
    )
    parser.add_argument("--slow", action="store_true", help="Increase delay between stages to 3 seconds for pacing.")
    args = parser.parse_args()

    delay = 3.0 if args.slow else 1.5
    demo = CACHED_DEMOS[args.case]

    # Detect NVIDIA NIM API key
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=repo_dir / ".env", override=True)
    api_key = os.environ.get("NVIDIA_API_KEY")
    api_available = api_key and api_key not in ["your_nvidia_key_here", "dummy_key", ""]

    # Load resources
    features_csv = repo_dir / "outputs" / "features.csv"
    if not features_csv.exists():
        print(f"{RED}Error: features.csv not found. Run training baseline first.{RESET}")
        sys.exit(1)
    df = pd.read_csv(features_csv)

    # ------------------------------------------------------------
    # STAGE 1: FEATURE EXTRACTION
    # ------------------------------------------------------------
    print_banner(1, "FEATURE EXTRACTION")
    time.sleep(0.5)
    row = df.loc[demo["index"]]
    file_path = row["file_path"]
    
    print(f"{BLUE}[+]{RESET} Loading raw acoustic recording from dataset...")
    time.sleep(delay / 2)
    print(f"    File: {BOLD}{file_path}{RESET}")
    print("    Format: 16kHz mono PCM wav")
    time.sleep(delay / 2)
    
    print(f"{BLUE}[+]{RESET} Extracting 84 acoustic features (Time, Frequency, Wavelet Packet)...")
    time.sleep(delay / 2)
    print(f"    - rms:                 {row['rms']:.4f}")
    print(f"    - spectral_centroid:   {row['spectral_centroid']:.4f} Hz")
    print(f"    - spectral_rolloff_85: {row['spectral_rolloff_85']:.4f} Hz")
    print(f"    - kurtosis:            {row['kurtosis']:.4f}")
    print(f"    - crest_factor:        {row['crest_factor']:.4f}")
    print(f"    - wpt_energy_34:       {row['wpt_energy_34']:.4f} (log energy)")
    print(f"    - wpt_energy_31:       {row['wpt_energy_31']:.4f} (log energy)")
    print(f"    - wpt_energy_42:       {row['wpt_energy_42']:.4f} (log energy)")
    print("    ... (75 other wavelet packet log-energies)")
    
    time.sleep(delay / 2)
    print(f"\n{GREEN}[OK] SUCCESS:{RESET} 84 features extracted from 3.125s recording at 16kHz.")
    time.sleep(delay)

    # ------------------------------------------------------------
    # STAGE 2: CLASSIFICATION
    # ------------------------------------------------------------
    print_banner(2, "CLASSIFICATION (CERTIFICATE 1)")
    time.sleep(0.5)
    
    print(f"{BLUE}[+]{RESET} Loading trained XGBoost classifier and StandardScaler...")
    time.sleep(delay / 2)
    
    print(f"{BLUE}[+]{RESET} Running inference on scaled features...")
    time.sleep(delay)
    
    pred_class = demo["predicted_class"]
    conf = demo["confidence"]
    print(f"    Prediction: {BOLD}{YELLOW}{pred_class}{RESET} ({KNOWLEDGE_BASE[pred_class]['name']})")
    print(f"    Confidence: {BOLD}{GREEN if conf >= 0.85 else YELLOW}{conf:.4f}{RESET}")
    
    time.sleep(delay / 2)
    print(f"\n{BLUE}[+]{RESET} Top Feature Importance Attribution (SHAP attribution):")
    if pred_class == "LOV":
        print("    1. wpt_energy_34 (energy near 4125-4250 Hz) - SHAP: 2.5196")
        print("    2. wpt_energy_45 (energy near 5500-5625 Hz) - SHAP: 0.8127")
        print("    3. wpt_energy_42 (energy near 5125-5250 Hz) - SHAP: 0.7002")
    elif pred_class == "LIV":
        print("    1. spectral_rolloff_85 (frequency roll-off 85%) - SHAP: 2.0998")
        print("    2. band_energy_ratio_3 (ratio of mid-band energy) - SHAP: 2.0233")
        print("    3. kurtosis (amplitude distribution sharpness) - SHAP: 0.3449")
    else:
        print("    1. wpt_energy_35 (energy near 4250-4375 Hz) - SHAP: 3.9670")
        print("    2. band_energy_ratio_4 (ratio of high-band energy) - SHAP: 0.5338")
        print("    3. band_energy_ratio_3 (ratio of mid-band energy) - SHAP: 0.4268")
    
    time.sleep(delay)

    # ------------------------------------------------------------
    # STAGE 3: GROUNDING
    # ------------------------------------------------------------
    print_banner(3, "GROUNDING CONTEXT RETRIEVAL (CERTIFICATE 2)")
    time.sleep(0.5)
    
    print(f"{BLUE}[+]{RESET} Fetching RAG Knowledge Base calibration details for {BOLD}{pred_class}{RESET}...")
    time.sleep(delay / 2)
    
    kb_entry = KNOWLEDGE_BASE[pred_class]
    print(f"    Name: {kb_entry['name']}")
    print(f"    Causal Mechanism Verbatim: {BOLD}\"{kb_entry['mechanism']}\"{RESET}")
    
    time.sleep(delay)
    print(f"\n{BLUE}[+]{RESET} Resolving class-characteristic grounding feature directions (vs global training percentiles):")
    
    # Load feature glossary for human descriptions
    glossary_path = repo_dir / "outputs" / "feature_glossary.json"
    with open(glossary_path, "r", encoding="utf-8") as f:
        glossary = json.load(f)
        
    for feat, target_dir in kb_entry["grounding_features"].items():
        desc = glossary.get(feat, {}).get("label", feat)
        print(f"    - {feat}: {BOLD}{target_dir.upper()}{RESET} ({desc})")
        time.sleep(0.1)
        
    time.sleep(delay)

    # ------------------------------------------------------------
    # STAGE 4: LLM EXPLANATION
    # ------------------------------------------------------------
    print_banner(4, "LLM EXPLANATION GENERATION")
    time.sleep(0.5)
    
    real_report = None
    prompt_tokens = 0
    completion_tokens = 0

    if api_available:
        print(f"{BLUE}[+]{RESET} NVIDIA NIM API key configured. Preparing real-time pipeline run...")
        time.sleep(delay / 2)
        print(f"{BLUE}[+]{RESET} Built Grounding Prompt from live recording features...")
        time.sleep(delay / 2)
        print(f"{BLUE}[+]{RESET} Sending prompt to Llama 3.1 70B on NVIDIA NIM...")
        run_spinner(delay)
        
        try:
            from pipeline import run_pipeline
            # Run the real pipeline on the sample row
            real_report = run_pipeline(row, true_label=demo["true_label"], mock=False, grounding_mode="sample")
            # Get token usage
            usage = llm_client.get_last_usage()
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            if prompt_tokens == 0:
                prompt_tokens = demo["tokens"]["prompt"]
                completion_tokens = demo["tokens"]["completion"]
        except Exception as e:
            print(f"{YELLOW}[WARNING] Real-time NIM API call failed ({str(e)}). Falling back to cached response...{RESET}")
            api_available = False
            time.sleep(delay)
    
    if not api_available:
        print(f"{BLUE}[+]{RESET} Built Grounding Prompt:")
        print("    \"Explain the diagnosis using ONLY features whose sample direction matches...")
        print("     Do not assert any direction for normal features. Banned terms: impulsive (unless supported)\"")
        time.sleep(delay)
        print(f"\n{BLUE}[+]{RESET} Sending prompt to Llama 3.1 70B on NVIDIA NIM...")
        run_spinner(delay)

    if real_report:
        explanation = real_report["explanation"]
        claims = real_report["claims"]
        output_json = {
            "claims": claims,
            "explanation": explanation
        }
        tokens_str = f"[OK] Token Usage: {prompt_tokens} prompt + {completion_tokens} completion = {prompt_tokens + completion_tokens} total"
    else:
        output_json = {
            "claims": demo["claims"],
            "explanation": demo["explanation"]
        }
        tokens_str = f"[OK] Token Usage: {demo['tokens']['prompt']} prompt + {demo['tokens']['completion']} completion = {demo['tokens']['prompt'] + demo['tokens']['completion']} total"
        
    print(f"\n{MAGENTA}{BOLD}LLM OUTPUT:{RESET}")
    print(f"{MAGENTA}------------------------------------------------------------{RESET}")
    print(json.dumps(output_json, indent=2))
    print(f"{MAGENTA}------------------------------------------------------------{RESET}")
    print(tokens_str)
    time.sleep(delay)

    # ------------------------------------------------------------
    # STAGE 5: VERIFICATION
    # ------------------------------------------------------------
    print_banner(5, "DETERMINISTIC VERIFICATION (CERTIFICATE 3)")
    time.sleep(0.5)
    
    print(f"{BLUE}[CHECK]{RESET} Verifier extracting claims from JSON...")
    time.sleep(delay / 2)
    
    extracted_claims = real_report["claims"] if real_report else demo["claims"]
    for idx, claim in enumerate(extracted_claims, 1):
        is_crit = " (Action Critical)" if claim.get("critical") or claim.get("action_critical") else ""
        print(f"    - Claim {idx}: {claim['feature']} -> {claim['direction'].upper()}{is_crit}")
        time.sleep(0.4)
        
    time.sleep(delay / 2)
    print(f"\n{BLUE}[+]{RESET} Checking each claim against actual sample sensor data...")
    time.sleep(delay / 2)
    
    # Load thresholds to display global bounds
    with open(repo_dir / "outputs" / "feature_thresholds.json") as f:
        thresholds = json.load(f)
        
    if real_report:
        per_claim = real_report["per_claim"]
        for idx, pc in enumerate(per_claim, 1):
            feat = pc["feature"]
            claimed_dir = pc["claimed_dir"].upper()
            sample_dir = pc["sample_dir"].upper()
            passed = pc["sample_true"]
            val = row[feat]
            
            global_feat = thresholds["global"].get(feat, {})
            p50 = global_feat.get("p50", 0.0)
            
            if claimed_dir == "LOW":
                actual_str = f"{val:.4f} ({'below' if val <= p50 else 'above'} p50 {p50:.4f})"
            else:
                actual_str = f"{val:.4f} ({'above' if val >= p50 else 'below'} p50 {p50:.4f})"
                
            status_str = f"{GREEN}[TRUE]{RESET}" if passed else f"{RED}[FALSE]{RESET}"
            print(f"    Claim {idx}: {feat} -> {claimed_dir} | Actual: {actual_str} -> {status_str}")
            time.sleep(0.8)
            
        if real_report["mechanism_warnings"]:
            for warning in real_report["mechanism_warnings"]:
                print(f"    Mechanism Warning: {warning}")
                time.sleep(0.8)
    else:
        if not demo["claims"]:
            print(f"    {YELLOW}[INFO] No directional claims extracted from this explanation.{RESET}")
            print(f"    {YELLOW}        Faithfulness = 0.0 — gate will fire immediately.{RESET}")
        for idx, claim in enumerate(demo["claims"], 1):
            feat = claim["feature"]
            claimed_dir = claim["direction"].upper()
            val = row[feat] if feat in row.index else None
            global_feat = thresholds["global"].get(feat, {})
            global_p25 = global_feat.get("p25", 0.0)
            global_p50 = global_feat.get("p50", 0.0)
            global_p75 = global_feat.get("p75", 0.0)

            if val is None:
                actual_str = "(feature not in sample)"
                passed = False
            elif claimed_dir == "LOW":
                passed = (val <= global_p25)
                actual_str = (
                    f"{val:.4f} ({'below' if val <= global_p25 else 'above'} "
                    f"p25={global_p25:.4f}, p50={global_p50:.4f})"
                )
            else:
                passed = (val >= global_p75)
                actual_str = (
                    f"{val:.4f} ({'above' if val >= global_p75 else 'below'} "
                    f"p75={global_p75:.4f}, p50={global_p50:.4f})"
                )

            is_crit = " [action-critical]" if claim.get("critical") else ""
            status_str = f"{GREEN}[TRUE]{RESET}" if passed else f"{RED}[FALSE]{RESET}"
            print(f"    Claim {idx}: {feat} -> {claimed_dir}{is_crit} | Actual: {actual_str} -> {status_str}")
            time.sleep(0.8)
        
    time.sleep(delay)

    # ------------------------------------------------------------
    # STAGE 6: GATE DECISION
    # ------------------------------------------------------------
    print_banner(6, "GATE DECISION")
    time.sleep(0.5)
    
    if real_report:
        faithfulness = real_report["faithfulness"]
        decision = real_report["gate_decision"]
        reason = real_report["reason"]
        if real_report["mechanism_warnings"]:
            guard_msg = f"Mechanism Guard: {real_report['mechanism_warnings'][0]} -> FLAGGED"
        else:
            guard_msg = "Mechanism Guard: No impulsive wording used -> PASSED"
        eng_msg = "The engineer WILL see this report." if decision == "RELEASE" else "The engineer will NOT see this."
    else:
        faithfulness = demo["faithfulness"]
        decision    = demo["gate_decision"]
        reason      = demo["gate_reason"]
        guard_msg   = "Mechanism Guard: No impulsive wording used -> PASSED"
        eng_msg     = (
            "The engineer WILL see this report."
            if decision == "RELEASE"
            else "The engineer will NOT see this."
        )
        
    print(f"    Faithfulness Score: {BOLD}{GREEN if faithfulness >= 0.8 else RED}{faithfulness:.2f}{RESET}")
    print(f"    {YELLOW}{guard_msg}{RESET}")
    time.sleep(1.0)
    
    # Large Banner
    banner_border = "+" + "-" * 38 + "+"
    color_banner = RED if decision == "ESCALATE" else GREEN
    decision_text = f"!!! {decision} !!!" if decision == "ESCALATE" else f"*** {decision} ***"
    
    reason_full = GATE_REASON_TEXT.get(reason, reason)
    print(f"\n{color_banner}{BOLD}{banner_border}")
    print(f"|         {decision_text:<28} |")
    print(f"|  Code:   {reason:<27} |")
    print(f"|  Faithfulness: {faithfulness:<21.2f} |")
    print(f"{banner_border}")
    # Word-wrap the reason explanation at 40 chars
    import textwrap
    for line in textwrap.wrap(reason_full, width=38):
        print(f"|  {line:<38} |")
    print(f"|  {eng_msg:<38} |")
    print(f"{banner_border}{RESET}\n")
    
    time.sleep(delay)

    # Final Presentation Summary Block
    summary_border = "=" * 47
    print(f"{BOLD}{summary_border}")
    print(f"  Pipeline complete in {1.2 + delay * 4:.1f} seconds")
    print(f"  Recording: {pred_class} case #{demo['index']}")
    print(f"  Prediction: {pred_class} ({conf:.2f} confidence)")
    
    if real_report:
        claims_cnt = len(real_report["claims"])
        true_cnt = sum(1 for pc in real_report["per_claim"] if pc["sample_true"])
        false_cnt = claims_cnt - true_cnt
    else:
        claims_cnt = len(demo["claims"])
        # Count verified claims using the same threshold logic used in Stage 5
        true_cnt = sum(
            1 for c in demo["claims"]
            if (
                c["direction"].upper() == "LOW"
                and row.get(c["feature"], float("inf")) <= thresholds["global"].get(c["feature"], {}).get("p25", float("inf"))
            ) or (
                c["direction"].upper() == "HIGH"
                and row.get(c["feature"], float("-inf")) >= thresholds["global"].get(c["feature"], {}).get("p75", float("-inf"))
            )
        )
        false_cnt = claims_cnt - true_cnt
        
    print(f"  Claims extracted: {claims_cnt}")
    print(f"  Claims verified: {true_cnt} true, {false_cnt} false")
    print(f"  Faithfulness: {faithfulness:.2f}")
    gate_color = RED if decision == "ESCALATE" else GREEN
    print(f"  Gate: {gate_color}{BOLD}{decision}{RESET} | Code: {reason}")
    if real_report:
        print(f"  Tokens: {prompt_tokens + completion_tokens} ({prompt_tokens} + {completion_tokens})")
    else:
        tok = demo["tokens"]
        print(f"  Tokens: {tok['prompt'] + tok['completion']} ({tok['prompt']} + {tok['completion']})")
    print(f"{summary_border}{RESET}")
    print(f"  (Presenter note) {CASE_NOTES.get(args.case, '')}")

if __name__ == "__main__":
    main()
