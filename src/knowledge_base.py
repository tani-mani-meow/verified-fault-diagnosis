# src/knowledge_base.py
# Structured RAG knowledge base (Python dict, not a vector DB).
# RULE: no hardcoded numeric thresholds are invented here. Any numeric bound is
# computed at runtime from TRAINING-set per-class percentiles (deterministic),
# so Certificate 3 verifies LLM claims against real measured values.

KNOWLEDGE_BASE = {
    "LIV": {
        "name": "Leakage Inlet (Suction) Valve",
        "mechanism": (
            "The inlet/suction reed valve fails to seal during the compression "
            "stroke, so pressurized gas leaks back into the suction line. This "
            "produces a repeated impulsive event once per compression cycle and "
            "reduces volumetric efficiency."
        ),
        "acoustic_signature": {
            "time_domain": (
                "Periodic impulsive bursts synchronized to the compression cycle "
                "(observed as regularly spaced transient spikes in the waveform), "
                "raising impulsiveness metrics relative to Healthy."
            ),
            "frequency_domain": (
                "Broadband impulse energy with elevated mid-band content; energy "
                "concentrated in the lower-to-mid bands with a sharper roll-off at "
                "high frequency compared to Healthy."
            ),
        },
        # Features the LLM is ALLOWED to cite; verifier checks each against features.csv
        "grounding_features": {
            "kurtosis": "higher_than_healthy",          # impulsive transients
            "crest_factor": "higher_than_healthy",
            "band_energy_ratio_2": "elevated",           # ~0.75-1.5 kHz band
            "band_energy_ratio_3": "elevated",           # ~1.5-2.25 kHz band (top SHAP driver)
            "spectral_rolloff_85": "shifted",
            "zero_crossing_rate": "context_dependent",
        },
        "distinguishing_from": {
            "LOV": "LIV leaks on the SUCTION side; LOV leaks on the DISCHARGE side. "
                   "Discriminate primarily via the mid-band energy distribution and "
                   "roll-off, since the two are the closest confusable pair.",
            "Healthy": "Healthy shows uniform, non-impulsive noise (low kurtosis/crest factor).",
        },
        "threshold_source": "per_class_training_percentiles",
        "references": [
            "Verma, Sevakula, Dixit, Salour (2016), IEEE Trans. Reliability 65(1):291-309, DOI 10.1109/TR.2015.2459684",
        ],
    },

    "LOV": {
        "name": "Leakage Outlet (Discharge) Valve",
        "mechanism": (
            "The discharge/outlet reed valve fails to seal, so compressed gas leaks "
            "back from the discharge line into the cylinder during the suction "
            "stroke. Causes re-expansion losses and a cyclic impulsive signature on "
            "the discharge side."
        ),
        "acoustic_signature": {
            "time_domain": (
                "Cyclic impulsive events tied to the discharge phase; impulsiveness "
                "elevated vs Healthy but phased differently from LIV within the cycle."
            ),
            "frequency_domain": (
                "Elevated mid-band energy; the mid-band ratio and high-frequency "
                "roll-off differ subtly from LIV — this subtlety is the reason LIV/LOV "
                "is the main confusable pair in the baseline."
            ),
        },
        "grounding_features": {
            "kurtosis": "higher_than_healthy",
            "crest_factor": "higher_than_healthy",
            "band_energy_ratio_3": "elevated",
            "band_energy_ratio_4": "elevated",           # ~2.25-3.0 kHz band
            "spectral_centroid": "shifted_vs_healthy",
            "spectral_rolloff_85": "shifted",
        },
        "distinguishing_from": {
            "LIV": "Discharge-side vs suction-side leak. Key discriminators are the "
                   "relative weights of adjacent mid-bands (ratio_3 vs ratio_4) and "
                   "the spectral centroid.",
            "NRV": "NRV involves the non-return valve (system-level backflow), giving "
                   "a different low-band and cyclic pattern.",
        },
        "threshold_source": "per_class_training_percentiles",
        "references": [
            "Verma et al. (2016), IEEE Trans. Reliability 65(1):291-309.",
        ],
    },

    "NRV": {
        "name": "Non-Return Valve fault",
        "mechanism": (
            "The non-return (check) valve, which prevents backflow from the receiver/"
            "tank into the compressor, degrades or sticks. This alters the pressure "
            "boundary conditions and introduces backflow-related transients distinct "
            "from the inlet/outlet reed-valve leaks."
        ),
        "acoustic_signature": {
            "time_domain": (
                "Irregular/altered impulsive content associated with backflow through "
                "the check valve; differs from the clean per-cycle periodicity of LIV."
            ),
            "frequency_domain": (
                "Altered low-to-mid band energy distribution; NRV was cleanly "
                "separated in the baseline (recall ~0.96, precision 1.00), indicating "
                "a distinctive spectral fingerprint."
            ),
        },
        "grounding_features": {
            "band_energy_ratio_2": "distinctive",
            "band_energy_ratio_3": "distinctive",
            "wpt_energy_35": "distinctive",              # strong SHAP driver
            "wpt_energy_5": "distinctive",
            "kurtosis": "context_dependent",
            "spectral_flatness": "context_dependent",
        },
        "distinguishing_from": {
            "LIV": "NRV is a system check-valve fault, not a cylinder reed-valve leak; "
                   "the cyclic periodicity and mid-band balance differ.",
            "LOV": "Both affect discharge-side behavior, but NRV's fingerprint is more "
                   "separable (fewer confusions in the baseline confusion matrix).",
        },
        "threshold_source": "per_class_training_percentiles",
        "references": [
            "Verma et al. (2016), IEEE Trans. Reliability 65(1):291-309.",
        ],
    },
}
