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
            "causes backflow that shifts the energy distribution toward lower "
            "frequencies and reduces volumetric efficiency."
        ),
        "acoustic_signature": {
            "time_domain": (
                "Decreased skewness vs the global distribution, indicating altered "
                "waveform symmetry due to suction backflow."
            ),
            "frequency_domain": (
                "Lower spectral centroid and spectral roll-off (85%) compared to the "
                "global baseline, reflecting a shift in energy away from high frequencies."
            ),
        },
        "grounding_features": {
            "spectral_rolloff_85": "low",
            "skewness": "low",
            "spectral_centroid": "low",
            "wpt_energy_15": "low",
            "wpt_energy_18": "low",
            "wpt_energy_50": "low"
        },
        "grounding_features_human": [
            "frequency below which 85% of energy lies (Hz) (spectral_rolloff_85) is LOW",
            "asymmetry of the amplitude distribution (skewness) is LOW",
            "brightness - center of spectral mass (Hz) (spectral_centroid) is LOW",
            "energy near 1750-1875 Hz (wpt_energy_15) is LOW",
            "energy near 2125-2250 Hz (wpt_energy_18) is LOW",
            "energy near 6125-6250 Hz (wpt_energy_50) is LOW"
        ],
        "distinguishing_from": {
            "LOV": "LIV leaks on the SUCTION side; LOV leaks on the DISCHARGE side. "
                   "Discriminate primarily via the mid-band energy distribution and "
                   "roll-off, since the two are the closest confusable pair.",
            "Healthy": "Healthy shows uniform, non-impulsive noise (low kurtosis/crest factor).",
        },
        "threshold_source": "data_selected_percentiles_p75_p25",
        "references": [
            "Verma, Sevakula, Dixit, Salour (2016), IEEE Trans. Reliability 65(1):291-309, DOI 10.1109/TR.2015.2459684",
        ],
    },

    "LOV": {
        "name": "Leakage Outlet (Discharge) Valve",
        "mechanism": (
            "The discharge/outlet reed valve fails to seal, so compressed gas leaks "
            "back from the discharge line into the cylinder during the suction "
            "stroke. Causes re-expansion losses and elevated high-frequency band "
            "energy on the discharge side."
        ),
        "acoustic_signature": {
            "time_domain": (
                "Elevated WPT high-band energy relative to global baseline due to "
                "discharge-related high frequency transients."
            ),
            "frequency_domain": (
                "Concentrated high-band energy (WPT bands 34, 31, 42, 33, 39, 5 are high)."
            ),
        },
        "grounding_features": {
            "wpt_energy_34": "high",
            "wpt_energy_31": "high",
            "wpt_energy_42": "high",
            "wpt_energy_33": "high",
            "wpt_energy_39": "high",
            "wpt_energy_5": "high"
        },
        "grounding_features_human": [
            "energy near 4125-4250 Hz (wpt_energy_34) is HIGH",
            "energy near 3750-3875 Hz (wpt_energy_31) is HIGH",
            "energy near 5125-5250 Hz (wpt_energy_42) is HIGH",
            "energy near 4000-4125 Hz (wpt_energy_33) is HIGH",
            "energy near 4750-4875 Hz (wpt_energy_39) is HIGH",
            "energy near 500-625 Hz (wpt_energy_5) is HIGH"
        ],
        "distinguishing_from": {
            "LIV": "Discharge-side vs suction-side leak. Key discriminators are the "
                   "relative weights of adjacent mid-bands (ratio_3 vs ratio_4) and "
                   "the spectral centroid.",
            "NRV": "NRV involves the non-return valve (system-level backflow), giving "
                   "a different low-band and cyclic pattern.",
        },
        "threshold_source": "data_selected_percentiles_p75_p25",
        "references": [
            "Veret al. (2016), IEEE Trans. Reliability 65(1):291-309.",
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
                "Altered check-valve backflow transients showing lower energy in specific WPT bands."
            ),
            "frequency_domain": (
                "Lower energy concentrated across WPT bands (WPT 35, 36, 38, 34, 30, 37 are low)."
            ),
        },
        "grounding_features": {
            "wpt_energy_35": "low",
            "wpt_energy_36": "low",
            "wpt_energy_38": "low",
            "wpt_energy_34": "low",
            "wpt_energy_30": "low",
            "wpt_energy_37": "low"
        },
        "grounding_features_human": [
            "energy near 4250-4375 Hz (wpt_energy_35) is LOW",
            "energy near 4375-4500 Hz (wpt_energy_36) is LOW",
            "energy near 4625-4750 Hz (wpt_energy_38) is LOW",
            "energy near 4125-4250 Hz (wpt_energy_34) is LOW",
            "energy near 3625-3750 Hz (wpt_energy_30) is LOW",
            "energy near 4500-4625 Hz (wpt_energy_37) is LOW"
        ],
        "distinguishing_from": {
            "LIV": "NRV is a system check-valve fault, not a cylinder reed-valve leak; "
                   "the cyclic periodicity and mid-band balance differ.",
            "LOV": "Both affect discharge-side behavior, but NRV's fingerprint is more "
                   "separable (fewer confusions in the baseline confusion matrix).",
        },
        "threshold_source": "data_selected_percentiles_p75_p25",
        "references": [
            "Verma et al. (2016), IEEE Trans. Reliability 65(1):291-309.",
        ],
    },

    "Bearing": {
        "name": "Bearing fault (rolling-element defect)",
        "mechanism": (
            "A localized defect on a bearing raceway or rolling element alters "
            "the mechanical rotation profile, exciting high-frequency structural "
            "resonances and shifting the energy distribution."
        ),
        "acoustic_signature": {
            "time_domain": "Impacts shifting low-frequency and band-energy content downward.",
            "frequency_domain": (
                "Altered energy distribution with low values in WPT bands 4, 5, 8, 25, "
                "band_energy_ratio_4, and dominant_frequency compared to the global baseline."
            ),
        },
        "grounding_features": {
            "wpt_energy_4": "low",
            "wpt_energy_5": "low",
            "wpt_energy_8": "low",
            "band_energy_ratio_4": "low",
            "wpt_energy_25": "low",
            "wpt_energy_9": "low"
        },
        "grounding_features_human": [
            "energy near 375-500 Hz (wpt_energy_4) is LOW",
            "energy near 500-625 Hz (wpt_energy_5) is LOW",
            "energy near 875-1000 Hz (wpt_energy_8) is LOW",
            "frequency band 3000-4000 Hz energy ratio (band_energy_ratio_4) is LOW",
            "energy near 3000-3125 Hz (wpt_energy_25) is LOW",
            "energy near 1000-1125 Hz (wpt_energy_9) is LOW"
        ],
        "distinguishing_from": {
            "Flywheel": "MAIN confusable pair. Bearing = higher-frequency impulsive "
                        "energy + higher kurtosis; Flywheel = lower-frequency, "
                        "rotational-harmonic energy. Discriminate via high-band ratios "
                        "and kurtosis.",
        },
        "threshold_source": "data_selected_percentiles_p75_p25",
        "references": ["Verma et al. (2016), IEEE Trans. Reliability 65(1):291-309."],
    },

    "Flywheel": {
        "name": "Flywheel fault (imbalance / looseness)",
        "mechanism": (
            "Flywheel imbalance or mechanical looseness increases vibration at the "
            "shaft rotational frequency and its harmonics, dominated by lower-"
            "frequency energy rather than sharp high-frequency impacts."
        ),
        "acoustic_signature": {
            "time_domain": "Low energy in WPT band 4 and 5; elevated energy in WPT bands 16, 17, 49, 48.",
            "frequency_domain": "Low-frequency dominated energy distribution.",
        },
        "grounding_features": {
            "wpt_energy_4": "low",
            "wpt_energy_16": "high",
            "wpt_energy_5": "low",
            "wpt_energy_17": "high",
            "wpt_energy_49": "high",
            "wpt_energy_48": "high"
        },
        "grounding_features_human": [
            "energy near 375-500 Hz (wpt_energy_4) is LOW",
            "energy near 1875-2000 Hz (wpt_energy_16) is HIGH",
            "energy near 500-625 Hz (wpt_energy_5) is LOW",
            "energy near 2000-2125 Hz (wpt_energy_17) is HIGH",
            "energy near 6000-6125 Hz (wpt_energy_49) is HIGH",
            "energy near 5875-6000 Hz (wpt_energy_48) is HIGH"
        ],
        "distinguishing_from": {
            "Bearing": "MAIN confusable pair. Flywheel = low-frequency rotational "
                       "energy, lower kurtosis; Bearing = high-frequency impulsive.",
        },
        "threshold_source": "data_selected_percentiles_p75_p25",
        "references": ["Verma et al. (2016), IEEE Trans. Reliability 65(1):291-309."],
    },

    "Piston": {
        "name": "Piston / piston-ring fault",
        "mechanism": (
            "Worn piston rings or excessive piston-cylinder clearance cause piston "
            "slap (lateral contact once or twice per revolution) and gas blow-by, "
            "altering mid-frequency band energy and broadband content."
        ),
        "acoustic_signature": {
            "time_domain": "Low zero_crossing_rate indicating lower transient oscillations.",
            "frequency_domain": "Low energy in WPT bands 25, 9, 8, 24, 56 compared to the global distribution.",
        },
        "grounding_features": {
            "zero_crossing_rate": "low",
            "wpt_energy_25": "low",
            "wpt_energy_9": "low",
            "wpt_energy_8": "low",
            "wpt_energy_24": "low",
            "wpt_energy_56": "low"
        },
        "grounding_features_human": [
            "rate of sign changes (proxy for high-frequency content) (zero_crossing_rate) is LOW",
            "energy near 3000-3125 Hz (wpt_energy_25) is LOW",
            "energy near 1000-1125 Hz (wpt_energy_9) is LOW",
            "energy near 875-1000 Hz (wpt_energy_8) is LOW",
            "energy near 2875-3000 Hz (wpt_energy_24) is LOW",
            "energy near 6875-7000 Hz (wpt_energy_56) is LOW"
        ],
        "distinguishing_from": {
            "Riderbelt": "Piston = periodic impact (higher kurtosis); Riderbelt = "
                         "friction-driven broadband noise (higher spectral flatness, "
                         "lower impulsiveness).",
        },
        "threshold_source": "data_selected_percentiles_p75_p25",
        "references": ["Verma et al. (2016), IEEE Trans. Reliability 65(1):291-309."],
    },

    "Riderbelt": {
        "name": "Rider belt / rider ring fault",
        "mechanism": (
            "Wear or damage of the rider ring/belt (which supports the piston and "
            "prevents metal-to-metal contact) increases friction and produces a "
            "more continuous broadband noise rather than discrete impacts."
        ),
        "acoustic_signature": {
            "time_domain": "Lower energy in WPT band 3 and band_energy_ratio_1; elevated band_energy_ratio_2 and spectral_centroid.",
            "frequency_domain": "Broadband noise with elevated spectral flatness and centroid.",
        },
        "grounding_features": {
            "band_energy_ratio_2": "high",
            "band_energy_ratio_1": "low",
            "spectral_centroid": "high",
            "wpt_energy_3": "low",
            "wpt_energy_20": "high",
            "wpt_energy_52": "high"
        },
        "grounding_features_human": [
            "frequency band 1000-2000 Hz energy ratio (band_energy_ratio_2) is HIGH",
            "frequency band 0-1000 Hz energy ratio (band_energy_ratio_1) is LOW",
            "brightness - center of spectral mass (Hz) (spectral_centroid) is HIGH",
            "energy near 250-375 Hz (wpt_energy_3) is LOW",
            "energy near 2375-2500 Hz (wpt_energy_20) is HIGH",
            "energy near 6375-6500 Hz (wpt_energy_52) is HIGH"
        ],
        "distinguishing_from": {
            "Piston": "Riderbelt = broadband friction noise (high flatness/entropy, "
                      "low kurtosis); Piston = periodic impacts (higher kurtosis).",
        },
        "threshold_source": "data_selected_percentiles_p75_p25",
        "references": ["Verma et al. (2016), IEEE Trans. Reliability 65(1):291-309."],
    },
}

# Shared class name constants (single source of truth for all modules)
FAULT_CLASSES = sorted([k for k in KNOWLEDGE_BASE.keys()])
ALL_CLASSES = sorted(list(KNOWLEDGE_BASE.keys()) + ["Healthy"])

# Shared direction normalizer for calibrate_kb.py and verifier.py
def normalize_direction(direction):
    """Normalize a direction string to 'high', 'low', or 'unspecified'."""
    d = direction.lower()
    if any(w in d for w in ["elevated", "high", "higher", "shifted", "widened", "moderately"]):
        return "high"
    elif any(w in d for w in ["low", "lower"]):
        return "low"
    else:
        return "unspecified"

# Verification assertion
assert len(KNOWLEDGE_BASE) == 7, f"Expected 7 classes, got {len(KNOWLEDGE_BASE)}"
print("KNOWLEDGE_BASE loaded successfully. Keys:", list(KNOWLEDGE_BASE.keys()))
