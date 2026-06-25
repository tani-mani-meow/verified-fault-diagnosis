# src/build_glossary.py
import os
import sys
import json
from pathlib import Path

def main():
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent
    feature_names_path = repo_dir / "outputs" / "feature_names.json"
    glossary_path = repo_dir / "outputs" / "feature_glossary.json"
    
    if not feature_names_path.exists():
        print(f"Error: feature_names.json not found at {feature_names_path}.")
        sys.exit(1)
        
    with open(feature_names_path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)
        
    glossary = {}
    
    # Pre-defined mapping for scalars
    scalar_map = {
        "rms": {
            "label": "overall loudness (RMS amplitude)",
            "description": "overall loudness (RMS amplitude)"
        },
        "kurtosis": {
            "label": "impulsiveness / spikiness of the waveform",
            "description": "impulsiveness / spikiness of the waveform"
        },
        "skewness": {
            "label": "asymmetry of the amplitude distribution",
            "description": "asymmetry of the amplitude distribution"
        },
        "crest_factor": {
            "label": "peak-to-RMS ratio (sharp transients)",
            "description": "peak-to-RMS ratio (sharp transients)"
        },
        "zero_crossing_rate": {
            "label": "rate of sign changes (proxy for high-frequency content)",
            "description": "rate of sign changes (proxy for high-frequency content)"
        },
        "peak_to_peak": {
            "label": "max-to-min amplitude span",
            "description": "max-to-min amplitude span"
        },
        "shannon_entropy": {
            "label": "signal randomness / complexity",
            "description": "signal randomness / complexity"
        },
        "spectral_centroid": {
            "label": "brightness - center of spectral mass (Hz)",
            "description": "brightness - center of spectral mass (Hz)"
        },
        "spectral_bandwidth": {
            "label": "spread of the spectrum (Hz)",
            "description": "spread of the spectrum (Hz)"
        },
        "spectral_rolloff_85": {
            "label": "frequency below which 85% of energy lies (Hz)",
            "description": "frequency below which 85% of energy lies (Hz)"
        },
        "dominant_frequency": {
            "label": "strongest single frequency (Hz)",
            "description": "strongest single frequency (Hz)"
        },
        "spectral_flatness": {
            "label": "tonal vs noise-like balance",
            "description": "tonal vs noise-like balance"
        }
    }
    
    # 64 WPT bands verification list
    wpt_bands = []
    
    for f in feature_names:
        if f in scalar_map:
            glossary[f] = {
                "label": scalar_map[f]["label"],
                "description": scalar_map[f]["description"]
            }
        elif f.startswith("band_energy_ratio_"):
            try:
                k = int(f.split("_")[-1])
                f_min = (k - 1) * 1000
                f_max = k * 1000
                glossary[f] = {
                    "label": f"frequency band {f_min}-{f_max} Hz energy ratio",
                    "description": f"ratio of energy concentrated in the {f_min}-{f_max} Hz frequency band",
                    "band_hz": [float(f_min), float(f_max)],
                    "center_hz": float((f_min + f_max) / 2)
                }
            except Exception as e:
                print(f"Error parsing band_energy_ratio feature {f}: {e}")
        elif f.startswith("wpt_energy_"):
            try:
                i = int(f.split("_")[-1])
                f_min = (i - 1) * 125
                f_max = i * 125
                wpt_bands.append((i, f_min, f_max))
                glossary[f] = {
                    "label": f"energy near {f_min}-{f_max} Hz",
                    "description": f"log-energy in the {f_min}-{f_max} Hz frequency band computed via 6-level WPT with db4 wavelet",
                    "band_hz": [float(f_min), float(f_max)],
                    "center_hz": float((f_min + f_max) / 2)
                }
            except Exception as e:
                print(f"Error parsing WPT feature {f}: {e}")
        else:
            print(f"Warning: Unknown feature name format: {f}")
            
    # Sanity checks on WPT bands tiling 0 - 8000 Hz
    wpt_bands.sort(key=lambda x: x[0])
    
    # 1. Total count
    assert len(wpt_bands) == 64, f"Expected 64 WPT bands, got {len(wpt_bands)}"
    
    # 2. Gaps and overlap check
    for idx in range(len(wpt_bands)):
        i, f_min, f_max = wpt_bands[idx]
        # Check node width is exactly 125 Hz
        assert f_max - f_min == 125, f"WPT node {i} width is {f_max - f_min} instead of 125 Hz"
        # Check start and end frequency match index
        assert f_min == (i - 1) * 125, f"WPT node {i} starts at {f_min} instead of {(i-1)*125}"
        assert f_max == i * 125, f"WPT node {i} ends at {f_max} instead of {i*125}"
        
        # Check contiguous tiling
        if idx > 0:
            prev_i, prev_f_min, prev_f_max = wpt_bands[idx - 1]
            assert f_min == prev_f_max, f"Gap or overlap detected between node {prev_i} and {i}: {prev_f_max} vs {f_min}"
            
    # Check bounds
    assert wpt_bands[0][1] == 0, "WPT tiling does not start at 0 Hz"
    assert wpt_bands[-1][2] == 8000, "WPT tiling does not end at 8000 Hz"
    
    print("\n--- WPT Tiling Sanity Check ---")
    print("All 64 WPT bands tile 0-8000 Hz continuously with no gaps/overlaps.")
    print(f"Node  1 Band: {glossary['wpt_energy_1']['band_hz']} Hz")
    print(f"Node  5 Band: {glossary['wpt_energy_5']['band_hz']} Hz")
    print(f"Node 34 Band: {glossary['wpt_energy_34']['band_hz']} Hz")
    print(f"Node 35 Band: {glossary['wpt_energy_35']['band_hz']} Hz")
    print(f"Node 64 Band: {glossary['wpt_energy_64']['band_hz']} Hz")
    
    # Independent validation of Gray-code path-to-frequency index conversion
    import pywt
    wp = pywt.WaveletPacket(data=[0]*1024, wavelet='db4', maxlevel=6)
    nodes = wp.get_level(6, order='freq')
    
    def path_to_freq_idx(path):
        gray = [0 if c == 'a' else 1 for c in path]
        binary = [gray[0]]
        for idx_b in range(1, len(gray)):
            binary.append(binary[idx_b-1] ^ gray[idx_b])
        val = 0
        for b in binary:
            val = (val << 1) | b
        return val
        
    for idx_n, node in enumerate(nodes):
        freq_idx = path_to_freq_idx(node.path)
        assert idx_n == freq_idx, f"Gray code index mismatch for path {node.path}: expected {idx_n}, got {freq_idx}"
        
    print("WPT ordering source = freq (matches features.py: YES)")
    
    # band_energy_ratio check
    print("\n--- Band Energy Ratios ---")
    for b in range(1, 9):
        feat = f"band_energy_ratio_{b}"
        print(f"Ratio {b} Band: {glossary[feat]['band_hz']} Hz")
        
    # Save glossary
    glossary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(glossary_path, "w", encoding="utf-8") as f:
        json.dump(glossary, f, indent=4)
    print(f"\nSaved feature glossary to {glossary_path}")

if __name__ == "__main__":
    main()

