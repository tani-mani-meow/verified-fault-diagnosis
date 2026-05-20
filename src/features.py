import numpy as np
import scipy.stats
import librosa
import pywt
from collections import OrderedDict

# Global flag to ensure feature count is printed only once
_feature_count_printed = False

def extract_features(signal, sr):
    global _feature_count_printed
    
    features = OrderedDict()
    
    # ------------------ TIME DOMAIN ------------------
    # 1. RMS
    rms = np.sqrt(np.mean(signal**2))
    features['rms'] = float(rms)
    
    # 2. Kurtosis
    features['kurtosis'] = float(scipy.stats.kurtosis(signal))
    
    # 3. Skewness
    features['skewness'] = float(scipy.stats.skew(signal))
    
    # 4. Crest Factor
    features['crest_factor'] = float(np.max(np.abs(signal)) / rms if rms > 0 else 0.0)
    
    # 5. Zero Crossing Rate
    features['zero_crossing_rate'] = float(np.mean(librosa.zero_crossings(signal)))
    
    # 6. Peak to Peak
    features['peak_to_peak'] = float(np.max(signal) - np.min(signal))
    
    # 7. Shannon Entropy (Time Domain)
    hist, _ = np.histogram(signal, bins=100, density=True)
    probs = hist / np.sum(hist)
    probs = probs[probs > 0]
    shannon_entropy = -np.sum(probs * np.log2(probs))
    features['shannon_entropy'] = float(shannon_entropy)
    
    # ------------------ FREQUENCY DOMAIN ------------------
    n = len(signal)
    fft_vals = np.fft.rfft(signal)
    fft_freqs = np.fft.rfftfreq(n, d=1/sr)
    fft_magnitude = np.abs(fft_vals)
    
    # 8. Spectral Centroid
    sum_mag = np.sum(fft_magnitude)
    if sum_mag > 0:
        centroid = np.sum(fft_freqs * fft_magnitude) / sum_mag
    else:
        centroid = 0.0
    features['spectral_centroid'] = float(centroid)
    
    # 9. Spectral Bandwidth
    if sum_mag > 0:
        bandwidth = np.sqrt(np.sum(((fft_freqs - centroid)**2) * fft_magnitude) / sum_mag)
    else:
        bandwidth = 0.0
    features['spectral_bandwidth'] = float(bandwidth)
    
    # 10. Spectral Rolloff 85%
    if sum_mag > 0:
        cumulative_mag = np.cumsum(fft_magnitude)
        threshold = 0.85 * cumulative_mag[-1]
        rolloff_idx = np.where(cumulative_mag >= threshold)[0][0]
        spectral_rolloff_85 = fft_freqs[rolloff_idx]
    else:
        spectral_rolloff_85 = 0.0
    features['spectral_rolloff_85'] = float(spectral_rolloff_85)
    
    # 11. Dominant Frequency
    dominant_frequency = fft_freqs[np.argmax(fft_magnitude)]
    features['dominant_frequency'] = float(dominant_frequency)
    
    # 12. Spectral Flatness
    power_spec = fft_magnitude**2
    eps = 1e-12
    gmean = np.exp(np.mean(np.log(power_spec + eps)))
    amean = np.mean(power_spec)
    spectral_flatness = gmean / amean if amean > 0 else 0.0
    features['spectral_flatness'] = float(spectral_flatness)
    
    # 13-20. Band Energy Ratios for 8 equal bands 0-8 kHz
    total_energy = np.sum(power_spec)
    for b in range(1, 9):
        f_min = (b - 1) * 1000
        f_max = b * 1000
        if b < 8:
            idx = np.where((fft_freqs >= f_min) & (fft_freqs < f_max))[0]
        else:
            idx = np.where((fft_freqs >= f_min) & (fft_freqs <= f_max))[0]
            
        band_energy = np.sum(power_spec[idx])
        ratio = band_energy / total_energy if total_energy > 0 else 0.0
        features[f'band_energy_ratio_{b}'] = float(ratio)
        
    # ------------------ TIME-FREQUENCY DOMAIN ------------------
    # Wavelet Packet Decomposition, 6 levels, wavelet 'db4'
    wp = pywt.WaveletPacket(data=signal, wavelet='db4', mode='symmetric', maxlevel=6)
    level6_nodes = wp.get_level(6, order='freq')
    for i, node in enumerate(level6_nodes):
        coeff = node.data
        energy = np.sum(coeff**2)
        log_energy = np.log(energy + 1e-12)
        features[f'wpt_energy_{i+1}'] = float(log_energy)
        
    if not _feature_count_printed:
        print(f"Exact feature count: {len(features)}")
        _feature_count_printed = True
        
    return features
