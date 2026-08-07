import numpy as np
import scipy.stats
import librosa
import pywt
import threading
from collections import OrderedDict

_feature_count_printed = threading.Event()

def extract_features(signal, sr):
    features = OrderedDict()
    
    rms = np.sqrt(np.mean(signal**2))
    features['rms'] = float(rms)
    features['kurtosis'] = float(scipy.stats.kurtosis(signal))
    features['skewness'] = float(scipy.stats.skew(signal))
    features['crest_factor'] = float(np.max(np.abs(signal)) / rms if rms > 0 else 0.0)
    features['zero_crossing_rate'] = float(np.mean(librosa.zero_crossings(signal)))
    features['peak_to_peak'] = float(np.max(signal) - np.min(signal))
    
    hist, _ = np.histogram(signal, bins=100, density=True)
    probs = hist / np.sum(hist)
    probs = probs[probs > 0]
    features['shannon_entropy'] = float(-np.sum(probs * np.log2(probs)))
    
    n = len(signal)
    fft_vals = np.fft.rfft(signal)
    fft_freqs = np.fft.rfftfreq(n, d=1/sr)
    fft_magnitude = np.abs(fft_vals)
    
    sum_mag = np.sum(fft_magnitude)
    centroid = np.sum(fft_freqs * fft_magnitude) / sum_mag if sum_mag > 0 else 0.0
    features['spectral_centroid'] = float(centroid)
    
    bandwidth = np.sqrt(np.sum(((fft_freqs - centroid)**2) * fft_magnitude) / sum_mag) if sum_mag > 0 else 0.0
    features['spectral_bandwidth'] = float(bandwidth)
    
    if sum_mag > 0:
        cumulative_mag = np.cumsum(fft_magnitude)
        rolloff_idx = np.where(cumulative_mag >= 0.85 * cumulative_mag[-1])[0][0]
        spectral_rolloff_85 = fft_freqs[rolloff_idx]
    else:
        spectral_rolloff_85 = 0.0
    features['spectral_rolloff_85'] = float(spectral_rolloff_85)
    
    features['dominant_frequency'] = float(fft_freqs[np.argmax(fft_magnitude)])
    
    power_spec = fft_magnitude**2
    gmean = np.exp(np.mean(np.log(power_spec + 1e-12)))
    amean = np.mean(power_spec)
    features['spectral_flatness'] = float(gmean / amean if amean > 0 else 0.0)
    
    total_energy = np.sum(power_spec)
    for b in range(1, 9):
        f_min = (b - 1) * 1000
        f_max = b * 1000
        idx = np.where((fft_freqs >= f_min) & (fft_freqs < f_max if b < 8 else fft_freqs <= f_max))[0]
        features[f'band_energy_ratio_{b}'] = float(np.sum(power_spec[idx]) / total_energy if total_energy > 0 else 0.0)
        
    wp = pywt.WaveletPacket(data=signal, wavelet='db4', mode='symmetric', maxlevel=6)
    level6_nodes = wp.get_level(6, order='freq')
    for i, node in enumerate(level6_nodes):
        features[f'wpt_energy_{i+1}'] = float(np.log(np.sum(node.data**2) + 1e-12))
        
    if not _feature_count_printed.is_set():
        print(f"Exact feature count: {len(features)}")
        _feature_count_printed.set()
        
    return features
