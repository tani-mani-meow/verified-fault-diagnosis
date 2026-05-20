import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

def main():
    # Resolve paths relative to this script
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent
    data_dir = repo_dir / "data" / "AirCompressorDataset"
    outputs_dir = repo_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    
    if not data_dir.exists():
        print(f"Error: Dataset directory {data_dir} does not exist. Run download_data.py first.")
        sys.exit(1)
        
    # Find all class folders (subdirectories under data_dir)
    class_folders = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    report_lines = []
    
    # 1. List of class folders found + file count per class
    report_lines.append("=== Class Folders and File Counts ===")
    total_files = 0
    class_files = {}
    for folder in class_folders:
        # MathWorks AirCompressorDataset might contain wav files
        files = list(folder.glob("*.wav"))
        class_files[folder.name] = files
        count = len(files)
        total_files += count
        report_lines.append(f"Class: {folder.name} | Files: {count}")
    report_lines.append(f"Total File Count: {total_files}\n")
    
    # 2. For one sample per class: sample rate, number of samples, duration in seconds
    report_lines.append("=== Sample Details (One per Class) ===")
    sample_rates = []
    durations = []
    
    representative_samples = {}
    
    for folder in class_folders:
        files = class_files[folder.name]
        if not files:
            report_lines.append(f"Class {folder.name}: No WAV files found.")
            continue
        first_file = files[0]
        representative_samples[folder.name] = first_file
        
        # Get metadata using soundfile info (fast)
        info = sf.info(str(first_file))
        report_lines.append(
            f"Class: {folder.name} | File: {first_file.name} | "
            f"SR: {info.samplerate} Hz | Samples: {info.frames} | "
            f"Duration: {info.duration:.4f} s"
        )
        
    report_lines.append("")
    
    # 3. Confirm/deny: sample_rate == 16000, and report the actual median duration of ALL files
    all_durations = []
    all_sample_rates = []
    for folder in class_folders:
        for f in class_files[folder.name]:
            info = sf.info(str(f))
            all_durations.append(info.duration)
            all_sample_rates.append(info.samplerate)
            
    median_dur = np.median(all_durations) if all_durations else 0.0
    unique_srs = set(all_sample_rates)
    
    report_lines.append("=== Dataset Verification Metrics ===")
    # Confirm/deny if sample_rate is 16000
    if len(unique_srs) == 1 and 16000 in unique_srs:
        report_lines.append("Confirm: All files have sample_rate == 16000 Hz.")
    elif 16000 in unique_srs:
        report_lines.append(f"Partial Confirm: Some files have sample_rate == 16000 Hz. All sample rates found: {unique_srs}")
    else:
        report_lines.append(f"Deny: No files have sample_rate == 16000 Hz. All sample rates found: {unique_srs}")
        
    report_lines.append(f"Actual Median Duration: {median_dur:.6f} s")
    
    # Write the report
    report_content = "\n".join(report_lines)
    print(report_content)
    
    report_file_path = outputs_dir / "dataset_report.txt"
    with open(report_file_path, "w") as rf:
        rf.write(report_content)
    print(f"Saved dataset report to {report_file_path}")
    
    # 4. Generate healthy_vs_liv.png (2x2 grid)
    # Healthy waveform, LIV waveform, Healthy spectrogram, LIV spectrogram
    healthy_file = representative_samples.get("Healthy")
    liv_file = representative_samples.get("LIV")
    
    if not healthy_file or not liv_file:
        print("Error: Could not find representative samples for Healthy and/or LIV classes. Check class names.")
        return
        
    print("Loading Healthy and LIV samples for visualization...")
    y_healthy, sr_healthy = librosa.load(str(healthy_file), sr=None)
    y_liv, sr_liv = librosa.load(str(liv_file), sr=None)
    
    # Compute spectrograms using librosa
    stft_healthy = librosa.stft(y_healthy)
    spec_healthy = librosa.amplitude_to_db(np.abs(stft_healthy), ref=np.max)
    
    stft_liv = librosa.stft(y_liv)
    spec_liv = librosa.amplitude_to_db(np.abs(stft_liv), ref=np.max)
    
    # Setup 2x2 plot
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Time axis for waveforms
    time_healthy = np.arange(len(y_healthy)) / sr_healthy
    time_liv = np.arange(len(y_liv)) / sr_liv
    
    # Waveforms
    axs[0, 0].plot(time_healthy, y_healthy, color='#2ca02c')
    axs[0, 0].set_title("Healthy Waveform")
    axs[0, 0].set_xlabel("Time (s)")
    axs[0, 0].set_ylabel("Amplitude")
    axs[0, 0].grid(True, alpha=0.3)
    
    axs[0, 1].plot(time_liv, y_liv, color='#d62728')
    axs[0, 1].set_title("LIV Waveform")
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Amplitude")
    axs[0, 1].grid(True, alpha=0.3)
    
    # Spectrograms
    img_healthy = librosa.display.specshow(spec_healthy, sr=sr_healthy, x_axis='time', y_axis='linear', ax=axs[1, 0], cmap='viridis')
    axs[1, 0].set_title("Healthy Spectrogram")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Frequency (Hz)")
    fig.colorbar(img_healthy, ax=axs[1, 0], format="%+2.0f dB")
    
    img_liv = librosa.display.specshow(spec_liv, sr=sr_liv, x_axis='time', y_axis='linear', ax=axs[1, 1], cmap='magma')
    axs[1, 1].set_title("LIV Spectrogram")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Frequency (Hz)")
    fig.colorbar(img_liv, ax=axs[1, 1], format="%+2.0f dB")
    
    plt.tight_layout()
    plot_file_path = outputs_dir / "healthy_vs_liv.png"
    plt.savefig(plot_file_path, dpi=150)
    plt.close()
    print(f"Saved visualization comparison to {plot_file_path}")

if __name__ == "__main__":
    main()
