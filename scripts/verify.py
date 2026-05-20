import sys
from pathlib import Path
import soundfile as sf

def main():
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent
    data_dir = repo_dir / "data" / "AirCompressorDataset"
    report_file = repo_dir / "outputs" / "dataset_report.txt"
    plot_file = repo_dir / "outputs" / "healthy_vs_liv.png"
    
    all_passed = True
    
    # 1. data/AirCompressorDataset/ exists
    if data_dir.exists() and data_dir.is_dir():
        print("data/AirCompressorDataset/ exists: PASS")
    else:
        print("data/AirCompressorDataset/ exists: FAIL")
        all_passed = False
        
    # 2. exactly 8 class folders found
    class_folders = []
    if data_dir.exists():
        class_folders = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    if len(class_folders) == 8:
        print(f"exactly 8 class folders found ({', '.join(d.name for d in class_folders)}): PASS")
    else:
        print(f"exactly 8 class folders found (found {len(class_folders)}): FAIL")
        all_passed = False
        
    # 3. every sample loads without error
    loaded_all = True
    total_files_checked = 0
    if len(class_folders) == 8:
        for folder in class_folders:
            files = list(folder.glob("*.wav"))
            if not files:
                loaded_all = False
                print(f"Class folder {folder.name} is empty: FAIL")
            for f in files:
                try:
                    # Quick test to make sure it loads
                    with sf.SoundFile(str(f)) as file_obj:
                        pass
                    total_files_checked += 1
                except Exception as e:
                    print(f"Failed to load file {f.name}: {e}")
                    loaded_all = False
    else:
        loaded_all = False
        
    if loaded_all and total_files_checked > 0:
        print(f"every sample loads without error (checked {total_files_checked} files): PASS")
    else:
        print("every sample loads without error: FAIL")
        all_passed = False
        
    # 4. outputs/dataset_report.txt exists and non-empty
    if report_file.exists() and report_file.stat().st_size > 0:
        print("outputs/dataset_report.txt exists and non-empty: PASS")
    else:
        print("outputs/dataset_report.txt exists and non-empty: FAIL")
        all_passed = False
        
    # 5. outputs/healthy_vs_liv.png exists
    if plot_file.exists():
        print("outputs/healthy_vs_liv.png exists: PASS")
    else:
        print("outputs/healthy_vs_liv.png exists: FAIL")
        all_passed = False
        
    if all_passed:
        print("\nAll verification checks passed.")
        sys.exit(0)
    else:
        print("\nVerification checks failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
