import os
import urllib.request
import zipfile
from pathlib import Path

def main():
    # Resolve paths relative to this script
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent
    data_dir = repo_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    zip_url = "https://www.mathworks.com/supportfiles/audio/AirCompressorDataset/AirCompressorDataset.zip"
    zip_path = data_dir / "AirCompressorDataset.zip"
    extract_dir = data_dir / "AirCompressorDataset"
    
    # 1. Download zip if not already present
    if not zip_path.exists():
        print(f"Downloading from {zip_url} to {zip_path}...")
        try:
            urllib.request.urlretrieve(zip_url, zip_path)
            print("Download completed.")
        except Exception as e:
            print(f"Error downloading from URL {zip_url}: {e}")
            raise e
    else:
        print(f"Zip file already exists at {zip_path}, skipping download.")
        
    # 2. Unzip if extraction directory does not exist or is empty
    if not extract_dir.exists() or not any(extract_dir.iterdir()):
        print(f"Extracting {zip_path} to {extract_dir}...")
        extract_dir.mkdir(exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # MathWorks ZIP might contain a subfolder. Let's see what the ZIP contains.
                # If the ZIP's top-level folder is 'AirCompressorDataset', we can just extract to data_dir.
                # Let's check the contents first.
                namelist = zip_ref.namelist()
                top_level_dirs = {Path(name).parts[0] for name in namelist if len(Path(name).parts) > 0}
                print(f"Zip content top-level elements: {top_level_dirs}")
                
                # If there's already a single folder named 'AirCompressorDataset' or similar, 
                # extracting to data_dir will create data_dir/AirCompressorDataset/...
                # Let's check if the top level matches 'AirCompressorDataset'
                if len(top_level_dirs) == 1 and list(top_level_dirs)[0] == 'AirCompressorDataset':
                    print("Extracting directly to data/ directory as it has a single top-level AirCompressorDataset directory...")
                    zip_ref.extractall(data_dir)
                else:
                    print("Extracting to data/AirCompressorDataset/ directory...")
                    zip_ref.extractall(extract_dir)
                    
            print("Extraction completed.")
        except Exception as e:
            print(f"Error extracting zip: {e}")
            raise e
    else:
        print(f"Extraction directory already exists and is not empty at {extract_dir}, skipping unzip.")
        
    # Print the resolved absolute path
    resolved_path = extract_dir.resolve()
    print(f"Resolved absolute path of dataset: {resolved_path}")

if __name__ == "__main__":
    main()
