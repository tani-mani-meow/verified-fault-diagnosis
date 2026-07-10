# scripts/test_llm.py
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure we can import from src
script_dir = Path(__file__).resolve().parent
repo_dir = script_dir.parent
sys.path.insert(0, str(repo_dir / "src"))

from llm_client import generate

def main():
    # Force reloading env from .env in repository root
    env_path = repo_dir / ".env"
    load_dotenv(dotenv_path=env_path, override=True)
    
    api_key = os.environ.get("NVIDIA_API_KEY")
    model_name = os.environ.get("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")
    
    print("--- Nvidia NIM API Smoke Test ---")
    print(f"Target Model: {model_name}")
    
    if not api_key or api_key == "your_nvidia_key_here" or api_key == "dummy_key":
        print("\n[NOTICE]: NVIDIA_API_KEY is not set or uses the placeholder 'your_nvidia_key_here' in .env.")
        print("Please paste your real API key in compressor-pdm/.env to enable LLM generation.")
        sys.exit(0)
        
    try:
        print("Sending smoke test request...")
        response = generate("Reply with exactly the word: OK")
        print(f"Raw Response: {repr(response.strip())}")
        if "OK" in response:
            print("Status: SUCCESS (API connection verified)")
        else:
            print("Status: WARNING (Response received but did not match expected 'OK')")
    except Exception as e:
        print(f"Status: FAILED")
        print(f"Error Details: {e}")
        
if __name__ == "__main__":
    main()
