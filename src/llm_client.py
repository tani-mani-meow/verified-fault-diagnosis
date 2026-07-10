# src/llm_client.py
import os
import time
import random
import requests
from dotenv import load_dotenv

load_dotenv()

# Read config from environment
API_KEY = os.environ.get("NVIDIA_API_KEY")
MODEL = os.environ.get("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")
BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

# If NVIDIA_API_KEY is missing, set a dummy key to prevent crash at start (smoke test will catch it)
if not API_KEY or API_KEY == "your_nvidia_key_here":
    API_KEY = "dummy_key"

def generate(prompt: str, system: str | None = None, max_retries: int = 5) -> str:
    """
    Generates content using the NVIDIA NIM API with exponential backoff on transient errors.
    """
    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    data = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.0
    }
    
    delay = 2.0
    for attempt in range(max_retries):
        try:
            r = requests.post(url, json=data, headers=headers)
            if r.status_code == 200:
                res_json = r.json()
                return res_json["choices"][0]["message"]["content"]
            elif r.status_code in [429, 503, 504]:
                # transient rate limit or service unavailable
                time.sleep(delay + random.uniform(0, 1))
                delay *= 2
            else:
                r.raise_for_status()
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay + random.uniform(0, 1))
            delay *= 2
            
    raise RuntimeError("unreachable")

def politeness_delay(seconds: float = 1.0):
    """
    A helper to sleep between successive requests to respect rate limits.
    """
    time.sleep(seconds)
