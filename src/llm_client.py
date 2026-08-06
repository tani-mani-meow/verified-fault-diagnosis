# src/llm_client.py
import os
import time
import random
import requests
import threading
from pathlib import Path
from dotenv import load_dotenv

# Load env file from the repository root relative to this file
script_dir = Path(__file__).resolve().parent
repo_dir = script_dir.parent
load_dotenv(dotenv_path=repo_dir / ".env", override=True)

# Read config from environment
API_KEY = os.environ.get("NVIDIA_API_KEY")
MODEL = os.environ.get("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")
BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

# If NVIDIA_API_KEY is missing, set a dummy key to prevent crash at start (smoke test will catch it)
if not API_KEY or API_KEY == "your_nvidia_key_here":
    API_KEY = "dummy_key"

# Thread-local storage to record the token usage of the last request in each thread
LAST_USAGE = threading.local()

def get_last_usage() -> dict:
    """
    Returns the token usage of the last request in the current thread.
    """
    if not hasattr(LAST_USAGE, "prompt_tokens"):
        LAST_USAGE.prompt_tokens = 0
    if not hasattr(LAST_USAGE, "completion_tokens"):
        LAST_USAGE.completion_tokens = 0
    return {
        "prompt_tokens": LAST_USAGE.prompt_tokens,
        "completion_tokens": LAST_USAGE.completion_tokens
    }

def generate_with_usage(prompt: str, system: str | None = None, max_retries: int = 10) -> tuple[str, int, int]:
    """
    Generates content using the NVIDIA NIM API and returns the generated text, prompt tokens, and completion tokens.
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
            r = requests.post(url, json=data, headers=headers, timeout=60)
            if r.status_code == 200:
                res_json = r.json()
                content = res_json["choices"][0]["message"]["content"]
                usage = res_json.get("usage", {})
                p_tokens = usage.get("prompt_tokens", 0)
                c_tokens = usage.get("completion_tokens", 0)
                
                global LAST_USAGE
                LAST_USAGE.prompt_tokens = p_tokens
                LAST_USAGE.completion_tokens = c_tokens
                return content, p_tokens, c_tokens
            else:
                r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code in [429, 503, 504]:
                if attempt == max_retries - 1:
                    raise
                time.sleep(delay + random.uniform(0, 1))
                delay *= 2
            else:
                raise  # Non-retryable HTTP error (4xx, etc.)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if attempt == max_retries - 1:
                raise
            time.sleep(delay + random.uniform(0, 1))
            delay *= 2
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay + random.uniform(0, 1))
            delay *= 2
            
    raise RuntimeError("unreachable")

def generate(prompt: str, system: str | None = None, max_retries: int = 10) -> str:
    """
    Generates content using the NVIDIA NIM API with exponential backoff on transient errors.
    """
    content, _, _ = generate_with_usage(prompt, system, max_retries)
    return content

def politeness_delay(seconds: float = 1.0):
    """
    A helper to sleep between successive requests to respect rate limits.
    """
    time.sleep(seconds)
