# src/llm_client.py
import os
import time
import random
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Check for API key presence
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key or api_key == "your_key_here":
    # Set a dummy key if missing to prevent startup crash; smoke test checks this
    api_key = "dummy_key"

_client = genai.Client(api_key=api_key)
_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

def generate(prompt: str, system: str | None = None, max_retries: int = 5) -> str:
    """
    Generates content using the Google GenAI SDK with exponential backoff on transient errors.
    """
    delay = 2.0
    
    from google.genai import types
    config = types.GenerateContentConfig(
        temperature=0.0,
        system_instruction=system
    )
    
    for attempt in range(max_retries):
        try:
            resp = _client.models.generate_content(
                model=_MODEL,
                contents=prompt,
                config=config
            )
            return resp.text
        except Exception as e:
            msg = str(e).lower()
            # transient errors: rate limit (429), quota exhaustion, service unavailable (503)
            transient = ("429" in msg or "rate" in msg or "quota" in msg
                         or "resource_exhausted" in msg or "503" in msg
                         or "unavailable" in msg)
            if attempt == max_retries - 1 or not transient:
                raise
            time.sleep(delay + random.uniform(0, 1))
            delay *= 2
            
    raise RuntimeError("unreachable")

def politeness_delay(seconds: float = 1.0):
    """
    A helper to sleep between successive requests to respect free-tier per-minute limits.
    """
    time.sleep(seconds)
