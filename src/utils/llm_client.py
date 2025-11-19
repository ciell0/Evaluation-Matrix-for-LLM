import os
import time
import logging
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

LLM_API_KEY = os.getenv("LLM_API_KEY")  # set in environment
LLM_API_URL = os.getenv("LLM_API_URL", "https://api.example.com/v1/generate")  # replace with provider URL

HEADERS = {
    "Authorization": f"Bearer {LLM_API_KEY}",
    "Content-Type": "application/json",
}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def call_llm(prompt: str, cfg: dict) -> dict:
    payload = {
        "model": cfg.get("model", "gemini-2.5-pro"),
        "prompt": prompt,
        "max_tokens": cfg.get("max_tokens", 512),
        "temperature": cfg.get("temperature", 0.0),
    }
    timeout = httpx.Timeout(cfg.get("timeout", 30.0))
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(LLM_API_URL, headers=HEADERS, json=payload)
        resp.raise_for_status()
        return resp.json()
