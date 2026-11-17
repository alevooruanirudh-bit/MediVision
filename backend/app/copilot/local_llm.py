"""OPTIONAL plug-in (OFF by default): polish Copilot wording with a LOCAL open-weight model via Ollama.

Only the Copilot may use it, only on localhost, never a cloud API. Facts always come from the grounded
answer composers; the local model may only rephrase. The app is complete without it.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

from app.config import get_settings

log = logging.getLogger("medivision.local_llm")
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "ollama"}  # "ollama" = docker-compose service name


def enabled() -> bool:
    s = get_settings()
    return bool(s.copilot_local_llm) and urlparse(s.ollama_url).hostname in LOCAL_HOSTS


def status() -> dict[str, object]:
    s = get_settings()
    host = urlparse(s.ollama_url).hostname
    return {
        "enabled": enabled(),
        "configured": bool(s.copilot_local_llm),
        "url_is_local": host in LOCAL_HOSTS,
        "model": s.ollama_model if enabled() else None,
    }


def polish(text: str, question: str) -> str:
    if not enabled():
        return text
    import httpx

    s = get_settings()
    prompt = (
        "Rephrase the ANSWER to the QUESTION more naturally for a clinician. Do not add, remove "
        f"or change any facts or numbers.\nQUESTION: {question}\nANSWER: {text}\nREPHRASED:"
    )
    try:
        r = httpx.post(
            f"{s.ollama_url}/api/generate",
            json={"model": s.ollama_model, "prompt": prompt, "stream": False},
            timeout=20,
        )
        out = r.json().get("response", "").strip()
        return out or text
    except Exception as exc:  # pragma: no cover - optional path
        log.warning("local LLM unavailable: %s", exc)
        return text
