"""Guard: no external / hosted AI SDKs or endpoints anywhere in the code or dependencies."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE_DIRS = [ROOT / "ml", ROOT / "backend" / "app", ROOT / "frontend" / "src"]
SKIP_PARTS = {"node_modules", ".venv", "__pycache__", ".next", "kaggle"}

FORBIDDEN_IMPORTS = [
    r"^\s*(import|from)\s+openai\b",
    r"^\s*(import|from)\s+anthropic\b",
    r"^\s*(import|from)\s+google\.generativeai\b",
    r"^\s*(import|from)\s+google\.genai\b",
    r"^\s*(import|from)\s+groq\b",
    r"^\s*(import|from)\s+cohere\b",
    r"^\s*(import|from)\s+mistralai\b",
    r"^\s*(import|from)\s+langchain",
    r"^\s*(import|from)\s+llama_index\b",
    r"^\s*(import|from)\s+replicate\b",
    r"^\s*(import|from)\s+together\b",
    r"InferenceClient\s*\(",
    r"from\s+['\"](openai|@anthropic-ai/sdk|@google/generative-ai|groq-sdk|ai)['\"]",
    r"require\(['\"](openai|@anthropic-ai/sdk|@google/generative-ai)['\"]\)",
]
FORBIDDEN_DOMAINS = [
    "api.openai.com",
    "generativelanguage.googleapis.com",
    "aiplatform.googleapis.com",
    "api.anthropic.com",
    "api.groq.com",
    "api-inference.huggingface.co",
    "router.huggingface.co",
    "api.cohere.ai",
    "api.mistral.ai",
    "api.together.xyz",
    "api.replicate.com",
    "copilot.microsoft.com",
    "openai.azure.com",
]
FORBIDDEN_PACKAGES = [
    "openai",
    "anthropic",
    "google-generativeai",
    "google-genai",
    "groq",
    "cohere",
    "mistralai",
    "langchain",
    "llama-index",
    "replicate",
    "together",
    "@anthropic-ai/sdk",
    "@google/generative-ai",
    "groq-sdk",
    "ai",
]


def _files() -> list[Path]:
    out = []
    for d in CODE_DIRS:
        for p in d.rglob("*"):
            if p.suffix in (".py", ".ts", ".tsx", ".js", ".mjs") and not (
                SKIP_PARTS & set(p.parts)
            ):
                out.append(p)
    return out


def test_no_external_ai_imports() -> None:
    bad = []
    for p in _files():
        text = p.read_text(errors="ignore")
        for pat in FORBIDDEN_IMPORTS:
            if re.search(pat, text, flags=re.M):
                bad.append(f"{p.relative_to(ROOT)}: {pat}")
    assert not bad, "External AI SDK usage found:\n" + "\n".join(bad)


def test_no_external_ai_domains() -> None:
    bad = []
    for p in _files():
        if p.name == Path(__file__).name:
            continue
        text = p.read_text(errors="ignore").lower()
        bad += [f"{p.relative_to(ROOT)}: {d}" for d in FORBIDDEN_DOMAINS if d in text]
    assert not bad, "External AI endpoint found:\n" + "\n".join(bad)


def test_no_external_ai_dependencies() -> None:
    reqs = (ROOT / "requirements.txt").read_text().lower().splitlines()
    names = {
        re.split(r"[<>=\[ ]", r.strip())[0] for r in reqs if r.strip() and not r.startswith("#")
    }
    assert not names & set(FORBIDDEN_PACKAGES), names & set(FORBIDDEN_PACKAGES)
    pkg = json.loads((ROOT / "frontend" / "package.json").read_text())
    deps = set(pkg.get("dependencies", {})) | set(pkg.get("devDependencies", {}))
    assert not deps & set(FORBIDDEN_PACKAGES), deps & set(FORBIDDEN_PACKAGES)


def test_local_llm_is_off_and_localhost_only() -> None:
    from app.config import get_settings
    from app.copilot import local_llm

    assert get_settings().copilot_local_llm is False
    assert local_llm.enabled() is False
    s = get_settings()
    old_flag, old_url = s.copilot_local_llm, s.ollama_url
    try:
        s.copilot_local_llm, s.ollama_url = True, "https://some-cloud-llm.example.com"
        assert local_llm.enabled() is False  # remote URLs are refused even when switched on
    finally:
        s.copilot_local_llm, s.ollama_url = old_flag, old_url
