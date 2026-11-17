"""Knowledge base access + BM25 retrieval (rank_bm25) for definitions and background."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent


@lru_cache
def entries() -> list[dict[str, Any]]:
    return json.loads((HERE / "knowledge_base.json").read_text())["entries"]


@lru_cache
def _index() -> dict[str, dict[str, Any]]:
    idx: dict[str, dict[str, Any]] = {}
    for e in entries():
        for k in [e["id"], e["name"], *e.get("synonyms", [])]:
            idx[k.lower().replace("_", " ")] = e
    return idx


def entry(key: str | None) -> dict[str, Any] | None:
    if not key:
        return None
    return _index().get(key.lower().replace("_", " "))


def _tok(text: str) -> list[str]:
    return re.findall(r"[a-z0-9+]+", text.lower())


@lru_cache
def _bm25():  # type: ignore[no-untyped-def]
    from rank_bm25 import BM25Okapi

    docs = []
    for e in entries():
        text = " ".join(
            [
                e["name"],
                " ".join(e.get("synonyms", [])),
                e.get("definition", ""),
                e.get("appearance", ""),
                " ".join(e.get("causes", [])),
            ]
        )
        docs.append(_tok(text))
    return BM25Okapi(docs)


def search(query: str, k: int = 3) -> list[tuple[dict[str, Any], float]]:
    scores = _bm25().get_scores(_tok(query))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return [(entries()[i], float(scores[i])) for i in order if scores[i] > 0]
