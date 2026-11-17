"""Rule-based report rewriting: concise / detailed / patient-friendly (lay-term glossary)."""

from __future__ import annotations

import copy
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.copilot.entities import find_terms
from app.copilot.kb import entry

HERE = Path(__file__).parent
PROB_NOTE = re.compile(r"\s*\(AI p = [^)]*\)")
STYLES = ("standard", "concise", "detailed", "patient")


@lru_cache
def glossary() -> dict[str, str]:
    return json.loads((HERE / "glossary.json").read_text())["terms"]


def lay(text: str) -> str:
    out = text
    for term in sorted(glossary(), key=len, reverse=True):
        out = re.sub(rf"\b{re.escape(term)}\b", glossary()[term], out, flags=re.I)
    out = re.sub(
        r"\(AI p = ([0-9.]+)[^)]*\)",
        lambda m: f"(the computer estimated about a {round(float(m.group(1)) * 100)}% chance)",
        out,
    )
    out = out.replace("Findings are consistent with", "The pictures show signs of")
    out = out.replace("Appearances are likely to represent", "This most likely shows")
    out = out.replace("cannot be excluded", "cannot be ruled out yet")
    out = out.replace("Findings may represent", "This might be")
    return out


def _first_sentence(text: str) -> str:
    parts = re.split(r"(?<=\.)\s+", PROB_NOTE.sub("", text).strip())
    return parts[0] if parts else text


def _apply(value: Any, fn: Any) -> Any:
    if isinstance(value, dict):
        return {k: fn(v) for k, v in value.items()}
    if isinstance(value, list):
        return [fn(v) for v in value]
    return fn(value) if isinstance(value, str) else value


def rewrite(sections: dict[str, Any], style: str, only: str | None = None) -> dict[str, Any]:
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}")
    s = copy.deepcopy(sections)
    keys = [only] if only and only != "all" else [k for k in s if k not in ("meta", "disclaimer")]
    for k in keys:
        if k not in s:
            continue
        if style == "concise":
            if k == "findings":
                s[k] = {
                    sk: _first_sentence(v)
                    for sk, v in s[k].items()
                    if not v.startswith("Not assessed")
                }
            elif k in ("recommendations", "ai_confidence") or k == "measurements":
                s[k] = s[k][:2]
            else:
                s[k] = _apply(s[k], lambda t: PROB_NOTE.sub("", t))
        elif style == "detailed":
            if k == "findings":
                for sk, v in s[k].items():
                    extra = []
                    for term in find_terms(v)[:2]:
                        e = entry(term)
                        if e and e.get("appearance"):
                            extra.append(f"[{e['name']}: {e['appearance']}]")
                    s[k][sk] = v + (" " + " ".join(extra) if extra else "")
            elif k == "recommendations":
                more = []
                for r in s[k]:
                    label = r.split(":")[0]
                    e = entry(label)
                    if e:
                        more += [f"{label}: {x}" for x in e.get("next_steps", [])[1:3]]
                s[k] = s[k][:-1] + more + s[k][-1:]
        elif style == "patient":
            if k == "findings":
                s[k] = {sk: lay(v) for sk, v in s[k].items()}
            elif k == "impression":
                items = [lay(re.sub(r"^\d+\.\s*", "", x)) for x in s[k]]
                lays = []
                for x in s[k]:
                    for term in find_terms(x)[:1]:
                        e = entry(term)
                        if e and e.get("lay") and e["lay"] not in lays:
                            lays.append(e["lay"])
                s[k] = [
                    f"In simple terms: {' '.join(items)}",
                    *lays,
                    "Your doctor will talk these results through with you.",
                ]
            elif k == "recommendations":
                s[k] = [lay(x) for x in s[k]]
            elif k == "ai_confidence":
                s[k] = [
                    "These results come from a computer program that is still a research "
                    "prototype. It can be wrong, so a doctor always checks them."
                ]
            else:
                s[k] = _apply(s[k], lay)
    meta = s.setdefault("meta", {})
    meta["style"] = style
    return s
