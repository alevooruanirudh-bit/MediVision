"""Mine the Copilot report phrase bank from Indiana University chest X-ray reports (OpenI).

    python -m ml.copilot.mine_phrases

Sentences are split, normalised ("XXXX" de-identification tokens dropped), and counted. We keep the most
frequent *normal* statements per system and *negative* statements per finding, plus example positive
sentences. The output (backend/app/copilot/phrase_bank.json) stores short generic sentence patterns with
their corpus frequency and the dataset citation.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

import pandas as pd

from ml.common import PROCESSED, ROOT
from ml.train_report import FINDING_KEYWORDS

OUT = ROOT / "backend" / "app" / "copilot" / "phrase_bank.json"

SYSTEM_PATTERNS = {
    "heart": r"\b(heart|cardiac|cardiomediastinal|mediastinal silhouette)\b",
    "lungs": r"\b(lungs?|pulmonary|airspace|parenchym)\w*",
    "pleura": r"\b(pleural|pneumothorax|effusion)\b",
    "bones": r"\b(osseous|bony|bones?|skeletal|spine|ribs?)\b",
    "mediastinum": r"\b(mediastin|hilar|trachea|aort)\w*",
}
NORMAL_CUES = (
    r"\b(normal|clear|unremarkable|no |without|within normal limits|intact|stable|negative)\b"
)
ABNORMAL_CUES = r"\b(enlarged|opacity|opacities|effusion is|there is (a|an|mild|moderate|small))\b"


def sentences(text: str) -> list[str]:
    text = re.sub(r"x{2,}", " ", str(text).lower())
    out = []
    for s in re.split(r"(?<=[.;])\s+", text):
        s = " ".join(s.strip(" .;,").split())
        if 3 <= len(s.split()) <= 16:
            out.append(s)
    return out


def mine(min_count: int = 3, top: int = 8) -> dict:
    df = pd.read_csv(PROCESSED / "iu_xray" / "reports.csv").fillna("")
    df = df[df["split"] == "train"]
    normal: dict[str, Counter] = defaultdict(Counter)
    negative: dict[str, Counter] = defaultdict(Counter)
    positive: dict[str, Counter] = defaultdict(Counter)
    impression_normal: Counter = Counter()
    n_sent = 0
    for _, r in df.iterrows():
        for s in sentences(f"{r['findings']} {r['impression']}"):
            n_sent += 1
            is_neg = bool(re.match(r"^(no|there is no|without|negative for)\b", s))
            if re.search(r"no acute (cardiopulmonary|disease|abnormality|process)", s):
                impression_normal[s] += 1
            for lab, kws in FINDING_KEYWORDS.items():
                if any(k in s for k in kws):
                    (negative if is_neg else positive)[lab][s] += 1
            if re.search(NORMAL_CUES, s) and not re.search(ABNORMAL_CUES, s):
                for sysname, pat in SYSTEM_PATTERNS.items():
                    if re.search(pat, s):
                        normal[sysname][s] += 1
                        break

    def keep(c: Counter, k: int = top) -> list[dict]:
        return [{"text": t, "count": n} for t, n in c.most_common(k) if n >= min_count]

    bank = {
        "_meta": {
            "source": "Indiana University Chest X-ray Collection (OpenI) — Demner-Fushman D. et al., "
            "JAMIA 2016; Kaggle copy masrursabab/iu-chest-x-rays-cleaned",
            "method": "frequency-ranked sentence patterns from the training split; "
            "de-identification tokens removed",
            "n_reports": int(len(df)),
            "n_sentences": n_sent,
        },
        "normal": {k: keep(v) for k, v in normal.items()},
        "negative": {k: keep(v, 5) for k, v in negative.items()},
        "positive_examples": {k: keep(v, 5) for k, v in positive.items()},
        "impression_normal": keep(impression_normal, 6),
    }
    OUT.write_text(json.dumps(bank, indent=2))
    print(f"[phrases] mined {n_sent} sentences from {len(df)} reports -> {OUT}")
    return bank


if __name__ == "__main__":
    mine()
