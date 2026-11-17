"""Build ml/copilot/intents.jsonl and train the Copilot intent model (TF-IDF + logistic regression).

    python -m ml.copilot.train_intent

Held-out evaluation uses *templates* that were never seen during training (split by template, not by
filled sentence), so the reported accuracy is not inflated by slot-filling near-duplicates.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from ml.common import artifact_dir, write_json
from ml.copilot.intents_source import FINDING_FILLERS, INTENTS
from ml.copilot.text import featurize

HERE = Path(__file__).parent


def expand(template: str, rng: random.Random, n: int = 2) -> list[str]:
    if "{f}" not in template:
        return [template]
    return [template.replace("{f}", rng.choice(FINDING_FILLERS)) for _ in range(n)]


def build_jsonl(seed: int = 7) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for intent, temps in INTENTS.items():
        for ti, t in enumerate(temps):
            for text in dict.fromkeys(expand(t, rng)):
                rows.append({"text": text, "intent": intent, "template_id": f"{intent}:{ti}"})
    with (HERE / "intents.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return rows


PREFIXES = [
    "",
    "",
    "",
    "copilot ",
    "please ",
    "can you tell me ",
    "quick question ",
    "ok ",
    "so ",
    "doctor here, ",
    "hey, ",
]
SUFFIXES = ["", "", "", "?", " please", " for this patient", " in this study", " here", " now"]


def augment(text: str, rng: random.Random, n: int = 3) -> list[str]:
    out = [text]
    for _ in range(n):
        words = text.split()
        if len(words) > 3:
            words = [w for w in words if rng.random() > 0.12] or words
        out.append(rng.choice(PREFIXES) + " ".join(words) + rng.choice(SUFFIXES))
    return out


def _fit(pipe: Pipeline, rows: list[dict[str, Any]], seed: int = 0) -> Pipeline:
    rng = random.Random(seed)
    xs, ys = [], []
    for r in rows:
        for t in augment(r["text"], rng):
            xs.append(featurize(t))
            ys.append(r["intent"])
    return pipe.fit(xs, ys)


def make_pipeline() -> Pipeline:
    feats = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
        ]
    )
    return Pipeline([("feats", feats), ("clf", LogisticRegression(C=4.0, max_iter=3000))])


def template_cv(rows: list[dict[str, Any]], folds: int = 5) -> list[float]:
    """Stricter check: hold out 20% of *templates* (unseen phrasings) over several seeds."""
    templates = sorted({r["template_id"] for r in rows})
    accs = []
    for seed in range(folds):
        rng = np.random.default_rng(seed)
        held = {t for t in templates if rng.random() < 0.2}
        tr = [r for r in rows if r["template_id"] not in held]
        te = [r for r in rows if r["template_id"] in held]
        pred = _fit(make_pipeline(), tr).predict([featurize(r["text"]) for r in te])
        accs.append(float(np.mean(pred == np.array([r["intent"] for r in te]))))
    return accs


def train(profile: str = "starter") -> dict[str, Any]:
    rows = build_jsonl()
    test = [json.loads(line) for line in (HERE / "intents_test.jsonl").read_text().splitlines()]
    final = _fit(make_pipeline(), rows)
    pred = final.predict([featurize(r["text"]) for r in test])
    y = [r["intent"] for r in test]
    cv = template_cv(rows)
    metrics = {
        "model": "copilot_intent",
        "profile": profile,
        "task": "intent",
        "n_intents": len(INTENTS),
        "n_examples": len(rows),
        "n_templates": len({r["template_id"] for r in rows}),
        "n_test": len(test),
        "test": {
            "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "template_cv_accuracy_mean": float(np.mean(cv)),
            "template_cv_accuracy_folds": cv,
            "per_intent": classification_report(y, pred, output_dict=True, zero_division=0),
            "errors": [
                {"text": r["text"], "true": r["intent"], "pred": p}
                for r, p in zip(test, pred)
                if p != r["intent"]
            ],
        },
        "eval_protocol": "held-out set of 110 hand-written natural questions (intents_test.jsonl, "
        "never used for training) + 5-fold held-out-template CV",
    }
    out = artifact_dir("copilot_intent", profile)
    joblib.dump(final, out / "model.joblib")
    write_json(out / "metrics.json", metrics)
    print(
        f"[train] copilot_intent: held-out test accuracy {metrics['test']['accuracy']:.3f} "
        f"({len(test)} questions, {len(INTENTS)} intents); template-CV {np.mean(cv):.3f}"
    )
    return metrics


if __name__ == "__main__":
    train()
