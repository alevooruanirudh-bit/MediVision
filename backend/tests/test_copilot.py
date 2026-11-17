"""Copilot: intent model accuracy, entity extraction, grounded answers for every intent."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.copilot.chat import COMPOSERS, answer, classify
from app.copilot.entities import extract
from app.inference.registry import registry
from ml.copilot.text import featurize

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module", autouse=True)
def _load() -> None:
    registry.load()
    if registry.extra.get("intent_model") is None:
        pytest.skip("intent model not trained")


def test_intent_model_accuracy_on_held_out_set() -> None:
    rows = [
        json.loads(x) for x in (ROOT / "ml/copilot/intents_test.jsonl").read_text().splitlines()
    ]
    model = registry.extra["intent_model"]
    pred = model.predict([featurize(r["text"]) for r in rows])
    acc = sum(p == r["intent"] for p, r in zip(pred, rows)) / len(rows)
    assert acc >= 0.90, acc
    assert len({r["intent"] for r in rows}) >= 20


def test_intent_dataset_has_40_paraphrases_per_intent() -> None:
    from ml.copilot.intents_source import INTENTS

    assert len(INTENTS) >= 20
    assert min(len(v) for v in INTENTS.values()) >= 40


def test_entities() -> None:
    e = extract("is there a pleural efusion in the left lower zone? make the impression concise")
    assert "Effusion" in e["findings"]  # fuzzy match despite the typo
    assert e["side"] == "left" and e["zone"] == "lower"
    assert e["section"] == "impression" and e["style"] == "concise"
    assert extract("show grad-cam++ for edema")["method"] == "gradcampp"


def _study() -> SimpleNamespace:
    f = [
        {
            "label": "Effusion",
            "key": "Effusion",
            "probability": 0.62,
            "threshold": 0.21,
            "uncertainty": 0.05,
            "ensemble_std": 0.04,
            "mc_dropout_var": 0.001,
            "positive": True,
            "certainty": "Likely",
            "ratio_to_threshold": 2.95,
            "zone": "left lower zone",
            "member_probs": {
                "chest_densenet121": 0.6,
                "chest_efficientnet_b2": 0.66,
                "chest_convnext_tiny": 0.6,
            },
            "kind": "classification",
        },
        {
            "label": "Pneumothorax",
            "key": "Pneumothorax",
            "probability": 0.03,
            "threshold": 0.15,
            "uncertainty": 0.01,
            "positive": False,
            "certainty": "Unlikely",
            "ratio_to_threshold": 0.2,
            "zone": None,
            "kind": "classification",
        },
    ]
    res = {
        "pipeline": "chest",
        "findings": f,
        "detections": [],
        "measurements": {
            "available": True,
            "ctr": 0.48,
            "ctr_flag": False,
            "right_lung_area_px": 5000,
            "left_lung_area_px": 4800,
            "symmetry_index": 0.96,
        },
        "modality": {"label": "chest_xray", "confidence": 0.97, "warning": None, "ood": False},
        "triage": {
            "score": 41.0,
            "level": "ROUTINE",
            "drivers": [{"label": "Effusion", "contribution": 41.0}],
            "note": None,
        },
        "extra": {
            "chest": {
                "members": ["a", "b", "c"],
                "energy": 3.2,
                "energy_threshold": 1.0,
                "energy_ood": False,
            }
        },
        "model_profile": "starter",
        "backend": "onnx",
    }
    return SimpleNamespace(id="s1", patient_id="p1", results=res, indication=None, created_at=None)


CANON = {
    "greeting_help": "hello",
    "list_findings": "what did you find?",
    "explain_finding": "explain the effusion",
    "why_finding": "why do you think there is effusion",
    "where_finding": "where is the effusion",
    "confidence": "how confident are you about the effusion",
    "next_steps": "what should we do next",
    "compare_previous": "compare with the previous study",
    "define_term": "what is atelectasis",
    "show_heatmap": "show the heatmap for effusion",
    "measurements": "what is the ctr",
    "normal_or_not": "is this scan normal",
    "urgency": "how urgent is this",
    "limitations": "what are your limitations",
    "rewrite_section": "make the impression concise",
    "summary": "summarize the study",
    "differential": "what else could the effusion be",
    "uncertainty": "do the models disagree",
    "model_info": "which datasets were used to train you",
    "image_quality": "is the image quality ok",
    "patient_explanation": "explain the effusion to the patient in simple terms",
    "check_finding": "is there a pneumothorax",
}


@pytest.mark.parametrize("intent,question", list(CANON.items()))
def test_every_intent_routes_and_answers(intent: str, question: str) -> None:
    got, conf, _ = classify(question)
    assert got == intent, (question, got, conf)
    a = answer(question, _study())  # type: ignore[arg-type]
    assert a.text and a.intent in (intent, "clarify")
    assert "definitely" not in a.text.lower()


def test_answers_are_grounded_in_study_data() -> None:
    s = _study()
    assert "62%" in answer("how confident are you about the effusion", s).text  # type: ignore[arg-type]
    assert "left lower zone" in answer("where is the effusion", s).text  # type: ignore[arg-type]
    assert "0.48" in answer("what is the ctr", s).text  # type: ignore[arg-type]
    assert "not flagged" in answer("is there a pneumothorax", s).text  # type: ignore[arg-type]
    why = answer("why effusion?", s)  # type: ignore[arg-type]
    assert any(a["type"] == "show_heatmap" for a in why.actions)
    assert "correlate clinically" in answer("is this scan normal", s).text  # type: ignore[arg-type]


def test_unknown_question_falls_back_honestly() -> None:
    a = answer("purple monkey dishwasher quantum banana", _study())  # type: ignore[arg-type]
    assert a.intent in ("fallback", "greeting_help") and a.suggestions


def test_all_composers_registered() -> None:
    assert len(COMPOSERS) >= 20
