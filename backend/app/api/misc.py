"""Triage worklist, finding reviews, compare, enhance, federated, metrics, audit, dashboard, system."""

from __future__ import annotations

import csv
import io
import json
import platform
import time
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import Response
from pydantic import BaseModel
from sqlmodel import Session, col, select
from sse_starlette.sse import EventSourceResponse

from app.api.deps import APIError, get_study, study_out
from app.config import APP_VERSION, DISCLAIMER, get_settings
from app.db import get_session
from app.inference.registry import registry
from app.models import AuditLog, Finding, Review, Study
from app.services import storage
from app.services.audit import audit

router = APIRouter(tags=["workflow"])
START = time.time()
LEVEL_ORDER = {"CRITICAL": 0, "URGENT": 1, "ROUTINE": 2, "NORMAL": 3, None: 4}


# ----------------------------------------------------------------------------- triage
@router.get("/api/triage/worklist", summary="Triage worklist sorted by urgency")
def worklist(
    level: str | None = None,
    q: str | None = None,
    modality: str | None = None,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    rows = [study_out(session, s, full=False) for s in session.exec(select(Study)).all()]
    if level:
        rows = [r for r in rows if (r["urgency_level"] or "") == level.upper()]
    if modality:
        rows = [r for r in rows if (r["modality"] or "").startswith(modality)]
    if q:
        rows = [r for r in rows if q.lower() in json.dumps(r).lower()]
    rows.sort(
        key=lambda r: (
            LEVEL_ORDER.get(r["urgency_level"], 4),
            -(r["urgency_score"] or 0),
            r["created_at"],
        )
    )
    counts = Counter(r["urgency_level"] or "PENDING" for r in rows)
    return {"items": rows, "counts": counts}


# ----------------------------------------------------------------------------- reviews
class ReviewRequest(BaseModel):
    action: str  # accept | reject | edit
    new_label: str | None = None
    comment: str | None = None
    reviewer: str = "radiologist"


@router.post("/api/findings/{finding_id}/review", summary="Accept / reject / edit a finding")
def review(
    finding_id: str, body: ReviewRequest, session: Session = Depends(get_session)
) -> dict[str, Any]:
    f = session.get(Finding, finding_id)
    if f is None:
        raise APIError(404, "finding_not_found", "Finding not found")
    if body.action not in ("accept", "reject", "edit"):
        raise APIError(422, "bad_action", "action must be accept, reject or edit")
    if body.action == "edit" and not body.new_label:
        raise APIError(422, "missing_label", "edit requires new_label")
    r = Review(
        finding_id=f.id,
        study_id=f.study_id,
        action=body.action,
        reviewer=body.reviewer,
        original_label=f.label,
        new_label=body.new_label,
        comment=body.comment,
        model_probability=f.probability,
    )
    f.status = {"accept": "accepted", "reject": "rejected", "edit": "edited"}[body.action]
    f.edited_label = body.new_label
    f.note = body.comment
    session.add(r)
    session.add(f)
    session.commit()
    audit(
        session,
        f"finding.{body.action}",
        "finding",
        f.id,
        actor=body.reviewer,
        label=f.label,
        new_label=body.new_label,
    )
    return {"finding_id": f.id, "status": f.status, "review_id": r.id}


@router.get("/api/findings/reviews", summary="All human reviews (AI–radiologist agreement)")
def reviews(session: Session = Depends(get_session)) -> dict[str, Any]:
    rows = session.exec(select(Review).order_by(col(Review.created_at).desc())).all()
    n = len(rows)
    agree = sum(1 for r in rows if r.action == "accept")
    return {
        "n": n,
        "agreement_rate": agree / n if n else None,
        "items": [r.model_dump(mode="json") for r in rows[:200]],
    }


@router.get("/api/findings/reviews/export", summary="Disagreements as a re-training CSV")
def export_reviews(
    only_disagreements: bool = True, session: Session = Depends(get_session)
) -> Response:
    rows = session.exec(select(Review)).all()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(
        [
            "study_id",
            "image_path",
            "model_label",
            "model_probability",
            "action",
            "corrected_label",
            "reviewer",
            "comment",
            "timestamp",
        ]
    )
    for r in rows:
        if only_disagreements and r.action == "accept":
            continue
        s = session.get(Study, r.study_id)
        label = "" if r.action == "reject" else (r.new_label or r.original_label)
        w.writerow(
            [
                r.study_id,
                s.image_path if s else "",
                r.original_label,
                f"{r.model_probability:.4f}",
                r.action,
                label,
                r.reviewer,
                r.comment or "",
                r.created_at.isoformat(),
            ]
        )
    audit(session, "reviews.export", "review", None, n=len(rows))
    return Response(
        buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="retraining_set.csv"'},
    )


# ----------------------------------------------------------------------------- compare
class CompareRequest(BaseModel):
    prior_id: str
    current_id: str
    finding: str | None = None


@router.post("/api/compare", summary="Longitudinal comparison of two studies")
def compare(body: CompareRequest, session: Session = Depends(get_session)) -> dict[str, Any]:
    from app.inference.compare import compare_results, heatmap_difference

    a, b = get_study(session, body.prior_id), get_study(session, body.current_id)
    if a.status != "analyzed" or b.status != "analyzed":
        raise APIError(409, "not_analyzed", "Both studies must be analysed")
    if a.results.get("pipeline") != b.results.get("pipeline"):
        raise APIError(422, "modality_mismatch", "Studies must be the same modality")
    out = compare_results(a, b)
    key = body.finding or next(
        (d["key"] for d in out["deltas"] if d["status"] != "absent"),
        out["deltas"][0]["key"] if out["deltas"] else None,
    )
    if key:
        try:
            out["heatmap_diff"] = {"finding": key, "url": heatmap_difference(a, b, key)}
        except Exception as exc:  # pragma: no cover
            out["heatmap_diff"] = {"finding": key, "error": str(exc)}
    out["prior_study"] = study_out(session, a, full=False)
    out["current_study"] = study_out(session, b, full=False)
    audit(session, "study.compare", "study", b.id, prior=a.id)
    return out


# ----------------------------------------------------------------------------- enhance
class EnhanceRequest(BaseModel):
    study_id: str
    ops: dict[str, Any] = {}


@router.post("/api/enhance", summary="Enhancement Lab (denoiser, CLAHE, unsharp, gamma, low dose)")
def enhance(body: EnhanceRequest, session: Session = Depends(get_session)) -> dict[str, Any]:
    from app.inference.enhance import apply_ops

    s = get_study(session, body.study_id)
    t0 = time.perf_counter()
    out = apply_ops(storage.load_image(s.image_path), body.ops)
    out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    audit(session, "study.enhance", "study", s.id, ops=list(body.ops))
    return out


# ----------------------------------------------------------------------------- federated
class FedRequest(BaseModel):
    rounds: int = 10
    local_epochs: int = 1


@router.post("/api/federated/run", summary="Run FedAvg across 3 simulated hospitals (SSE)")
async def federated(body: FedRequest) -> Any:
    import asyncio

    from ml.federated.fedavg import run

    rounds = max(2, min(body.rounds, 25))
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Any] = asyncio.Queue()

    def worker() -> None:
        for ev in run(rounds=rounds, local_epochs=max(1, min(body.local_epochs, 3))):
            loop.call_soon_threadsafe(queue.put_nowait, ev)
        loop.call_soon_threadsafe(queue.put_nowait, None)

    loop.run_in_executor(None, worker)

    async def gen():  # type: ignore[no-untyped-def]
        while True:
            ev = await queue.get()
            if ev is None:
                break
            yield {"event": ev["type"], "data": json.dumps(ev)}

    return EventSourceResponse(gen())


@router.get("/api/federated/last", summary="Last federated run summary")
def federated_last() -> dict[str, Any]:
    p = get_settings().artifacts_dir / "federated" / "last_run.json"
    return json.loads(p.read_text()) if p.exists() else {}


# ----------------------------------------------------------------------------- metrics
def _read(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


@router.get("/api/metrics", summary="All evaluation JSON (starter + full) for the Lab page")
def metrics() -> dict[str, Any]:
    root = get_settings().artifacts_dir
    out: dict[str, Any] = {"models": {}}
    for d in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
        if d.name == "federated":
            continue
        for prof in ("starter", "full"):
            m = _read(d / prof / "metrics.json")
            if m is None:
                continue
            m = dict(m)
            m["export"] = _read(d / prof / "export.json")
            m["train_summary"] = _read(d / prof / "train_summary.json")
            m["config"] = _read(d / prof / "config.json")
            m["plots"] = (
                [
                    f"/artifacts/{d.name}/{prof}/plots/{p.name}"
                    for p in sorted((d / prof / "plots").glob("*.png"))
                ]
                if (d / prof / "plots").exists()
                else []
            )
            out["models"].setdefault(d.name, {})[prof] = m
    out["federated"] = _read(root / "federated" / "last_run.json")
    from ml.data.download import SOURCES

    out["datasets"] = [
        {
            "key": s.key,
            "title": s.title,
            "license": s.license,
            "citation": s.citation,
            "url": s.url,
            "profiles": list(s.profiles),
        }
        for s in SOURCES
    ]
    out["dataset_sizes"] = _dataset_sizes()
    return out


def _dataset_sizes() -> dict[str, Any]:
    from ml.common import ROOT

    sizes: dict[str, Any] = {}
    for f in (ROOT / "ml" / "data" / "splits").glob("*.csv"):
        try:
            import pandas as pd

            df = pd.read_csv(f, usecols=["split"])
            sizes[f.stem] = df["split"].value_counts().to_dict()
        except Exception:
            continue
    return sizes


# ----------------------------------------------------------------------------- audit + dashboard
@router.get("/api/audit", summary="Audit log")
def audit_log(
    limit: int = 300, action: str | None = None, session: Session = Depends(get_session)
) -> list[dict[str, Any]]:
    q = select(AuditLog).order_by(col(AuditLog.ts).desc()).limit(limit)
    rows = session.exec(q).all()
    out = [
        {
            "id": r.id,
            "ts": r.ts.isoformat(),
            "actor": r.actor,
            "action": r.action,
            "entity": r.entity,
            "entity_id": r.entity_id,
            "details": json.loads(r.details_json) if r.details_json else {},
        }
        for r in rows
    ]
    return [o for o in out if not action or o["action"].startswith(action)]


@router.get("/api/dashboard", summary="Dashboard KPIs")
def dashboard(session: Session = Depends(get_session)) -> dict[str, Any]:
    studies = session.exec(select(Study)).all()
    analyzed = [s for s in studies if s.status == "analyzed"]
    levels = Counter(s.urgency_level or "PENDING" for s in studies)
    findings = Counter()
    times = []
    modality = Counter(s.modality or "unknown" for s in analyzed)
    for s in analyzed:
        r = s.results
        for f in r.get("findings", []):
            if f.get("positive"):
                findings[f["label"]] += 1
        t = r.get("timings_ms", {}).get("total")
        if t:
            times.append(
                {
                    "id": s.id,
                    "ms": t,
                    "at": s.analyzed_at.isoformat() if s.analyzed_at else None,
                    "modality": s.modality,
                }
            )
    revs = session.exec(select(Review)).all()
    agree = sum(1 for r in revs if r.action == "accept")
    recent = audit_log(limit=12, action=None, session=session)
    now = datetime.now(UTC)
    week = [s for s in studies if s.created_at.replace(tzinfo=UTC) > now - timedelta(days=7)]
    return {
        "total_studies": len(studies),
        "analyzed": len(analyzed),
        "this_week": len(week),
        "critical": levels.get("CRITICAL", 0),
        "urgent": levels.get("URGENT", 0),
        "levels": levels,
        "findings_frequency": findings.most_common(12),
        "modalities": modality,
        "analysis_times": sorted(times, key=lambda x: x["at"] or "")[-30:],
        "mean_analysis_ms": round(sum(t["ms"] for t in times) / len(times), 1) if times else None,
        "reviews": len(revs),
        "agreement_rate": agree / len(revs) if revs else None,
        "ood_count": sum(1 for s in analyzed if s.ood),
        "recent_activity": recent,
    }


# ----------------------------------------------------------------------------- system
@router.get("/api/system/health", summary="Health: models, profiles, runtime, versions")
def health() -> dict[str, Any]:
    import onnxruntime
    import torch

    from app.copilot.local_llm import status as llm_status

    st = registry.status()
    loaded = [k for k, v in st.items() if v.get("loaded")]
    profiles = {v["profile"] for v in st.values() if v.get("loaded")}
    return {
        "status": "ok",
        "version": APP_VERSION,
        "disclaimer": DISCLAIMER,
        "uptime_s": round(time.time() - START),
        "models": st,
        "n_models_loaded": len(loaded),
        "active_profile": "full" if "full" in profiles else "starter" if profiles else None,
        "chest_profile": registry.extra.get("chest_profile"),
        "inference_backend": get_settings().inference_backend,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "onnxruntime": onnxruntime.__version__,
        },
        "local_llm": llm_status(),
        "external_ai_apis": "none",
        "settings": {
            "mc_dropout_samples": get_settings().mc_dropout_samples,
            "max_upload_mb": get_settings().max_upload_mb,
            "model_profile": get_settings().model_profile,
        },
    }


@router.post("/api/system/reload", summary="Reload model weights from artifacts/")
def reload_models() -> dict[str, Any]:
    registry.load()
    return registry.status()


@router.post("/api/demo/load", summary="Load demo patients/studies")
def load_demo(session: Session = Depends(get_session)) -> dict[str, Any]:
    from app.seed import seed

    return seed(session)


@router.get(
    "/api/model-card", summary="Model card data: triage weights, thresholds, headline metrics"
)
def model_card() -> dict[str, Any]:
    from app.inference.triage import config as triage_config

    cal = registry.extra.get("chest_calibration", {})
    headline: dict[str, Any] = {}
    for name, h in registry.handles.items():
        t = h.metrics.get("test", {})
        headline[name] = {
            "profile": h.profile,
            **{k: v for k, v in t.items() if isinstance(v, int | float)},
        }
    cm = registry.extra.get("chest_metrics", {})
    if cm:
        headline["chest_ensemble"] = {
            "profile": registry.extra.get("chest_profile"),
            "macro_auroc": cm.get("test", {}).get("macro_auroc"),
            "fairness_sex_gap": (cm.get("fairness") or {}).get("sex_gap"),
        }
    return {
        "triage": triage_config(),
        "thresholds": cal.get("thresholds", {}),
        "temperatures": cal.get("temperatures", {}),
        "labels": cal.get("labels", []),
        "energy_threshold": cal.get("energy_threshold"),
        "chest_profile": registry.extra.get("chest_profile"),
        "headline": headline,
    }
