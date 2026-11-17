"""Metrics shared by training, evaluation and the backend (calibration helpers)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def safe_auroc(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def multilabel_auroc(y: np.ndarray, p: np.ndarray) -> tuple[list[float | None], float]:
    per = [safe_auroc(y[:, k], p[:, k]) for k in range(y.shape[1])]
    vals = [v for v in per if v is not None]
    return per, float(np.mean(vals)) if vals else float("nan")


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    t = float(thr[int(np.argmax(j))])
    return float(min(max(t, 1e-4), 0.9999))


def sens_spec(y: np.ndarray, p: np.ndarray, t: float) -> tuple[float, float]:
    pred = p >= t
    tp = float(np.sum(pred & (y == 1)))
    fn = float(np.sum(~pred & (y == 1)))
    tn = float(np.sum(~pred & (y == 0)))
    fp = float(np.sum(pred & (y == 0)))
    return tp / max(tp + fn, 1), tn / max(tn + fp, 1)


def roc_points(y: np.ndarray, p: np.ndarray, n: int = 50) -> list[list[float]]:
    if len(np.unique(y)) < 2:
        return []
    fpr, tpr, _ = roc_curve(y, p)
    grid = np.linspace(0, 1, n)
    return [[float(g), float(np.interp(g, fpr, tpr))] for g in grid]


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """1-D temperature minimising binary NLL on validation logits (grid + golden refinement)."""
    if len(np.unique(y)) < 2:
        return 1.0

    def nll(t: float) -> float:
        p = np.clip(sigmoid(logits / t), 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    grid = np.exp(np.linspace(np.log(0.25), np.log(8.0), 60))
    best = float(grid[int(np.argmin([nll(t) for t in grid]))])
    lo, hi = best / 1.15, best * 1.15
    for _ in range(30):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if nll(m1) < nll(m2):
            hi = m2
        else:
            lo = m1
    return float((lo + hi) / 2)


def fit_temperature_multiclass(logits: np.ndarray, y: np.ndarray) -> float:
    def nll(t: float) -> float:
        z = logits / t
        z = z - z.max(1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return float(-np.mean(logp[np.arange(len(y)), y]))

    grid = np.exp(np.linspace(np.log(0.25), np.log(8.0), 80))
    return float(grid[int(np.argmin([nll(t) for t in grid]))])


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict[str, Any]:
    """Reliability diagram bins + expected calibration error (ECE) for binary probabilities."""
    edges = np.linspace(0, 1, bins + 1)
    out, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum() == 0:
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        ece += m.mean() * abs(conf - acc)
        out.append(
            {
                "bin": [float(lo), float(hi)],
                "confidence": conf,
                "accuracy": acc,
                "count": int(m.sum()),
            }
        )
    return {"bins": out, "ece": float(ece)}


def classification_report(y: np.ndarray, logits: np.ndarray, classes: list[str]) -> dict[str, Any]:
    pred = logits.argmax(1)
    z = logits - logits.max(1, keepdims=True)
    prob = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    per_auc = [safe_auroc((y == k).astype(int), prob[:, k]) for k in range(len(classes))]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "per_class_f1": dict(zip(classes, f1_score(y, pred, average=None).tolist())),
        "per_class_auroc": dict(zip(classes, per_auc)),
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(len(classes)))).tolist(),
        "classes": classes,
    }


def dice_iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> tuple[float, float]:
    """Mean per-image Dice and IoU on binary masks (N,H,W). Empty-vs-empty counts as 1."""
    dices, ious = [], []
    for p, g in zip(pred.astype(bool), gt.astype(bool)):
        inter = float((p & g).sum())
        s = float(p.sum() + g.sum())
        union = float((p | g).sum())
        dices.append(1.0 if s == 0 else 2 * inter / (s + eps))
        ious.append(1.0 if union == 0 else inter / (union + eps))
    return float(np.mean(dices)), float(np.mean(ious))


def psnr(clean: np.ndarray, test: np.ndarray) -> float:
    mse = float(np.mean((clean.astype(np.float64) - test.astype(np.float64)) ** 2))
    return float(10 * np.log10(1.0 / max(mse, 1e-12)))


def ssim(clean: np.ndarray, test: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    return float(structural_similarity(clean, test, data_range=1.0))


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None] - inter + 1e-9)


def map50(
    preds: list[dict[str, np.ndarray]], gts: list[np.ndarray], iou_t: float = 0.5
) -> dict[str, float]:
    """Single-class VOC-style AP@IoU (all-point interpolation) + recall at score>=0.05."""
    scores, tps = [], []
    n_gt = sum(len(g) for g in gts)
    for pr, gt in zip(preds, gts):
        order = np.argsort(-pr["scores"])
        boxes, sc = pr["boxes"][order], pr["scores"][order]
        matched = np.zeros(len(gt), bool)
        ious = box_iou(boxes, gt)
        for i in range(len(boxes)):
            scores.append(sc[i])
            if len(gt):
                j = int(np.argmax(ious[i]))
                if ious[i, j] >= iou_t and not matched[j]:
                    matched[j] = True
                    tps.append(1)
                    continue
            tps.append(0)
    if n_gt == 0 or not scores:
        return {"map50": 0.0, "recall": 0.0, "precision": 0.0}
    order = np.argsort(-np.array(scores))
    tp = np.array(tps)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return {"map50": ap, "recall": float(rec[-1]), "precision": float(prec[-1])}
