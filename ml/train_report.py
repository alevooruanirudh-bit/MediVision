"""Training + evaluation for the EXPERIMENTAL report generator (IU X-ray, from scratch).

The BPE tokenizer is trained by us on the training-split reports (HuggingFace `tokenizers` library,
used offline as a tokenisation algorithm only — no pretrained vocabularies or models).
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.common import ARTIFACTS, NIH_LABELS, PROCESSED, artifact_dir, get_device, write_json

SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]

# Keyword gazetteer used for the "clinical consistency" check (report text vs labels).
FINDING_KEYWORDS: dict[str, list[str]] = {
    "Atelectasis": ["atelectasis", "atelectatic"],
    "Cardiomegaly": [
        "cardiomegaly",
        "enlarged heart",
        "heart is enlarged",
        "heart size is enlarged",
    ],
    "Effusion": ["effusion"],
    "Infiltration": ["infiltrate", "infiltration"],
    "Mass": ["mass"],
    "Nodule": ["nodule", "nodular"],
    "Pneumonia": ["pneumonia"],
    "Pneumothorax": ["pneumothorax"],
    "Consolidation": ["consolidation", "airspace disease", "airspace opacity"],
    "Edema": ["edema", "vascular congestion"],
    "Emphysema": ["emphysema", "hyperinflat", "copd"],
    "Fibrosis": ["fibrosis", "fibrotic", "scarring"],
    "Pleural_Thickening": ["pleural thickening"],
    "Hernia": ["hernia"],
}
NEGATION = re.compile(r"\b(no|without|negative for|free of|resolved|clear of)\b[^.]*$")


def mentioned_findings(text: str) -> set[str]:
    """Findings asserted (not negated) in a report, sentence by sentence."""
    found: set[str] = set()
    for sent in re.split(r"[.;]", text.lower()):
        for lab, kws in FINDING_KEYWORDS.items():
            for kw in kws:
                pos = sent.find(kw)
                if pos >= 0 and not NEGATION.search(sent[:pos]):
                    found.add(lab)
    return found


def train_tokenizer(texts: list[str], vocab_size: int) -> Any:
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL, min_frequency=2)
    tok.train_from_iterator(texts, trainer)
    return tok


# ----------------------------------------------------------------------------- text metrics
def _ngrams(toks: list[str], n: int) -> Counter:
    return Counter(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))


def corpus_bleu4(refs: list[str], hyps: list[str]) -> float:
    num = [0] * 4
    den = [0] * 4
    ref_len = hyp_len = 0
    for r, h in zip(refs, hyps):
        rt, ht = r.split(), h.split()
        ref_len += len(rt)
        hyp_len += len(ht)
        for n in range(1, 5):
            hc, rc = _ngrams(ht, n), _ngrams(rt, n)
            num[n - 1] += sum(min(c, rc[g]) for g, c in hc.items())
            den[n - 1] += max(sum(hc.values()), 0)
    if min(num) == 0:
        return 0.0
    logp = sum(math.log(num[i] / max(den[i], 1)) for i in range(4)) / 4
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return float(bp * math.exp(logp))


def rouge_l(refs: list[str], hyps: list[str]) -> float:
    scores = []
    for r, h in zip(refs, hyps):
        a, b = r.split(), h.split()
        if not a or not b:
            scores.append(0.0)
            continue
        dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(len(a)):
            for j in range(len(b)):
                dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i][j + 1], dp[i + 1][j])
        lcs = dp[-1][-1]
        p, rc = lcs / len(b), lcs / len(a)
        scores.append(0.0 if lcs == 0 else 2 * p * rc / (p + rc))
    return float(np.mean(scores))


def consistency(refs: list[str], hyps: list[str]) -> dict[str, float]:
    """Micro P/R/F1 of asserted findings in generated vs reference reports, plus normal agreement."""
    tp = fp = fn = 0
    normal_agree = []
    for r, h in zip(refs, hyps):
        a, b = mentioned_findings(r), mentioned_findings(h)
        tp += len(a & b)
        fp += len(b - a)
        fn += len(a - b)
        normal_agree.append(float((not a) == (not b)))
    p = tp / max(tp + fp, 1)
    rc = tp / max(tp + fn, 1)
    return {
        "precision": p,
        "recall": rc,
        "f1": 2 * p * rc / max(p + rc, 1e-9),
        "normal_vs_abnormal_agreement": float(np.mean(normal_agree)),
    }


# ----------------------------------------------------------------------------- training
def train_report(cfg: dict[str, Any], resume: bool = False) -> dict[str, Any]:
    from ml.data.datasets import report_text
    from ml.models import build_model
    from ml.train import EarlyStopper, cosine_with_warmup, finalize, make_datasets

    t = cfg["train"]
    device = get_device(t.get("device", "auto"))
    out = artifact_dir(cfg["name"], cfg["profile"])
    meta = pd.read_csv(PROCESSED / "iu_xray" / "reports.csv").fillna("")
    texts = [report_text(r) for _, r in meta[meta["split"] == "train"].iterrows()]
    tok = train_tokenizer(texts, cfg["model"]["vocab_size"])
    tok.save(str(out / "tokenizer.json"))
    cfg["model"]["vocab_size"] = tok.get_vocab_size()
    train_ds, val_ds, test_ds = make_datasets(cfg, tok)
    tl = DataLoader(train_ds, t["batch_size"], shuffle=True, drop_last=True)
    vl = DataLoader(val_ds, t["batch_size"])
    model = build_model(cfg).to(device)
    init = t.get("encoder_init")
    if init:  # reuse OUR chest classifier's DenseNet trunk
        for prof in (cfg["profile"], "starter"):
            p = ARTIFACTS / init / prof / "model.pt"
            if p.exists():
                sd = torch.load(p, map_location="cpu", weights_only=False)["state_dict"]
                enc = {k[len("backbone.") :]: v for k, v in sd.items() if k.startswith("backbone.")}
                missing = model.encoder.load_state_dict(enc, strict=False)
                print(f"[train] report_gen encoder <- {p} (missing {len(missing.missing_keys)})")
                break
    pad = tok.token_to_id("<pad>")
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad, label_smoothing=0.1)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"].get("weight_decay", 0.01),
    )
    epochs = t["epochs"]
    sched = cosine_with_warmup(opt, t.get("warmup_epochs", 1) * len(tl), epochs * len(tl))
    stopper = EarlyStopper(t.get("patience", 4), "min")
    history: list[dict[str, Any]] = []
    freeze = t.get("freeze_encoder_epochs", 0)
    print(
        f"[train] report_gen on {device}: {len(train_ds)} train / {len(val_ds)} val, "
        f"vocab {tok.get_vocab_size()}, params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M"
    )
    (out / "checkpoints").mkdir(exist_ok=True)
    for epoch in range(epochs):
        for p in model.encoder.parameters():
            p.requires_grad = epoch >= freeze
        model.train()
        t0, run, n = time.time(), 0.0, 0
        for x, seq in tl:
            x, seq = x.to(device), seq.to(device)
            logits = model(x, seq[:, :-1])
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), seq[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item()
            n += 1
        model.eval()
        vloss, vn = 0.0, 0
        with torch.no_grad():
            for x, seq in vl:
                x, seq = x.to(device), seq.to(device)
                logits = model(x, seq[:, :-1])
                vloss += loss_fn(
                    logits.reshape(-1, logits.shape[-1]), seq[:, 1:].reshape(-1)
                ).item()
                vn += 1
        rec = {
            "epoch": epoch + 1,
            "train_loss": run / n,
            "val_loss": vloss / max(vn, 1),
            "seconds": round(time.time() - t0, 1),
        }
        history.append(rec)
        print(f"[train] report_gen epoch {epoch + 1}/{epochs} {rec}", flush=True)
        if stopper.step(rec["val_loss"]):
            torch.save(model.state_dict(), out / "checkpoints" / "best.pt")
        if stopper.stop:
            break
    model.load_state_dict(torch.load(out / "checkpoints" / "best.pt", map_location=device))
    summary = finalize(cfg, model, history, {"val_loss": stopper.best})
    evaluate_report(cfg, model.cpu(), tok, test_ds)
    return summary


def evaluate_report(
    cfg: dict[str, Any], model: nn.Module, tok: Any, test_ds: Any, limit: int = 150
) -> dict[str, Any]:
    model.eval()
    bos, eos = tok.token_to_id("<bos>"), tok.token_to_id("<eos>")
    refs, hyps, examples = [], [], []
    n = min(limit, len(test_ds))
    for i in range(n):
        x, _ = test_ds[i]
        ids = model.generate(x[None], bos, eos, max_new=80, beam=2)[0]
        hyp = " ".join(tok.decode(ids).split())
        ref = test_ds.texts[i]
        refs.append(ref)
        hyps.append(hyp)
        if len(examples) < 5:
            examples.append({"reference": ref, "generated": hyp})
    # Baseline: always emit the most common training report (a strong, honest reference point).
    common = Counter(test_ds.texts).most_common(1)[0][0]
    metrics = {
        "n_test": n,
        "bleu4": corpus_bleu4(refs, hyps),
        "rouge_l": rouge_l(refs, hyps),
        "clinical_consistency": consistency(refs, hyps),
        "baseline_most_common_report": {
            "bleu4": corpus_bleu4(refs, [common] * n),
            "rouge_l": rouge_l(refs, [common] * n),
        },
        "examples": examples,
        "labels_checked": NIH_LABELS,
    }
    out = artifact_dir(cfg["name"], cfg["profile"])
    write_json(
        out / "metrics.json",
        {"model": cfg["name"], "profile": cfg["profile"], "task": "report", "test": metrics},
    )
    print(f"[eval] report_gen BLEU-4 {metrics['bleu4']:.3f} ROUGE-L {metrics['rouge_l']:.3f}")
    return metrics
