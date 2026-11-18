"""Render the measured metrics of every trained model into docs/METRICS.md and the README table.

python -m ml.metrics_table
"""

from __future__ import annotations

import json
import re
from typing import Any

from ml.common import ARTIFACTS, ROOT

ROWS = [
    (
        "chest_ensemble",
        "Chest X-ray ensemble (14 labels)",
        "macro AUROC",
        lambda t: t.get("macro_auroc"),
    ),
    ("chest_densenet121", "· DenseNet-121", "macro AUROC", lambda t: t.get("macro_auroc")),
    ("chest_efficientnet_b2", "· EfficientNet-B2", "macro AUROC", lambda t: t.get("macro_auroc")),
    ("chest_convnext_tiny", "· ConvNeXt-Tiny", "macro AUROC", lambda t: t.get("macro_auroc")),
    ("lung_unet", "Lung U-Net", "Dice", lambda t: t.get("dice")),
    ("fracture_cls", "Fracture classifier", "AUROC", lambda t: t.get("auroc")),
    ("fracture_det", "Fracture detector", "mAP@0.5", lambda t: t.get("map50")),
    (
        "brain_cls",
        "Brain MRI classifier",
        "accuracy / macro-F1",
        lambda t: f"{t.get('accuracy', 0):.3f} / {t.get('macro_f1', 0):.3f}",
    ),
    (
        "brain_seg",
        "Brain tumour Attention U-Net",
        "Dice (tumour slices)",
        lambda t: t.get("dice_positive_slices"),
    ),
    (
        "router",
        "Modality router",
        "accuracy / macro-F1",
        lambda t: f"{t.get('accuracy', 0):.3f} / {t.get('macro_f1', 0):.3f}",
    ),
    (
        "denoiser",
        "DnCNN denoiser",
        "PSNR noisy → ours (dB)",
        lambda t: f"{t.get('psnr_noisy', 0):.1f} → {t.get('psnr_denoised', 0):.1f}",
    ),
    (
        "report_gen",
        "Report generator (experimental)",
        "BLEU-4 / ROUGE-L",
        lambda t: f"{t.get('bleu4', 0):.3f} / {t.get('rouge_l', 0):.3f}",
    ),
    ("copilot_intent", "Copilot intent model", "held-out accuracy", lambda t: t.get("accuracy")),
    (
        "head_ct_cls",
        "Head-CT hemorrhage classifier",
        "AUROC · sens / spec",
        lambda t: f"{t.get('auroc', 0):.3f} · {t.get('sensitivity', 0):.2f} / "
        f"{t.get('specificity', 0):.2f}",
    ),
    (
        "nodule3d",
        "Chest-CT nodule malignancy (3-D CNN)",
        "AUROC · sens / spec",
        lambda t: f"{t.get('auroc', 0):.3f} · {t.get('sensitivity', 0):.2f} / "
        f"{t.get('specificity', 0):.2f}",
    ),
    (
        "breast_us",
        "Breast-ultrasound malignancy",
        "AUROC · sens / spec",
        lambda t: f"{t.get('auroc', 0):.3f} · {t.get('sensitivity', 0):.2f} / "
        f"{t.get('specificity', 0):.2f}",
    ),
    (
        "retina_dr",
        "Diabetic-retinopathy grader (fundus)",
        "referable AUROC · κ",
        lambda t: f"{t.get('referable_auroc', 0):.3f} · {t.get('quadratic_kappa', 0):.3f}",
    ),
    (
        "mammo_cls",
        "Screening mammography (Mini-DDSM)",
        "exam AUROC · sens / spec (view AUROC)",
        lambda t: f"{t['exam_level']['auroc']:.3f} · {t['exam_level']['sensitivity']:.2f} / "
        f"{t['exam_level']['specificity']:.2f} ({t.get('auroc', 0):.3f})",
    ),
    (
        "lung_nodule_det",
        "Whole-scan lung-nodule detector (LUNA16)",
        "FROC CPM · sens @1 / @8 FP per scan",
        lambda t: f"{t['test_froc']['cpm']:.3f} · {t['test_froc']['sens']['1']:.2f} / "
        f"{t['test_froc']['sens']['8']:.2f}",
    ),
    (
        "petct_seg",
        "PET-CT tumour segmentation (7 pts, LOPO)",
        "Dice CT / PET / PET+CT",
        lambda t: f"{t['modes']['ct']['mean_dice']:.2f} / {t['modes']['pet']['mean_dice']:.2f} / "
        f"{t['modes']['petct']['mean_dice']:.2f}",
    ),
    (
        "fmri_glm",
        "fMRI motor GLM (2 subjects)",
        "somatotopic order reproduced",
        lambda t: " · ".join(
            f"{k} {'yes' if v['somatotopy_pass'] else 'no'}"
            for k, v in t.get("subjects", {}).items()
        ),
    ),
    (
        "sr_gan",
        "Super-resolution x2 (SR-GAN)",
        "PSNR bicubic → ours (dB)",
        lambda t: f"{t.get('psnr_bicubic', 0):.2f} → {t.get('psnr_ours', 0):.2f}",
    ),
    (
        "radiogenomics",
        "Radiogenomics (LGG / TCGA)",
        "targets above chance",
        lambda t: f"{sum(v['above_chance'] for v in t.get('targets', {}).values())}"
        f"/{len(t.get('targets', {}))}",
    ),
]


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return "—" if v is None else str(v)


def table(profile: str = "starter") -> str:
    lines = ["| Model | Metric (held-out test) | Value | CPU ms/img (ONNX) |", "|---|---|---|---|"]
    for key, name, metric, fn in ROWS:
        p = ARTIFACTS / key / profile / "metrics.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        exp = ARTIFACTS / key / profile / "export.json"
        cpu = json.loads(exp.read_text()).get("cpu_ms_per_image") if exp.exists() else None
        cpu_txt = f"{cpu:.1f}" if cpu else "—"
        lines.append(f"| {name} | {metric} | {_fmt(fn(m.get('test', m)))} | {cpu_txt} |")
    return "\n".join(lines)


def fairness(profile: str = "starter") -> str:
    p = ARTIFACTS / "chest_ensemble" / profile / "metrics.json"
    if not p.exists():
        return ""
    f = json.loads(p.read_text()).get("fairness") or {}
    if not f.get("groups"):
        return ""
    rows = ["| Subgroup | n | macro AUROC |", "|---|---|---|"]
    rows += [f"| {g} | {v['n']} | {v['macro_auroc']:.3f} |" for g, v in f["groups"].items()]
    return (
        f"Dataset: {f.get('dataset')}\n\n"
        + "\n".join(rows)
        + (f"\n\nSex gap: {f.get('sex_gap', 0):.3f} · max subgroup gap: {f.get('max_gap', 0):.3f}")
    )


def second_reader_section(profile: str = "starter") -> list[str]:
    """Sensitivity / specificity / FPR / FNR at each model's operating point (paper M7)."""
    rows = ["| Model | Target | Sens | Spec | FPR | FNR |", "|---|---|---|---|---|---|"]

    def add(model: str, target: str, se: float | None, sp: float | None) -> None:
        if se is None or sp is None:
            return
        rows.append(f"| {model} | {target} | {se:.3f} | {sp:.3f} | {1 - sp:.3f} | {1 - se:.3f} |")

    ce = ARTIFACTS / "chest_ensemble" / profile / "metrics.json"
    if ce.exists():
        for lab, v in json.loads(ce.read_text())["test"].get("per_label", {}).items():
            add("Chest ensemble", lab, v.get("sensitivity"), v.get("specificity"))
    for key, name in (
        ("fracture_cls", "Fracture classifier"),
        ("head_ct_cls", "Head-CT hemorrhage"),
        ("nodule3d", "CT nodule malignancy"),
        ("breast_us", "Breast-US malignancy"),
        ("mammo_cls", "Mammography (per view)"),
    ):
        p = ARTIFACTS / key / profile / "metrics.json"
        if p.exists():
            t = json.loads(p.read_text())["test"]
            add(name, "positive", t.get("sensitivity"), t.get("specificity"))
    p = ARTIFACTS / "mammo_cls" / profile / "metrics.json"
    if p.exists():
        e = json.loads(p.read_text())["test"].get("exam_level") or {}
        add("Mammography (per exam)", "cancer exam", e.get("sensitivity"), e.get("specificity"))
    p = ARTIFACTS / "retina_dr" / profile / "metrics.json"
    if p.exists():
        r = json.loads(p.read_text())["test"].get("referable") or {}
        add("Fundus DR grader", "referable DR", r.get("sensitivity"), r.get("specificity"))
    return [
        "## Second-reader error rates (operating point = Youden's J on validation)",
        "",
        *rows,
        "",
    ]


def main() -> None:
    fed = ARTIFACTS / "federated" / "last_run.json"
    fed_txt = ""
    if fed.exists():
        s = json.loads(fed.read_text())
        fed_txt = (
            f"FedAvg ({s['rounds']} rounds): test AUROC {s['federated']['auroc']:.3f} · "
            f"centralised {s['centralized']['auroc']:.3f} · local-only mean "
            f"{s['local_only_mean_auroc']:.3f} · {s['seconds']} s on CPU"
        )
    body = [
        "# Measured metrics",
        "",
        "Generated by `python -m ml.metrics_table` from `artifacts/*/<profile>/metrics.json`.",
        "",
    ]
    for prof in ("starter", "full"):
        t = table(prof)
        if t.count("\n") > 1:
            body += [f"## {prof} profile", "", t, ""]
            fa = fairness(prof)
            if fa:
                body += ["### Chest fairness (sex / age / view)", "", fa, ""]
    if fed_txt:
        body += ["## Federated learning demo", "", fed_txt, ""]
        ps = json.loads(fed.read_text()).get("per_site")
        if ps:
            sites = list(next(iter(ps["auroc"].values())))
            rows = [
                "| Model | " + " | ".join(sites) + " | worst site | gap |",
                "|---" * (len(sites) + 3) + "|",
            ]
            for mname, row in ps["auroc"].items():
                sm = ps["summary"][mname]
                rows.append(
                    f"| {mname} | "
                    + " | ".join(f"{row[x]:.3f}" for x in sites)
                    + f" | {sm['worst_site_auroc']:.3f} | {sm['site_gap']:.3f} |"
                )
            body += ["### Per-site generalisation (AUROC on each hospital's scanner style)", ""]
            body += rows + [""]
    body += second_reader_section()
    (ROOT / "docs" / "METRICS.md").write_text("\n".join(body))
    readme = ROOT / "README.md"
    if readme.exists():
        txt = readme.read_text()
        block = "<!-- METRICS:START -->\n" + table("starter") + "\n<!-- METRICS:END -->"
        txt = re.sub(r"<!-- METRICS:START -->.*?<!-- METRICS:END -->", block, txt, flags=re.S)
        readme.write_text(txt)
    print("\n".join(body))


if __name__ == "__main__":
    main()
