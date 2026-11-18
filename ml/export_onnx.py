"""Export trained models to ONNX for fast CPU inference with onnxruntime.

    python -m ml.export_onnx --all --profile starter

Classifiers export `(logits, features)` so the backend can run MC-dropout on the head in numpy
(head weights are saved to head.npz) without re-running the backbone. The report generator and
Grad-CAM stay on PyTorch. Each export is verified against PyTorch outputs.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ml.common import ARTIFACTS, write_json
from ml.evaluate import load_trained

EXPORTABLE = [
    "chest_densenet121",
    "chest_efficientnet_b2",
    "chest_convnext_tiny",
    "lung_unet",
    "fracture_cls",
    "fracture_det",
    "brain_cls",
    "brain_seg",
    "router",
    "denoiser",
    "head_ct_cls",
    "breast_us",
    "mammo_cls",
    "retina_dr",
    "nodule3d",
    "sr_gan",
]


class WithFeatures(nn.Module):
    def __init__(self, m: nn.Module) -> None:
        super().__init__()
        self.m = m

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.m.forward_features(x)  # type: ignore[operator]
        return self.m.head(f), f  # type: ignore[operator]


def _export(
    module: nn.Module,
    x: torch.Tensor,
    path: str,
    inputs: list[str],
    outputs: list[str],
    dynamic: dict[str, Any] | None,
) -> None:
    try:
        torch.onnx.export(
            module,
            (x,),
            path,
            input_names=inputs,
            output_names=outputs,
            dynamic_axes=dynamic,
            opset_version=18,
            dynamo=False,
        )
    except TypeError:
        torch.onnx.export(
            module,
            (x,),
            path,
            input_names=inputs,
            output_names=outputs,
            dynamic_axes=dynamic,
            opset_version=18,
        )


def export(name: str, profile: str) -> dict[str, Any]:
    import onnxruntime as ort

    model, cfg = load_trained(name, profile, "cpu")
    d = ARTIFACTS / name / profile
    size = cfg["img_size"]
    task = cfg["task"]
    path = str(d / "model.onnx")
    info: dict[str, Any] = {"model": name, "profile": profile, "img_size": size}
    if task == "detection":
        x = torch.rand(3, size, size)
        try:
            torch.onnx.export(
                model,
                ([x],),
                path,
                input_names=["image"],
                output_names=["boxes", "labels", "scores"],
                opset_version=18,
                dynamo=False,
            )
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            sess.run(None, {"image": x.numpy()})
            info["onnx"] = True
        except Exception as exc:  # torchvision detection export is fragile across versions
            info["onnx"] = False
            info["onnx_error"] = str(exc)[:200]
        write_json(d / "export.json", info)
        return info
    arch = cfg["model"]["arch"]
    chans = 3 if arch in ("classifier", "smallcnn") else 1
    x = torch.randn(1, chans, size, size)
    if arch == "cnn3d":
        x = torch.randn(1, 1, size, size, size)
    if arch == "srgan":
        x = torch.rand(1, 1, size, size)
    if arch == "srgan":
        _export(
            model,
            x,
            path,
            ["input"],
            ["output"],
            {"input": {0: "batch", 2: "h", 3: "w"}, "output": {0: "batch", 2: "oh", 3: "ow"}},
        )
    elif arch == "classifier":
        _export(
            WithFeatures(model),
            x,
            path,
            ["input"],
            ["logits", "features"],
            {"input": {0: "batch"}, "logits": {0: "batch"}, "features": {0: "batch"}},
        )
        lin = [m for m in model.head if isinstance(m, nn.Linear)]  # type: ignore[attr-defined]
        np.savez(
            d / "head.npz",
            w1=lin[0].weight.detach().numpy(),
            b1=lin[0].bias.detach().numpy(),
            w2=lin[1].weight.detach().numpy(),
            b2=lin[1].bias.detach().numpy(),
            dropout=float(cfg["model"].get("dropout", 0.3)),
        )
    else:
        _export(
            model,
            x,
            path,
            ["input"],
            ["output"],
            (
                {"input": {0: "batch", 2: "h", 3: "w"}, "output": {0: "batch", 2: "h", 3: "w"}}
                if task in ("denoise", "segmentation")
                else {"input": {0: "batch"}, "output": {0: "batch"}}
            ),
        )
    model.eval()
    with torch.no_grad():
        ref = model(x).numpy()
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input": x.numpy()})[0]
    t0 = time.perf_counter()
    for _ in range(5):
        sess.run(None, {"input": x.numpy()})
    info.update(
        {
            "onnx": True,
            "max_abs_diff": float(np.abs(got - ref).max()),
            "cpu_ms_per_image": round((time.perf_counter() - t0) / 5 * 1000, 1),
        }
    )
    write_json(d / "export.json", info)
    print(
        f"[export] {name}/{profile}: diff {info['max_abs_diff']:.2e}, "
        f"{info['cpu_ms_per_image']} ms/img"
    )
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--profile", default="starter")
    args = ap.parse_args(argv)
    names = EXPORTABLE if args.all else [args.model]
    report = {}
    for n in names:
        if not (ARTIFACTS / n / args.profile / "model.pt").exists():
            print(f"[export] {n}/{args.profile}: not trained, skipped")
            continue
        report[n] = export(n, args.profile)
    print(json.dumps({k: v.get("onnx") for k, v in report.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
