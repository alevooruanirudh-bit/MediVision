"""Model registry: loads our trained weights, preferring `full` over `starter`, ONNX over PyTorch.

PyTorch models are loaded lazily (Grad-CAM, detection, report generator). Missing models never crash
the app: the pipeline degrades gracefully and the UI shows what is active.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.config import get_settings

log = logging.getLogger("medivision.registry")

MODELS = [
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
    "report_gen",
    "copilot_intent",
    "pneumonia_det",
    "head_ct_cls",
    "breast_us",
    "retina_dr",
    "nodule3d",
    "sr_gan",
    "mammo_cls",
    "lung_nodule_det",
    "petct_seg",
]
CHEST_MEMBERS = MODELS[:3]


def _read_json(p: Path) -> dict[str, Any]:
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


@dataclass
class ModelHandle:
    name: str
    profile: str
    path: Path
    cfg: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    session: Any = None
    head: dict[str, np.ndarray] | None = None
    _torch: Any = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    _version: str | None = None

    @property
    def version(self) -> str:
        """Content hash of the weights file (12 hex) — pins exactly which model produced a result."""
        if self._version is None:
            f = next(
                (
                    self.path / n
                    for n in ("model.onnx", "model.pt", "model.joblib")
                    if (self.path / n).exists()
                ),
                None,
            )
            h = hashlib.sha256()
            if f is not None:
                with f.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
            self._version = h.hexdigest()[:12]
        return self._version

    def version_info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "profile": self.profile,
            "version": self.version,
            "trained_at": self.metrics.get("generated_at"),
            "backend": self.backend,
        }

    @property
    def img_size(self) -> int:
        return int(self.cfg.get("img_size", 128))

    @property
    def backend(self) -> str:
        return "onnx" if self.session is not None else "torch"

    def torch_model(self) -> Any:
        if self._torch is None:
            import torch

            from ml.models import build_model

            ck = torch.load(self.path / "model.pt", map_location="cpu", weights_only=False)
            m = build_model(ck["config"], pretrained=False)
            m.load_state_dict(ck["state_dict"])
            self._torch = m.eval()
        return self._torch

    def run(self, x: np.ndarray) -> list[np.ndarray]:
        """Forward pass. ONNX when available, else PyTorch (same outputs)."""
        if self.session is not None:
            return self.session.run(None, {"input": x.astype(np.float32)})
        import torch

        m = self.torch_model()
        with self.lock, torch.no_grad():
            t = torch.from_numpy(x.astype(np.float32))
            if hasattr(m, "forward_features") and hasattr(m, "head"):
                f = m.forward_features(t)
                return [m.head(f).numpy(), f.numpy()]
            return [m(t).numpy()]


class Registry:
    def __init__(self) -> None:
        self.handles: dict[str, ModelHandle] = {}
        self.extra: dict[str, Any] = {}
        self.loaded_at: float | None = None
        self._lock = threading.Lock()

    @property
    def root(self) -> Path:
        return get_settings().artifacts_dir

    def resolve_profile(self, name: str) -> str | None:
        want = get_settings().model_profile
        order = ["full", "starter"] if want == "auto" else [want]
        for prof in order:
            d = self.root / name / prof
            if (
                (d / "model.onnx").exists()
                or (d / "model.pt").exists()
                or (d / "model.joblib").exists()
            ):
                return prof
        return None

    def load(self) -> None:
        with self._lock:
            self.handles.clear()
            use_onnx = get_settings().inference_backend == "onnx"
            for name in MODELS:
                prof = self.resolve_profile(name)
                if prof is None:
                    continue
                d = self.root / name / prof
                h = ModelHandle(
                    name, prof, d, _read_json(d / "config.json"), _read_json(d / "metrics.json")
                )
                if (
                    use_onnx
                    and (d / "model.onnx").exists()
                    and name not in ("fracture_det", "pneumonia_det")
                ):
                    try:
                        import onnxruntime as ort

                        so = ort.SessionOptions()
                        so.intra_op_num_threads = 4
                        h.session = ort.InferenceSession(
                            str(d / "model.onnx"), so, providers=["CPUExecutionProvider"]
                        )
                    except Exception as exc:  # pragma: no cover
                        log.warning("onnx load failed for %s: %s", name, exc)
                if (d / "head.npz").exists():
                    z = np.load(d / "head.npz")
                    h.head = {k: z[k] for k in z.files}
                if name == "copilot_intent":
                    import joblib

                    self.extra["intent_model"] = joblib.load(d / "model.joblib")
                self.handles[name] = h
            for prof in ("full", "starter"):
                cal = self.root / "chest_ensemble" / prof / "calibration.json"
                if cal.exists() and (get_settings().model_profile in ("auto", prof)):
                    self.extra["chest_calibration"] = _read_json(cal)
                    self.extra["chest_profile"] = prof
                    self.extra["chest_metrics"] = _read_json(cal.parent / "metrics.json")
                    break
            self.loaded_at = time.time()
            log.info(
                "registry loaded: %s", {k: (v.profile, v.backend) for k, v in self.handles.items()}
            )

    def get(self, name: str) -> ModelHandle | None:
        if self.loaded_at is None:
            self.load()
        return self.handles.get(name)

    def chest_members(self) -> list[ModelHandle]:
        return [h for m in CHEST_MEMBERS if (h := self.get(m)) is not None]

    def status(self) -> dict[str, Any]:
        if self.loaded_at is None:
            self.load()
        out = {}
        for name in MODELS:
            h = self.handles.get(name)
            out[name] = (
                {
                    "loaded": True,
                    "profile": h.profile,
                    "backend": h.backend,
                    "img_size": h.cfg.get("img_size"),
                    "version": h.version,
                    "trained_at": h.metrics.get("generated_at"),
                }
                if h
                else {"loaded": False}
            )
        return out


registry = Registry()
