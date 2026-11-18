"""Shared helpers: paths, seeding, device selection, config loading."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("MEDIVISION_DATA_DIR", ROOT / "data"))
PROCESSED = DATA_DIR / "processed"
ARTIFACTS = Path(os.environ.get("MEDIVISION_ARTIFACTS_DIR", ROOT / "artifacts"))
CONFIGS = ROOT / "ml" / "configs"

NIH_LABELS: list[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]
BRAIN_CLASSES: list[str] = ["glioma", "meningioma", "notumor", "pituitary"]
ROUTER_CLASSES: list[str] = [
    "chest_xray",
    "bone_xray",
    "brain_mri",
    "head_ct",
    "breast_us",
    "fundus",
    "other_medical",
    "not_medical",
]
# Previous 5-class router (artifacts trained before D-21 carry their own label list).
ROUTER_CLASSES_V1: list[str] = [
    "chest_xray",
    "bone_xray",
    "brain_mri",
    "other_medical",
    "not_medical",
]
RETINA_GRADES: list[str] = [
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR",
]


def pretrained_backbone() -> bool:
    return os.environ.get("PRETRAINED_BACKBONE", "true").lower() in ("1", "true", "yes")


def seed_everything(seed: int = 42) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(prefer: str = "auto") -> str:
    import torch

    if prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and os.environ.get("MEDIVISION_NO_MPS") != "1":
        return "mps"
    return "cpu"


def load_config(name: str, profile: str = "starter") -> dict[str, Any]:
    """Load ml/configs/<name>.yaml and merge `common` with the requested profile."""
    import yaml

    raw = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())
    cfg: dict[str, Any] = dict(raw.get("common", {}))
    for k, v in raw.get(profile, {}).items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k] = {**cfg[k], **v}
        else:
            cfg[k] = v
    cfg["name"] = name
    cfg["profile"] = profile
    return cfg


def artifact_dir(model: str, profile: str) -> Path:
    d = ARTIFACTS / model / profile
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))
