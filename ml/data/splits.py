"""Split helpers. Patient-grouped wherever identifiers exist, so no patient spans two splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def group_split(
    groups: np.ndarray, val_frac: float = 0.1, seed: int = 42, test_frac: float = 0.0
) -> np.ndarray:
    """Assign train/val/test per *group* (patient). Returns one split name per row."""
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_test = int(round(len(uniq) * test_frac))
    n_val = int(round(len(uniq) * val_frac))
    test_g = set(uniq[:n_test].tolist())
    val_g = set(uniq[n_test : n_test + n_val].tolist())
    return np.array(
        ["test" if g in test_g else "val" if g in val_g else "train" for g in groups.tolist()]
    )


def stratified_split(
    labels: np.ndarray, fracs: tuple[float, float, float] = (0.7, 0.15, 0.15), seed: int = 42
) -> np.ndarray:
    """Per-class random train/val/test split (used only where no patient IDs are published)."""
    rng = np.random.default_rng(seed)
    out = np.empty(len(labels), dtype=object)
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_tr = int(round(len(idx) * fracs[0]))
        n_va = int(round(len(idx) * fracs[1]))
        out[idx[:n_tr]] = "train"
        out[idx[n_tr : n_tr + n_va]] = "val"
        out[idx[n_tr + n_va :]] = "test" if fracs[2] > 0 else "val"
    return out.astype(str)


def save_split(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def leakage(df: pd.DataFrame, group_col: str = "patient_id") -> dict[str, int]:
    """Number of groups shared between each pair of splits (should be all zeros)."""
    sets = {s: set(g[group_col]) for s, g in df.groupby("split")}
    out = {}
    names = sorted(sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            out[f"{a}-{b}"] = len(sets[a] & sets[b])
    return out
