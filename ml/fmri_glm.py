"""Task-fMRI activation mapping (C7 functional imaging, D-35).

    python -m ml.fmri_glm

OpenNeuro ds000114 (Gorgolewski 2013), subjects 01 and 02 (retest session), finger / foot / lips motor blocks, fMRIPrep output
already in MNI152NLin2009cAsym space. We fit a classical mass-univariate GLM ourselves (numpy):

* regressors: 15 s blocks convolved with the canonical double-gamma HRF (peak 6 s, undershoot 16 s,
  ratio 1/6), sampled at the slice-timing reference (middle of the TR);
* nuisance: intercept + discrete-cosine high-pass basis (128 s);
* 6 mm FWHM Gaussian smoothing inside the brain mask, mask eroded by one voxel (edge/motion voxels);
* OLS per brain voxel, t-maps for each condition > mean of the other two.

This is statistics, not a trained AI model. Validation without ground truth: the motor homunculus
predicts the activation peaks are ordered lips (inferior, lateral) -> finger -> foot (superior,
medial) in primary motor cortex, so the check uses each contrast's peak in the cerebrum (z > 0; the
cerebellum carries its own, separate map). We report whether the peaks obey that ordering.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any

import numpy as np

from ml.common import DATA_DIR, artifact_dir, write_json

RAW = DATA_DIR / "raw" / "fmri_motor"
CONDITIONS = ["Finger", "Foot", "Lips"]


def hrf(tr: float, length: float = 32.0, dt: float = 0.1) -> tuple[np.ndarray, float]:
    """Canonical double-gamma HRF sampled at `dt` (SPM parameters)."""
    t = np.arange(0, length, dt)

    def gpdf(x: np.ndarray, a: float) -> np.ndarray:
        return x ** (a - 1) * np.exp(-x) / math.gamma(a)

    h = gpdf(t, 6.0) - gpdf(t, 16.0) / 6.0
    return h / h.sum(), dt


def design(events: list[dict[str, Any]], n: int, tr: float) -> tuple[np.ndarray, list[str]]:
    h, dt = hrf(tr)
    t_hi = np.arange(0, n * tr, dt)
    cols, names = [], []
    for c in CONDITIONS:
        box = np.zeros_like(t_hi)
        for e in events:
            if e["trial_type"] == c:
                box[(t_hi >= e["onset"]) & (t_hi < e["onset"] + e["duration"])] = 1.0
        conv = np.convolve(box, h)[: len(t_hi)]
        frame_t = np.arange(n) * tr + tr / 2
        cols.append(np.interp(frame_t, t_hi, conv))
        names.append(c)
    # discrete cosine high-pass (period > 128 s removed) + intercept
    k = int(np.floor(2 * n * tr / 128.0))
    tt = np.arange(n)
    for j in range(1, k + 1):
        cols.append(np.cos(np.pi * j * (2 * tt + 1) / (2 * n)))
        names.append(f"drift{j}")
    cols.append(np.ones(n))
    names.append("intercept")
    X = np.stack(cols, 1)
    X[:, : len(CONDITIONS)] /= X[:, : len(CONDITIONS)].max(0)
    return X, names


def glm_t(Y: np.ndarray, X: np.ndarray, contrasts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Y: (T, V). OLS betas, residual variance, t for each contrast vector."""
    pinv = np.linalg.pinv(X)
    beta = pinv @ Y
    resid = Y - X @ beta
    dof = X.shape[0] - np.linalg.matrix_rank(X)
    sigma2 = (resid**2).sum(0) / dof
    xtx_inv = pinv @ pinv.T
    out = {}
    for name, c in contrasts.items():
        var = float(c @ xtx_inv @ c) * sigma2
        out[name] = (c @ beta) / np.sqrt(np.maximum(var, 1e-12))
    return out


def smooth_in_mask(data: np.ndarray, mask: np.ndarray, fwhm_mm: float, vox_mm: float) -> np.ndarray:
    """Normalised Gaussian smoothing restricted to the mask (no bleed of background into brain)."""
    from scipy.ndimage import gaussian_filter

    sigma = fwhm_mm / 2.3548 / vox_mm
    w = gaussian_filter(mask.astype(np.float32), sigma)
    out = np.empty_like(data)
    for t in range(data.shape[-1]):
        out[..., t] = gaussian_filter(data[..., t] * mask, sigma) / np.maximum(w, 1e-3)
    return out


def peak_mni(
    tmap: np.ndarray, affine: np.ndarray, z_min: float | None = None
) -> tuple[list[float], float]:
    t = tmap.copy()
    if z_min is not None:
        zs = affine[2, 2] * np.arange(t.shape[2]) + affine[2, 3]
        t[:, :, zs <= z_min] = -np.inf
    ijk = np.unravel_index(int(np.nanargmax(t)), t.shape)
    xyz = affine @ np.array([*ijk, 1.0])
    return [round(float(v), 1) for v in xyz[:3]], float(tmap[ijk])


SUBJECTS = ["sub-01", "sub-02"]
MOTION = ["X", "Y", "Z", "RotX", "RotY", "RotZ"]


def _files(sub: str) -> tuple[Any, Any, Any]:
    stem = f"{sub}_ses-retest_task-fingerfootlips_bold"
    return (
        RAW / f"{stem}_space-MNI152NLin2009cAsym_preproc.nii.gz",
        RAW / f"{stem}_space-MNI152NLin2009cAsym_brainmask.nii.gz",
        RAW / f"{stem}_confounds.tsv",
    )


def _motion(path: Any, n: int) -> np.ndarray | None:
    """6 rigid-body motion parameters from the fMRIPrep confounds (z-scored), if the file exists."""
    if not path.exists():
        return None
    import pandas as pd

    df = pd.read_csv(path, sep="\t", na_values="n/a").fillna(0.0)
    m = df[MOTION].to_numpy(np.float64)[:n]
    return (m - m.mean(0)) / np.maximum(m.std(0), 1e-9)


def analyse_subject(sub: str, events: list[dict[str, Any]], tr: float) -> dict[str, Any]:
    import nibabel as nib
    from scipy import stats
    from scipy.ndimage import binary_erosion

    bold, mask_f, conf = _files(sub)
    if not bold.exists():
        raise FileNotFoundError(
            f"{bold.name} missing (python -m ml.data.download --only fmri_motor)"
        )
    img = nib.load(str(bold))
    data = np.asarray(img.dataobj, dtype=np.float32)
    mask = binary_erosion(np.asarray(nib.load(str(mask_f)).dataobj) > 0)
    data = smooth_in_mask(data, mask, 6.0, float(img.header.get_zooms()[0]))
    n = data.shape[-1]
    X, names = design(events, n, tr)
    mot = _motion(conf, n)
    if mot is not None:  # nuisance columns go before the intercept; condition columns stay first
        X = np.concatenate([X[:, :-1], mot, X[:, -1:]], 1)
    Y = data[mask].T  # (T, V)
    Y = 100 * (Y / np.maximum(Y.mean(0, keepdims=True), 1e-6) - 1)  # percent signal change
    contrasts = {}
    for i, c in enumerate(CONDITIONS):
        v = np.zeros(X.shape[1])
        v[i] = 1.0
        for j in range(len(CONDITIONS)):
            if j != i:
                v[j] = -0.5
        contrasts[f"{c} > others"] = v
    tv = glm_t(Y, X, contrasts)
    dof = int(X.shape[0] - np.linalg.matrix_rank(X))
    # Bonferroni over brain voxels (conservative; AR(1) autocorrelation is not modelled)
    t_thr = float(stats.t.isf(0.05 / int(mask.sum()), dof))
    maps, peaks = {}, {}
    for name, t in tv.items():
        vol = np.zeros(mask.shape, np.float32)
        vol[mask] = t
        maps[name] = vol
        xyz, tmax = peak_mni(vol, img.affine)
        cxyz, ctmax = peak_mni(vol, img.affine, z_min=0.0)
        peaks[name] = {
            "mni": xyz,
            "t": round(tmax, 2),
            "cerebral_mni": cxyz,
            "cerebral_t": round(ctmax, 2),
            "cerebral_significant": bool(ctmax > t_thr),
            "n_voxels_sig": int((t > t_thr).sum()),
        }
    zs = {c: peaks[f"{c} > others"]["cerebral_mni"][2] for c in CONDITIONS}
    xs = {c: abs(peaks[f"{c} > others"]["cerebral_mni"][0]) for c in CONDITIONS}
    somatotopy = {
        "lips_below_finger": bool(zs["Lips"] < zs["Finger"]),
        "finger_below_foot": bool(zs["Finger"] < zs["Foot"]),
        "foot_most_medial": bool(xs["Foot"] < min(xs["Finger"], xs["Lips"])),
    }
    return {
        "img_affine": img.affine,
        "mean": data.mean(-1),
        "mask": mask,
        "maps": maps,
        "summary": {
            "volumes": n,
            "dof": dof,
            "motion_regressors": mot is not None,
            "t_threshold_bonferroni": round(t_thr, 2),
            "peaks": peaks,
            "somatotopy": somatotopy,
            "somatotopy_pass": all(somatotopy.values()),
            "all_cerebral_peaks_significant": all(
                p["cerebral_significant"] for p in peaks.values()
            ),
        },
    }


def run(profile: str = "starter") -> dict[str, Any]:
    import csv

    t0 = time.time()
    tr = float(json.loads((RAW / "task-fingerfootlips_bold.json").read_text())["RepetitionTime"])
    with (RAW / "task-fingerfootlips_events.tsv").open() as fh:
        events = [
            {
                "onset": float(r["onset"]),
                "duration": float(r["duration"]),
                "trial_type": r["trial_type"],
            }
            for r in csv.DictReader(fh, delimiter="\t")
        ]
    out = artifact_dir("fmri_glm", profile)
    subjects = {}
    for sub in SUBJECTS:
        try:
            r = analyse_subject(sub, events, tr)
        except FileNotFoundError as exc:
            print(f"[fmri] {sub}: skipped ({exc})")
            continue
        maps, thr = r["maps"], r["summary"]["t_threshold_bonferroni"]
        np.savez_compressed(
            out / f"maps_{sub}.npz",
            mean=r["mean"].astype(np.float16),
            mask=r["mask"],
            affine=r["img_affine"],
            t_thr=np.float32(thr),
            **{k.split(" ")[0].lower(): v.astype(np.float16) for k, v in maps.items()},
        )
        _plot(r["mean"], maps, r["summary"]["peaks"], r["img_affine"], thr, out, sub)
        subjects[sub] = r["summary"]
    if not subjects:
        raise FileNotFoundError("no fMRI subjects found")
    metrics = {
        "name": "fmri_glm",
        "profile": profile,
        "task": "fmri_activation",
        "dataset": "OpenNeuro ds000114 retest session, finger/foot/lips (fMRIPrep, MNI152NLin2009cAsym)",
        "method": "mass-univariate OLS GLM, double-gamma HRF, DCT high-pass 128 s, 6 mm FWHM "
        "smoothing, eroded brain mask, 6 motion regressors where available (ours, numpy)",
        "tr_s": tr,
        "subjects": subjects,
        "somatotopy_pass_all": all(s["somatotopy_pass"] for s in subjects.values()),
        "caveat": "Two subjects, single session; sub-01 has no motion regressors in this release; "
        "AR(1) not modelled, so t is optimistic - a Bonferroni threshold is used to compensate.",
        "seconds": round(time.time() - t0, 1),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_json(out / "metrics.json", metrics)
    for sub, s in subjects.items():
        print(sub, "thr", s["t_threshold_bonferroni"], s["somatotopy"])
        for k, p in s["peaks"].items():
            print(
                "  ",
                k,
                p["cerebral_mni"],
                p["cerebral_t"],
                "sig" if p["cerebral_significant"] else "n.s.",
            )
    return metrics


def _plot(
    mean: np.ndarray,
    maps: dict[str, np.ndarray],
    peaks: dict[str, Any],
    affine: np.ndarray,
    thr: float,
    out: Any,
    sub: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (out / "plots").mkdir(exist_ok=True)
    inv = np.linalg.inv(affine)
    fig, axs = plt.subplots(1, len(maps), figsize=(3.2 * len(maps), 3.4))
    for ax, (name, t) in zip(axs, maps.items(), strict=True):
        xyz = peaks[name]["cerebral_mni"]
        i, j, k = (inv @ np.array([*xyz, 1.0]))[:3].round().astype(int)
        ax.imshow(np.rot90(mean[:, :, k]), cmap="gray")
        ax.imshow(np.rot90(np.ma.masked_less(t[:, :, k], thr)), cmap="autumn", alpha=0.85)
        ax.set_title(
            f"{name}\nz = {xyz[2]:.0f} mm, t = {peaks[name]['cerebral_t']:.1f}", fontsize=8
        )
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "plots" / f"activation_peaks_{sub}.png", dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="starter")
    run(ap.parse_args(argv).profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
