"""Push a generated notebook to Kaggle, wait for it on the free GPU, pull the trained weights back.

    python -m ml.kaggle.run --model brain_cls            # full profile
    python -m ml.kaggle.run --model brain_cls --smoke    # quick starter-profile verification run

Requires a Kaggle token (~/.kaggle/kaggle.json or ~/.kaggle/access_token). Without one it prints
step-by-step manual instructions instead (see docs/TRAINING.md).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ml.common import ARTIFACTS
from ml.data.download import has_kaggle_token
from ml.kaggle.generate import PLANS, generate

MANUAL = """
No Kaggle token found, so the notebook was generated but not pushed.
  1. Create a free account at https://www.kaggle.com and verify your phone number (enables GPU).
  2. Account -> Settings -> API -> "Create New Token"; save kaggle.json to ~/.kaggle/ (chmod 600).
  3. For pneumonia_det accept the rules at
     https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/rules
  4. Re-run:  make kaggle-train MODEL={model}
Or upload {folder}/notebook.ipynb manually (File -> Import), add the datasets listed in
kernel-metadata.json, switch the accelerator to GPU, "Save & Run All", then download
/kaggle/working/output into artifacts/.
"""


def _kaggle(*args: str, check: bool = True) -> str:
    exe = shutil.which("kaggle") or str(Path(sys.executable).with_name("kaggle"))
    res = subprocess.run([exe, *args], capture_output=True, text=True)
    if check and res.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)} failed: {res.stderr.strip() or res.stdout}")
    return res.stdout


def username() -> str:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    u = api.get_config_value("username")
    if not u:
        raise RuntimeError("could not determine Kaggle username")
    return str(u)


def run(model: str, smoke: bool = False, poll: int = 60, timeout_h: float = 12.0) -> int:
    if model not in PLANS:
        member_of = [k for k, v in PLANS.items() if model in v.get("members", [])]
        if not member_of:
            print(f"unknown model {model}; choose from {sorted(PLANS)}")
            return 2
        print(f"{model} is trained as part of {member_of[0]}")
        model = member_of[0]
    folder = generate([model], smoke)[0]
    if not has_kaggle_token():
        print(MANUAL.format(model=model, folder=folder))
        return 0
    meta_path = folder / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["id"] = meta["id"].replace("USERNAME", username())
    meta_path.write_text(json.dumps(meta, indent=2))
    kid = meta["id"]
    print(f"[kaggle] pushing {kid} (GPU, private)…")
    print(_kaggle("kernels", "push", "-p", str(folder)).strip())
    t0 = time.time()
    status = ""
    while time.time() - t0 < timeout_h * 3600:
        time.sleep(poll)
        out = _kaggle("kernels", "status", kid, check=False).lower()
        status = next(
            (s for s in ("complete", "error", "cancel", "running", "queued") if s in out),
            out.strip(),
        )
        print(f"[kaggle] {kid}: {status} ({(time.time() - t0) / 60:.0f} min)", flush=True)
        if status in ("complete", "error", "cancel"):
            break
    out_dir = folder / "output"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir()
    _kaggle("kernels", "output", kid, "-p", str(out_dir), check=False)
    if status != "complete":
        print(f"[kaggle] run ended with status '{status}'. Logs are in {out_dir}.")
        return 1
    profile = "starter" if smoke else "full"
    pulled = 0
    for src in out_dir.rglob(f"*/{profile}"):
        if src.is_dir() and src.parent.name in (
            *PLANS,
            *[m for v in PLANS.values() for m in v.get("members", [])],
        ):
            dest = ARTIFACTS / src.parent.name / (profile if not smoke else "kaggle_smoke")
            shutil.copytree(src, dest, dirs_exist_ok=True)
            pulled += 1
            print(f"[kaggle] pulled {src.parent.name} -> {dest}")
    if not smoke and pulled:
        subprocess.run(
            [sys.executable, "-m", "ml.export_onnx", "--all", "--profile", "full"], check=False
        )
    print(f"[kaggle] done: {pulled} model folder(s). The backend now prefers 'full' weights.")
    return 0 if pulled else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--poll", type=int, default=60)
    a = ap.parse_args()
    raise SystemExit(run(a.model, a.smoke, a.poll))
