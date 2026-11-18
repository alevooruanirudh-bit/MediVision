"""Train, evaluate and export EVERY starter model in dependency order (make train-starter).

    python -m ml.train_all --profile starter [--skip-existing] [--only brain_cls router]

Order matters: the chest DenseNet is trained before the report generator, which reuses its trunk.
A failing model is logged and skipped so one missing dataset never blocks the rest.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback

from ml.common import ARTIFACTS, write_json

ORDER = [
    "brain_cls",
    "brain_seg",
    "fracture_cls",
    "fracture_det",
    "lung_unet",
    "chest_densenet121",
    "chest_efficientnet_b2",
    "chest_convnext_tiny",
    "head_ct_cls",
    "breast_us",
    "mammo_cls",
    "retina_dr",
    "nodule3d",
    "router",
    "denoiser",
    "sr_gan",
    "radiogenomics",
    "petct_seg",
    "lung_nodule_det",
    "fmri_glm",
    "report_gen",
    "copilot_intent",
]
EVAL_AFTER = {"chest_convnext_tiny": "chest_ensemble"}


def main(argv: list[str] | None = None) -> int:
    from ml.copilot.train_intent import train as train_intent
    from ml.evaluate import evaluate_model, write_card
    from ml.export_onnx import EXPORTABLE, export
    from ml.train import train

    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="starter")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args(argv)
    status: dict[str, str] = {}
    for name in args.only or ORDER:
        t0 = time.time()
        try:
            done = (ARTIFACTS / name / args.profile / "model.pt").exists()
            standalone = {
                "radiogenomics": "ml.radiogenomics",
                "petct_seg": "ml.petct_fusion",
                "lung_nodule_det": "ml.lung_detect",
                "fmri_glm": "ml.fmri_glm",
            }
            if name in standalone:
                import importlib

                importlib.import_module(standalone[name]).main(["--profile", args.profile])
                status[name] = f"ok ({time.time() - t0:.0f}s)"
                print(f"[train_all] {name}: {status[name]}", flush=True)
                continue
            if name == "copilot_intent":
                train_intent(args.profile)
            elif not (args.skip_existing and done):
                train(name, args.profile)
            if name.startswith("chest_") and name != "chest_convnext_tiny":
                pass  # members are evaluated jointly as the ensemble
            elif name not in ("copilot_intent", "report_gen", "sr_gan"):
                target = EVAL_AFTER.get(name, name)
                evaluate_model(target, args.profile)
                write_card(target, args.profile)
            if name in EXPORTABLE:
                export(name, args.profile)
            if name == "chest_convnext_tiny":
                for m in ("chest_densenet121", "chest_efficientnet_b2"):
                    if (ARTIFACTS / m / args.profile / "model.pt").exists():
                        export(m, args.profile)
            status[name] = f"ok ({time.time() - t0:.0f}s)"
        except Exception as exc:  # keep going
            traceback.print_exc()
            status[name] = f"FAILED: {exc.__class__.__name__}: {exc}"
        print(f"[train_all] {name}: {status[name]}", flush=True)
    status_file = ARTIFACTS / f"train_all_{args.profile}.json"
    merged = json.loads(status_file.read_text()) if status_file.exists() else {}
    merged.update(status)
    write_json(status_file, merged)
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
