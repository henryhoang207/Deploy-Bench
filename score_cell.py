"""Score one grid cell from its YAML config (used by run_grid.sh).

Dispatches on `runtime` and REFUSES, with exit 3 (the run_cell.sh convention for
an unhonourable cell), anything the scorers cannot actually do. Review B2: this
used to forward `runtime` to score.py as a bare label, so every ort/executorch
cell would have been PyTorch fp32 scores in disguise.
  pytorch     -> score.py       (fp32 only)
  ort | onnx  -> score_onnx.py  on cfg['onnx'] or exports/<model>_<precision>.onnx
  executorch  -> score_pte.py (ET_PYTHON) on deploybench.artifacts.et_artifact (DEV-7)
A non-clean condition needs `stress_manifest` (deploybench/stress.py output); without
it the label would describe clean audio, so the cell is refused.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml

UNHONOURABLE = 3
REPO = Path(__file__).resolve().parent
REQUIRED = ("model", "precision", "runtime", "dataset", "condition", "seed", "out")


class Refused(Exception):
    """The cell asks for something no scorer here can honour."""


def build_cmd(cfg: dict) -> list:
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise Refused(f"config lacks required keys {missing}")
    stress = cfg.get("stress_manifest")
    if cfg["condition"] != "clean" and not stress:
        raise Refused(f"condition {cfg['condition']!r} needs a stress_manifest (the "
                      "deploybench/stress.py output); without one the scorer would read "
                      "clean audio under a stress label")
    if stress and not Path(stress).is_file():
        raise Refused(f"stress_manifest {stress} does not exist; generate it with "
                      "deploybench/stress.py --dataset")
    common = ["--model", cfg["model"], "--dataset", cfg["dataset"],
              "--out", cfg["out"], "--precision", cfg["precision"],
              "--runtime", cfg["runtime"], "--condition", cfg["condition"],
              "--seed", str(cfg["seed"])] + (["--stress-manifest", str(stress)] if stress else [])
    rt = cfg["runtime"]
    if rt == "pytorch":
        if cfg["precision"] != "fp32":
            raise Refused(f"runtime pytorch scores fp32 only, not {cfg['precision']!r}")
        return [sys.executable, str(REPO / "score.py"),
                "--device", cfg.get("device", "cpu")] + common
    if rt in ("ort", "onnx"):
        onnx = Path(cfg.get("onnx") or REPO / "exports" / f"{cfg['model']}_{cfg['precision']}.onnx")
        if not onnx.exists():
            raise Refused(f"{onnx} does not exist; export/quantize it first")
        return [sys.executable, str(REPO / "score_onnx.py"), "--onnx", str(onnx)] + common
    if rt == "executorch":
        # DEV-7: score_pte.py runs in the executorch venv (numpy >= 2), named by
        # ET_PYTHON; the artifact is resolved by the gate/canonical rules, never guessed.
        import os
        from deploybench.artifacts import NotHonourable, et_artifact
        et_py = os.environ.get("ET_PYTHON")
        if not et_py:
            raise Refused("runtime executorch needs ET_PYTHON (the envs/et interpreter)")
        if stress:
            raise Refused("score_pte.py has no stress-manifest path; refusing a stressed executorch cell")
        try:
            pte = et_artifact(cfg["model"], cfg["precision"], REPO)
        except NotHonourable as e:
            raise Refused(str(e))
        return [et_py, str(REPO / "score_pte.py"), "--pte", str(pte),
                "--model", cfg["model"], "--dataset", cfg["dataset"], "--out", cfg["out"],
                "--precision", cfg["precision"], "--condition", cfg["condition"],
                "--seed", str(cfg["seed"])]
    raise Refused(f"unknown runtime {rt!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config))
    try:
        cmd = build_cmd(cfg)
    except Refused as e:
        print(f"REFUSED (exit {UNHONOURABLE}): {e}", file=sys.stderr)
        raise SystemExit(UNHONOURABLE)
    raise SystemExit(subprocess.call(cmd, cwd=REPO))


if __name__ == "__main__":
    main()
