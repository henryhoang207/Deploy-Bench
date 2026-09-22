"""Score one grid cell from its YAML config (used by run_grid.sh).

Dispatches on `runtime` and REFUSES, with exit 3 (the run_cell.sh convention for
an unhonourable cell), anything the scorers cannot actually do. Review B2: this
used to forward `runtime` to score.py as a bare label, so every ort/executorch
cell would have been PyTorch fp32 scores in disguise.
  pytorch     -> score.py       (fp32 only)
  ort | onnx  -> score_onnx.py  on cfg['onnx'] or exports/<model>_<precision>.onnx
  executorch  -> refused: no .pte scorer exists (OD-4)
Non-clean conditions are refused as well: stressed audio is not wired into the
scorers, so forwarding the label would score clean audio as stressed.
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
    if cfg["condition"] != "clean":
        raise Refused(f"condition {cfg['condition']!r}: stressed audio is not wired "
                      "into the scorers (deploybench/stress.py output is unused)")
    common = ["--model", cfg["model"], "--dataset", cfg["dataset"],
              "--out", cfg["out"], "--precision", cfg["precision"],
              "--runtime", cfg["runtime"], "--condition", cfg["condition"],
              "--seed", str(cfg["seed"])]
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
        raise Refused("runtime executorch has no .pte scorer (OD-4)")
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
