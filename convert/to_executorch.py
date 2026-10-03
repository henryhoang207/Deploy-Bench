"""PyTorch -> ExecuTorch (.pte) via the XNNPACK partitioner (D48 / DEVIATIONS.md DEV-7).

Runs in the `envs/et` venv on sol1 (executorch 1.5.1 needs numpy >= 2; db310 is
fairseq-pinned). Every resolved version goes into the conversion log.

    python convert/to_executorch.py --model tssdnet --checkpoint C --precision fp32 --out exports/et
    python convert/to_executorch.py --model tssdnet --checkpoint C --precision int8 \
        --calib-manifest configs/conversion_sensitivity/calib_seed1.txt --calib-seed 1 --out exports/et

Precisions (DEV-7):
  fp32  export_module() as is (D13), lowered with to_edge_transform_and_lower + XnnpackPartitioner.
  fp16  a half copy of the module; fp32 input cast to fp16 and the score cast back to fp32
        INSIDE the graph, so the app feeds and reads fp32 for every precision.
  int8  PT2E static quantisation, XNNPACKQuantizer with its default symmetric config,
        per-channel weights, NO fp32 exclusions, calibrated on one D39 calibration
        manifest (500 ASVspoof19 LA dev clips) through score.py's exact preprocessing.

"Converts" = the executorch runtime loads the .pte and returns one finite score for a
zero waveform (the to_onnx.ort_smoke rule). A failure is logged as the RESULT; nothing
is retried here. The sidecar records the .pte sha256, which score_pte.py checks.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _versions() -> dict:
    import importlib.metadata as md
    out = {"python": sys.version.split()[0]}
    for pkg in ("torch", "executorch", "torchao", "numpy"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pte_smoke(path: Path, input_samples: int) -> float:
    """Raise unless the ExecuTorch runtime loads the program and returns a finite score."""
    import math
    import torch
    from executorch.runtime import Runtime
    method = Runtime.get().load_program(str(path)).load_method("forward")
    out = method.execute([torch.zeros(1, input_samples)])[0]
    v = float(out.reshape(-1)[0])
    if not math.isfinite(v):
        raise ValueError(f"runtime loads {path.name} but its output is not finite")
    return v


def calibration_batches(manifest: Path, input_samples: int):
    """Yield (1, n) fp32 tensors for the manifest's ASVspoof19 LA dev ids, preprocessed
    exactly as score.py does (load_wav + fix_length)."""
    import torch
    sys.path.insert(0, str(REPO))
    from deploybench.data import protocol_iter
    from deploybench.manifests import read_ids
    from score import fix_length, load_wav
    ids = read_ids(manifest)
    index = {u: p for u, p, _ in protocol_iter("asvspoof19_dev")}
    miss = [u for u in ids if u not in index]
    if miss:
        raise SystemExit(f"{len(miss)} calibration ids not in asvspoof19_dev (first {miss[:3]})")
    for u in ids:
        yield torch.from_numpy(fix_length(load_wav(Path(index[u])), input_samples)).unsqueeze(0)


class _Half:
    """Factory for the fp16 wrapper (torch imported lazily)."""
    @staticmethod
    def wrap(net):
        import torch

        class Half(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner.half()

            def forward(self, x):
                return self.inner(x.half()).float()
        return Half(net).eval()


def _quantize_int8(net, dummy, calib, log):
    import torch
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer, get_symmetric_quantization_config)
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
    q = XNNPACKQuantizer()
    q.set_global(get_symmetric_quantization_config(is_per_channel=True))
    m = prepare_pt2e(torch.export.export(net, dummy).module(), q)
    n = 0
    with torch.no_grad():
        for x in calib:
            m(x)
            n += 1
    log["calib_clips"] = n
    return convert_pt2e(m)


def _lower(net, dummy):
    import torch
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower
    return to_edge_transform_and_lower(
        torch.export.export(net, dummy), partitioner=[XnnpackPartitioner()]).to_executorch()


def export(model_name, precision, out_dir: Path, checkpoint=None,
           calib_manifest=None, calib_seed=None):
    import torch
    sys.path.insert(0, str(REPO))
    from models.base import SPECS, load_model
    if precision == "int8" and (calib_manifest is None or calib_seed is None):
        raise SystemExit("int8 needs --calib-manifest and --calib-seed (DEV-7)")
    if precision != "int8" and (calib_manifest or calib_seed is not None):
        raise SystemExit("--calib-* only apply to int8")
    spec = SPECS[model_name]
    stem = f"{model_name}_{precision}" + (f"_seed{calib_seed}" if precision == "int8" else "")
    out = out_dir / f"{stem}.pte"
    side = out_dir / f"conversion_log_{stem}_executorch.json"
    if out.exists() or side.exists():
        raise SystemExit(f"REFUSING to overwrite {out} / {side}")
    wrap = load_model(model_name, checkpoint, "cpu")
    if not hasattr(wrap, "export_module"):
        raise SystemExit(f"{model_name}: wrapper lacks export_module() (D13)")
    net = wrap.export_module().eval()
    dummy = (torch.zeros(1, spec.input_samples),)
    log = {"model": model_name, "runtime": "executorch", "precision": precision,
           "backend": "xnnpack", "input_samples": spec.input_samples,
           "versions": _versions(), "converts": None, "error": None,
           "calib_manifest": str(calib_manifest) if calib_manifest else None,
           "calib_seed": calib_seed, "started": time.time()}
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        if precision == "fp16":
            net = _Half.wrap(net)
        elif precision == "int8":
            from deploybench.manifests import file_sha256
            log["calib_manifest_sha256"] = file_sha256(Path(calib_manifest))
            net = _quantize_int8(net, dummy,
                                 calibration_batches(Path(calib_manifest), spec.input_samples), log)
        prog = _lower(net, dummy)
        with open(out, "wb") as fh:
            prog.write_to_file(fh)
        log["smoke_score"] = pte_smoke(out, spec.input_samples)
        log["converts"] = True
        log["bytes"] = out.stat().st_size
        log["sha256"] = _sha256(out)
    except Exception as e:                                      # noqa: BLE001
        log["converts"] = False
        log["error"] = f"{type(e).__name__}: {e}"               # a RESULT, not a crash
    log["wall_s"] = round(time.time() - log.pop("started"), 1)
    side.write_text(json.dumps(log, indent=2))
    print(json.dumps({k: log.get(k) for k in ("model", "precision", "calib_seed", "converts", "error")}))
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--precision", required=True, choices=["fp32", "fp16", "int8"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib-manifest", default=None)
    ap.add_argument("--calib-seed", type=int, default=None)
    a = ap.parse_args()
    export(a.model, a.precision, Path(a.out), a.checkpoint, a.calib_manifest, a.calib_seed)
