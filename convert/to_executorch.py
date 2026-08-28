"""PyTorch -> ExecuTorch (.pte) via the XNNPACK partitioner.

API VERIFIED 2026-07-25 against the live docs
(docs.pytorch.org/executorch/stable/backends-xnnpack.html and
pytorch/executorch@main docs/source/backends/xnnpack/xnnpack-overview.md).
The currently documented sequence is ONE call:

    to_edge_transform_and_lower(torch.export.export(net, inputs),
                                partitioner=[XnnpackPartitioner()]).to_executorch()

which replaces the older `to_edge(...).to_backend(XnnpackPartitioner())` used
by the previous version of this file. The `.pte` is written with
`program.write_to_file(fh)`; `program.buffer` still exists but materialises the
whole payload in memory. Both paths are kept below (current first, legacy
fallback) and which one ran is recorded in the conversion log.

VERSION PIN: executorch 1.3.1 (released 2026-05-29), requires Python
>=3.10,<3.14 — a 3.14 interpreter CANNOT install it. Resolved versions are
written into the conversion log for the provenance appendix.

    python convert/to_executorch.py --model tssdnet --checkpoint <path> --out exports/
"""
import argparse
import json
import time
from pathlib import Path


def _lower(net, dummy, log):
    """Current API first, pre-1.0 API as fallback. Records which one ran."""
    import torch
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner)
    exported = torch.export.export(net, dummy)
    try:
        from executorch.exir import to_edge_transform_and_lower
        log["api"] = "to_edge_transform_and_lower"
        return to_edge_transform_and_lower(
            exported, partitioner=[XnnpackPartitioner()]).to_executorch()
    except ImportError:
        from executorch.exir import to_edge
        log["api"] = "to_edge+to_backend (legacy)"
        return to_edge(exported).to_backend(XnnpackPartitioner()).to_executorch()


def _write_pte(program, out: Path, log):
    try:
        with open(out, "wb") as fh:
            program.write_to_file(fh)
        log["writer"] = "write_to_file"
    except AttributeError:
        out.write_bytes(program.buffer)
        log["writer"] = "buffer (legacy)"


def export(model_name: str, out_dir: Path, checkpoint=None):
    import torch
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from models.base import load_model, SPECS
    spec = SPECS[model_name]
    wrap = load_model(model_name, checkpoint, "cpu")
    if not hasattr(wrap, "export_module"):
        raise SystemExit(f"{model_name}: wrapper lacks export_module() — same "
                         "single-score contract as to_onnx.py.")
    net = wrap.export_module().eval()   # WAS wrap.model: raw [B,2] logits would
                                        # break the app/parity single-score contract
    dummy = (torch.zeros(1, spec.input_samples),)
    log = {"model": model_name, "runtime": "executorch",
           "input_samples": spec.input_samples, "converts": None,
           "api": None, "writer": None, "error": None,
           "torch_version": torch.__version__, "python": sys.version.split()[0],
           "started": time.time()}
    try:
        import executorch
        log["executorch_version"] = getattr(executorch, "__version__", "unknown")
    except ImportError:
        log["executorch_version"] = None
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{model_name}.pte"
    try:
        _write_pte(_lower(net, dummy, log), out, log)
        log["converts"] = True
        log["size_mb"] = round(out.stat().st_size / 1e6, 2)
    except Exception as e:                                     # noqa: BLE001
        log["converts"] = False
        log["error"] = f"{type(e).__name__}: {e}"              # a RESULT, not a crash
    log["wall_s"] = round(time.time() - log.pop("started"), 1)
    (out_dir / f"conversion_log_{model_name}_executorch.json").write_text(
        json.dumps(log, indent=2))
    print(json.dumps({k: log[k] for k in
                      ("model", "converts", "api", "executorch_version", "error")}))
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default="exports")
    a = ap.parse_args()
    export(a.model, Path(a.out), a.checkpoint)
