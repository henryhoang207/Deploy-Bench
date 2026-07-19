"""PyTorch -> ExecuTorch (.pte) via XNNPACK partitioner. UNTESTED — the
ExecuTorch export API has changed across 2025-26 releases; at Phase 2 kickoff
verify against the CURRENT docs (pytorch.org/executorch) and pin the version.
The structure below follows the stable pattern (export -> to_edge -> partner
-> to_executorch); expect symbol renames."""
import argparse
import json
import time
from pathlib import Path


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
    log = {"model": model_name, "runtime": "executorch", "converts": None,
           "error": None, "started": time.time()}
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from torch.export import export as texport
        from executorch.exir import to_edge
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner)                                # TODO-VERIFY path
        prog = to_edge(texport(net, dummy))
        prog = prog.to_backend(XnnpackPartitioner())
        et = prog.to_executorch()
        out = out_dir / f"{model_name}.pte"
        out.write_bytes(et.buffer)
        log["converts"] = True
        log["size_mb"] = round(out.stat().st_size / 1e6, 2)
    except Exception as e:                                     # noqa: BLE001
        log["converts"] = False
        log["error"] = f"{type(e).__name__}: {e}"
    log["wall_s"] = round(time.time() - log.pop("started"), 1)
    (out_dir / f"conversion_log_{model_name}_executorch.json").write_text(
        json.dumps(log, indent=2))
    print(json.dumps({k: log[k] for k in ("model", "converts", "error")}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default="exports")
    a = ap.parse_args()
    export(a.model, Path(a.out), a.checkpoint)
