"""PyTorch -> ONNX export (Phase 2). UNTESTED in sandbox (no torch/checkpoints);
first-run fix cycle expected. Version-pin torch+onnx at Phase 2 kickoff and
verify opset support against current ORT-Mobile docs at that time.

    python convert/to_onnx.py --model aasist --precision fp32 --out exports/
"""
import argparse
import json
import time
from pathlib import Path


def ort_smoke(path: Path, input_samples: int) -> None:
    """Raise unless ONNX Runtime (CPU EP) loads the graph and returns one finite
    score for a zero waveform. Any exception is logged as the conversion result."""
    import numpy as np
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    batch = inp.shape[0] if isinstance(inp.shape[0], int) else 1
    out = sess.run(None, {inp.name: np.zeros((batch, input_samples), np.float32)})[0]
    if not np.all(np.isfinite(np.asarray(out, dtype=np.float64))):
        raise ValueError(f"ORT loads {Path(path).name} but its output is not finite")


def export(model_name: str, precision: str, out_dir: Path, checkpoint=None):
    import torch
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from models.base import load_model, SPECS

    spec = SPECS[model_name]
    wrap = load_model(model_name, checkpoint, "cpu")
    if not hasattr(wrap, "export_module"):
        raise SystemExit(
            f"{model_name}: wrapper lacks export_module() — the raw module's "
            "output shape would not match the app/parity contract (single "
            "score, higher=bona fide). Add it to the wrapper first.")
    net = wrap.export_module().eval()
    dummy = torch.zeros(1, spec.input_samples)
    log = {"model": model_name, "runtime": "ort", "precision": precision,
           "input_samples": spec.input_samples, "opset": 17,
           "started": time.time(), "converts": None, "fallback_ops": [],
           "error": None}
    out = out_dir / f"{model_name}_{precision}.onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        # NOTE: do NOT net.half() before export — many CPU ops have no Half
        # kernel and tracing dies ("not implemented for 'Half'"). Export fp32,
        # then convert the ONNX graph. (fp16 on mobile CPU is often storage-only:
        # ORT may insert casts and compute in fp32 — record that in the log.)
        _to_fp16 = (precision == "fp16")
        torch.onnx.export(net, dummy, str(out), opset_version=17,
                          input_names=["waveform"], output_names=["score"],
                          dynamic_axes=None)  # FIXED shape by design (doc 08)
        import onnx
        m = onnx.load(str(out))
        if _to_fp16:
            from onnxconverter_common import float16
            m = float16.convert_float_to_float16(m, keep_io_types=True)
            onnx.save(m, str(out))
            # convert_float_to_float16 INLINES every initializer as fp16, so any
            # external-data sidecar torch.onnx.export wrote for the fp32 pass is
            # now orphaned — nothing references it, but it sits next to the model
            # at full fp32 size. Left alone it triples the apparent artifact
            # (710 KB graph + 1.39 MB dead sidecar) and would put a 3x-wrong
            # number in the H6 size column. Drop it once we confirm nothing
            # points at it.
            ext = [i for i in m.graph.initializer
                   if i.HasField("data_location") and i.data_location == 1]
            sidecar = out.with_name(out.name + ".data")
            if not ext and sidecar.exists():
                orphan_bytes = sidecar.stat().st_size
                sidecar.unlink()
                log["orphan_sidecar_removed_bytes"] = orphan_bytes
        onnx.checker.check_model(m)
        # The checker is not the runtime. rawnet2 fp16 (2026-09-29) passed
        # check_model and was logged converts=true, yet ORT refused to load it
        # (a Cast whose declared output type the fp16 pass changed). "Converts"
        # means the runtime we ship loads it and runs one inference.
        ort_smoke(out, spec.input_samples)
        log["converts"] = True
        log["ops"] = sorted({n.op_type for n in m.graph.node})
        # Every file that must actually ship, not just the .onnx: for fp32 the
        # weights live in the sidecar and the graph alone is a few KB.
        shipped = sum(f.stat().st_size for f in out.parent.glob(out.name + "*")
                      if f.is_file())
        log["size_mb"] = round(out.stat().st_size / 1e6, 2)
        log["shipped_bytes"] = shipped
    except Exception as e:                                    # noqa: BLE001
        log["converts"] = False
        log["error"] = f"{type(e).__name__}: {e}"             # a RESULT, not a crash
    log["wall_s"] = round(time.time() - log.pop("started"), 1)
    (out_dir / f"conversion_log_{model_name}_ort_{precision}.json").write_text(
        json.dumps(log, indent=2))
    print(json.dumps({k: log[k] for k in ("model", "precision", "converts", "error")}))
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default="exports")
    a = ap.parse_args()
    export(a.model, a.precision, Path(a.out), a.checkpoint)
