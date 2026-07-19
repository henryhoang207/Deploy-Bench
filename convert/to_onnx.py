"""PyTorch -> ONNX export (Phase 2). UNTESTED in sandbox (no torch/checkpoints);
first-run fix cycle expected. Version-pin torch+onnx at Phase 2 kickoff and
verify opset support against current ORT-Mobile docs at that time.

    python convert/to_onnx.py --model aasist --precision fp32 --out exports/
"""
import argparse
import json
import time
from pathlib import Path


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
        if precision == "fp16":
            net = net.half()
            dummy = dummy.half()
        torch.onnx.export(net, dummy, str(out), opset_version=17,
                          input_names=["waveform"], output_names=["score"],
                          dynamic_axes=None)  # FIXED shape by design (doc 08)
        import onnx
        m = onnx.load(str(out))
        onnx.checker.check_model(m)
        log["converts"] = True
        log["ops"] = sorted({n.op_type for n in m.graph.node})
        log["size_mb"] = round(out.stat().st_size / 1e6, 2)
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
