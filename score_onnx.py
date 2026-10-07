"""Score an exported ONNX artifact — the TEST side of the parity gate.

Deliberately reuses score.py's `load_wav` / `fix_length` / `read_manifest`, so
preprocessing is bit-identical to the PyTorch reference and the ONLY difference
between the two CSVs is the runtime. (The Android app ships its OWN resampler
and padding in Kotlin — that is a separate parity leg, phase_2 step 2.4.)

Emits the project score schema, so tests/parity.py consumes it directly.

    python score_onnx.py --onnx exports/tssdnet_fp32.onnx --model tssdnet \
        --dataset asvspoof19_eval --manifest configs/parity_manifest_tssdnet.txt \
        --precision fp32 \
        --out results/phase_2_conversion_app/parity_test_tssdnet_fp32.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import check_condition, fix_length, load_wav, read_manifest  # noqa: E402

RUNTIMES = ("onnx", "ort")    # both mean ONNX Runtime on the CPU EP; see progress.md
_INT8_OPS = {"QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear",
             "QLinearConv", "QLinearMatMul", "QLinearAdd", "QLinearMul",
             "ConvInteger", "MatMulInteger"}


def graph_precision(path) -> str:
    """What the graph actually is: int8 if it has quantization ops, fp16 if any
    initializer is float16, else fp32. --precision is checked against this so a
    label can never disagree with the artifact (B2's twin)."""
    import onnx
    m = onnx.load(str(path), load_external_data=False)
    if {n.op_type for n in m.graph.node} & _INT8_OPS:
        return "int8"
    if any(t.data_type == onnx.TensorProto.FLOAT16 for t in m.graph.initializer):
        return "fp16"
    return "fp32"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--model", required=True,
                    help="roster name — supplies input_samples and the CSV column")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--wav-dir", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--runtime", default="onnx")
    ap.add_argument("--condition", default="clean")
    ap.add_argument("--stress-manifest", default=None,
                    help="stress.py manifest: score the --condition rows of it, labels "
                         "joined from --dataset (H5 path)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--progress", type=int, default=0)
    ap.add_argument("--intra-threads", type=int, default=0,
                    help="0 = onnxruntime default")
    a = ap.parse_args()

    if a.runtime not in RUNTIMES:
        raise SystemExit(f"score_onnx.py runs ONNX Runtime; refusing to label its "
                         f"scores runtime={a.runtime!r} (allowed: {RUNTIMES})")
    got = graph_precision(a.onnx)
    if got != a.precision:
        raise SystemExit(f"{a.onnx} is an {got} graph; refusing to label its "
                         f"scores precision={a.precision!r}")
    from deploybench.data import native_conditions
    native = native_conditions(a.dataset) if a.dataset else None
    if a.stress_manifest and (a.manifest or a.wav_dir):
        raise SystemExit("--stress-manifest excludes --manifest and --wav-dir")
    check_condition(a.dataset, a.condition, native, bool(a.stress_manifest))

    import onnxruntime as ort
    from models.base import SPECS
    if SPECS[a.model].round != 1:
        raise SystemExit(f"{a.model} is a round-2 model (D54): scored by score.py (PyTorch fp32) only; "
                         "conversion is outside the audit paper")
    n = SPECS[a.model].input_samples

    so = ort.SessionOptions()
    if a.intra_threads:
        so.intra_op_num_threads = a.intra_threads
    sess = ort.InferenceSession(a.onnx, so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out_name = sess.get_outputs()[0].name
    print(f"onnx input {inp.name} {inp.shape} {inp.type} -> output {out_name}")
    # Exports are fixed-shape by design; a mismatch means the export and the
    # wrapper disagree about input length, which would silently change scores.
    fixed = [d for d in inp.shape if isinstance(d, int)]
    if fixed and fixed[-1] != n:
        raise SystemExit(f"graph expects {inp.shape}, spec says {n} samples")
    # convert/to_onnx.py exports with dynamic_axes=None — the BATCH dim is fixed
    # by design (doc 08: mobile runs batch=1). Feeding a 32-clip batch to a
    # batch-1 graph is an ORT InvalidArgument, so honour whatever the graph
    # declares. Scores are per-clip and independent, so batching is purely a
    # throughput choice and cannot change any value.
    if len(inp.shape) > 1 and isinstance(inp.shape[0], int):
        if a.batch != inp.shape[0]:
            print(f"graph has FIXED batch {inp.shape[0]}; overriding "
                  f"--batch {a.batch} -> {inp.shape[0]}")
        a.batch = inp.shape[0]

    def utts():
        if a.stress_manifest:
            from deploybench.stress import iter_stressed
            yield from iter_stressed(a.stress_manifest, a.dataset, a.condition)
            return
        if a.wav_dir:
            for p in sorted(Path(a.wav_dir).glob("**/*")):
                if p.suffix.lower() in (".wav", ".flac"):
                    yield p.stem, p, -1
            return
        from deploybench.data import protocol_iter
        if a.manifest:
            want = read_manifest(a.manifest)
            index = {u: (u, p, l) for u, p, l in protocol_iter(a.dataset)}
            miss = [u for u in want if u not in index]
            if miss:
                raise SystemExit(f"{len(miss)} manifest utt_ids absent from "
                                 f"{a.dataset} (first: {miss[:3]})")
            for u in want:
                yield index[u]
        else:
            yield from protocol_iter(a.dataset)

    rows, ids, waves, labs = [], [], [], []
    t0, seen = time.time(), 0

    def flush():
        if not waves:
            return
        x = np.stack(waves).astype("float32")
        s = np.asarray(sess.run([out_name], {inp.name: x})[0]).reshape(-1)
        for u, l, v in zip(ids, labs, s):
            rows.append((u, l, float(v)))
        ids.clear(); waves.clear(); labs.clear()

    for uid, path, lab in utts():
        ids.append(uid)
        labs.append(lab)
        waves.append(fix_length(load_wav(Path(path)), n))
        seen += 1
        if len(waves) >= a.batch:
            flush()
        if a.progress and seen % a.progress == 0:
            el = time.time() - t0
            print(f"  {seen} utts  {el:.0f}s  {seen/max(el,1e-9):.1f} utt/s",
                  flush=True)
    flush()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "label", "score", "model", "dataset",
                    "precision", "runtime", "condition", "seed"])
        for u, l, v in rows:
            cond = native[u] if native is not None else a.condition
            w.writerow([u, l, v, a.model, a.dataset or "adhoc", a.precision,
                        a.runtime, cond, a.seed])
    print(f"wrote {len(rows)} scores -> {out}")
    import score as _score
    print(f"FLAC files decoded via the MD5-verified fallback (D34): {_score.FLAC_FALLBACKS}")


if __name__ == "__main__":
    main()
