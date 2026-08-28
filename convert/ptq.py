"""Static int8 PTQ of an ONNX export (per-channel weights, per-tensor acts).
UNTESTED in sandbox. Calibration manifest: configs/calibration_500.txt
(500 balanced source-dev clips — generate in step 2.1 once data exists).

    python convert/ptq.py --onnx exports/aasist_fp32.onnx --calib-dir <wavs> \
        [--calib-method percentile] [--exclude-nodes n1,n2] [--keep-io-fp32]

PER-MODEL, NOT UNIVERSAL (this is H6): int8 coverage differs by architecture.
Pre-declared predictions for the FROZEN ROSTER (D30) — these are the H6 test:
  aasist / aasist_l : softmax, topk, gather, ScatterND, SELU -> expect fallback
  rawnet2           : GRU -> expect fallback, and it is 16.1M of 17.6M params,
                      so an int8 row that falls back here saves almost nothing
  tssdnet           : plain conv1d/BN/ReLU -> expect clean (CONFIRMED, but only
                      with the first 8 convs held fp32; see OD-6)
  xlsr_aasist       : LayerNorm/GELU -> expect fallback; expected not to fit
ALWAYS read the emitted *_ptq_log.json fallback list before calling a row "int8".

Note --exclude-nodes now exists chiefly for the OD-6 recipe (holding early convs
in fp32), not for feature frontends: D30 dropped LFCC-LCNN, the only roster model
that carried an in-graph STFT/log/DCT frontend.

RAW-AUDIO CAVEAT: quantizing the input tensor = ~256 levels of waveform (worse
than 8-bit PCM) into SincConv. --keep-io-fp32 is the usual mitigation; whichever
you choose, record it — it is a reportable methodological decision, not a default.

ARM vs x86: S8S8 (below) is the ARM-recommended config. The SAME artifact scored
on x86 desktop for the P5 grid can differ on non-VNNI AVX2 machines (saturation).
If int8 desktop-vs-device parity drifts, suspect this first.
"""
import argparse
from pathlib import Path
import numpy as np


class WavCalibrationReader:
    """onnxruntime.quantization CalibrationDataReader over a wav folder."""

    def __init__(self, wav_dir: Path, input_samples: int, input_name: str,
                 limit: int = 500, manifest=None):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from score import load_wav, fix_length
        # BUGFIX: the old form took `limit` of EACH extension (up to 2x limit)
        # and let extension ordering, not the manifest, decide the balance.
        allf = sorted([p for p in Path(wav_dir).glob("**/*")
                       if p.suffix.lower() in (".wav", ".flac")])
        if manifest:
            # D27 fixes a BALANCED 500-clip calibration set. Directory order is
            # not balanced (19LA dev lists bona fide and spoof in blocks), so
            # without a manifest the composition is an accident, not a choice.
            from score import read_manifest
            want = read_manifest(manifest)
            by_stem = {p.stem: p for p in allf}
            missing = [u for u in want if u not in by_stem]
            if missing:
                raise SystemExit(f"{len(missing)} calibration ids absent from "
                                 f"{wav_dir} (first: {missing[:3]})")
            allf = [by_stem[u] for u in want]
        self.files = allf[:limit]
        self.i = 0
        self.n = input_samples
        self.name = input_name
        self._load, self._fix = load_wav, fix_length

    def get_next(self):
        if self.i >= len(self.files):
            return None
        w = self._fix(self._load(self.files[self.i]), self.n)
        self.i += 1
        return {self.name: w[None, :].astype(np.float32)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--calib-manifest", default=None,
                    help="utt_id list selecting the BALANCED calibration set "
                         "(D27). Without it, directory order decides the "
                         "composition — which is not a choice, it is luck.")
    ap.add_argument("--input-samples", type=int, default=64600)
    ap.add_argument("--calib-method", default="minmax",
                    choices=["minmax", "percentile", "entropy"],
                    help="minmax is outlier-sensitive; audio activations have "
                         "heavy tails. Record whichever you use.")
    ap.add_argument("--exclude-nodes", default="",
                    help="comma-separated node names to keep in fp32 (the OD-6 "
                         "recipe: TSSDNet needs its first 8 convs excluded)")
    ap.add_argument("--activation-type", default="int8", choices=["int8", "uint8"],
                    help="int8 = S8S8 (ARM-recommended); uint8 = U8S8 (ORT default)")
    ap.add_argument("--symmetric-activations", action="store_true",
                    help="symmetric activation ranges (zero-point 0)")
    ap.add_argument("--no-per-channel", action="store_true",
                    help="per-tensor weight scales instead of per-channel "
                         "(diagnostic: isolates per-channel axis problems)")
    ap.add_argument("--no-pre-process", action="store_true",
                    help="skip ORT quant_pre_process (BN folding + shape "
                         "inference). Only for measuring what pre-processing "
                         "is worth — leaving it off produces broken artifacts")
    ap.add_argument("--keep-io-fp32", action="store_true",
                    help="do not quantize graph input/output tensors — avoids "
                         "8-bit-audio input degradation on raw-waveform models")
    a = ap.parse_args()
    import json
    from onnxruntime.quantization import (quantize_static, QuantType,
                                          QuantFormat, CalibrationMethod)
    _CAL = {"minmax": CalibrationMethod.MinMax,
            "percentile": CalibrationMethod.Percentile,
            "entropy": CalibrationMethod.Entropy}
    src = Path(a.onnx)
    dst = src.with_name(src.stem.replace("_fp32", "") + "_int8.onnx")
    exclude = [n for n in a.exclude_nodes.split(",") if n]
    # STALE-ARTIFACT GUARD. If quantisation dies partway (ORT can OOM building
    # histograms for percentile/entropy), a previous _int8.onnx left on disk is
    # perfectly loadable, and the next scoring step will happily measure THAT —
    # emitting a complete, plausible result set describing a model this run did
    # not produce. That happened once and cost a wrong conclusion. Delete first,
    # and assert afterwards that we really wrote something new.
    if dst.exists():
        print(f"removing previous artifact {dst.name} "
              f"({dst.stat().st_size} bytes) before re-quantising")
        dst.unlink()
    if not a.calib_manifest:
        print("WARNING: no --calib-manifest; the first "
              f"{a.input_samples and 500} files of {a.calib_dir} in directory "
              "order become the calibration set. D27 specifies a BALANCED set "
              "— pass --calib-manifest or record the deviation.")
    if a.keep_io_fp32:
        print("WARNING: --keep-io-fp32 currently sets extra_options "
              "{'QuantizeBias': False}, which is about BIAS, not graph I/O. "
              "This is an open bug (notes/tier1/nagel21_quantization_"
              "whitepaper.md). Verify the emitted graph's input tensor type "
              "before reporting any int8 row as input-fp32.")
    # extra_options assembled explicitly so several knobs can combine. Note
    # CalibTensorRangeSymmetric is one of only four Calib* keys quantize_static
    # forwards to the calibrator — the percentile VALUE is not forwardable, so
    # tuning that would need the lower-level create_calibrator API.
    xopt = {}
    if a.keep_io_fp32:
        xopt["QuantizeBias"] = False
    if a.symmetric_activations:
        xopt["ActivationSymmetric"] = True
        xopt["CalibTensorRangeSymmetric"] = True

    def _quantize(model_in: str, model_out: str):
        # ORT's quant_pre_process runs symbolic shape inference and, critically,
        # FOLDS BatchNormalization into the preceding Conv. Without it BN survives
        # as a standalone node: it gets its own quantise/dequantise boundary, and
        # per-channel quantisation along axis 1 is attempted on BN's rank-1 gamma
        # ("Axis 1 is out-of-range for weight '...bn3.weight' with rank 1"), which
        # silently degrades to something malformed. Skipping this step is what
        # produced the first int8 artifact whose output collapsed onto 127 values
        # below every threshold. ORT warns about it twice and the warning is real.
        if not a.no_pre_process:
            from onnxruntime.quantization.shape_inference import quant_pre_process
            pre = str(Path(model_in).with_name(Path(model_in).stem + "_pre.onnx"))
            quant_pre_process(model_in, pre, skip_symbolic_shape=False)
            model_in = pre
        quantize_static(
            model_in, model_out,
            WavCalibrationReader(Path(a.calib_dir), a.input_samples, "waveform",
                                 manifest=a.calib_manifest),
            quant_format=QuantFormat.QDQ,
            per_channel=not a.no_per_channel,      # per-channel weights
            weight_type=QuantType.QInt8,
            # S8S8 is the ARM-recommended default; U8S8 is ORT's own default and
            # is sometimes markedly more accurate, so it is worth measuring.
            activation_type=(QuantType.QUInt8 if a.activation_type == "uint8"
                             else QuantType.QInt8),
            calibrate_method=_CAL[a.calib_method],
            nodes_to_exclude=exclude,
            extra_options=xopt or None)

    # ORT quantization calls onnx.shape_inference.infer_shapes_path, which goes
    # through the ONNX C++ API. On Windows that API takes NARROW-CHAR paths and
    # silently fails on non-ASCII directories — it writes no '<stem>-inferred.onnx'
    # and the next onnx.load() dies with FileNotFoundError. This project lives
    # under "...\Tai lieu\...", so that is the default case here, for every model.
    # Stage through an ASCII-only temp dir; the emitted graph is identical, only
    # the working path differs.
    def _is_ascii(p) -> bool:
        try:
            str(p).encode("ascii")
            return True
        except UnicodeEncodeError:
            return False

    if _is_ascii(src) and _is_ascii(dst):
        _quantize(str(src), str(dst))
    else:
        import shutil
        import tempfile
        with tempfile.TemporaryDirectory(prefix="ptq_ascii_") as td:
            t_src, t_dst = Path(td) / src.name, Path(td) / dst.name
            if not _is_ascii(t_src):                 # e.g. non-ASCII TEMP too
                raise SystemExit(f"temp dir is also non-ASCII: {td}")
            # torch.onnx.export writes weights to an EXTERNAL DATA sidecar
            # ('<name>.onnx.data') whenever they exceed the protobuf limit, and
            # the .onnx alone is then just the graph. Copy every sibling that
            # shares the stem, or quantization loads a weightless model.
            staged = []
            for f in sorted(src.parent.glob(src.stem + "*")):
                if f.is_file():
                    shutil.copyfile(f, Path(td) / f.name)
                    staged.append(f.name)
            print(f"non-ASCII path detected; quantizing via {td}  "
                  f"(staged: {', '.join(staged)})")
            _quantize(str(t_src), str(t_dst))
            dst.parent.mkdir(parents=True, exist_ok=True)
            # the int8 result may carry its own sidecar too
            for f in sorted(Path(td).glob(dst.stem + "*")):
                if f.is_file():
                    shutil.copyfile(f, dst.parent / f.name)

    if not dst.exists():
        raise SystemExit(f"quantisation produced no artifact at {dst} — refusing "
                         "to continue; a later step would score a stale file")

    # Accounting: what fraction of the graph actually became int8.
    import onnx
    g = onnx.load(str(dst)).graph
    ops = [n.op_type for n in g.node]
    # QDQ format does NOT fuse into QLinear* operators — it keeps the original op
    # and wraps its tensors in QuantizeLinear/DequantizeLinear pairs, which the
    # runtime fuses at load time. Counting QLinear* here therefore reports "0
    # quantised, everything fell back" for a perfectly quantised QDQ graph, and
    # would have misreported the H6 conversion row for every model in the roster.
    # Count the Q/DQ pairs, and treat an op as still-fp32 only if none of its
    # inputs arrive through a DequantizeLinear.
    n_q = sum(1 for o in ops if o == "QuantizeLinear")
    n_dq = sum(1 for o in ops if o == "DequantizeLinear")
    dq_out = {o for n in g.node if n.op_type == "DequantizeLinear" for o in n.output}
    real_ops = [n for n in g.node
                if n.op_type not in ("QuantizeLinear", "DequantizeLinear")]
    quantized = sorted({n.op_type for n in real_ops
                        if any(i in dq_out for i in n.input)})
    fp32_left = sorted({n.op_type for n in real_ops
                        if not any(i in dq_out for i in n.input)})
    log = {"src": str(src), "dst": str(dst), "calib_method": a.calib_method,
           "calib_dir": str(a.calib_dir),
           "calib_manifest": a.calib_manifest,
           "calib_clips": len(WavCalibrationReader(
               Path(a.calib_dir), a.input_samples, "waveform",
               manifest=a.calib_manifest).files),
           "excluded_nodes": exclude, "keep_io_fp32": a.keep_io_fp32,
           # elem_type 1 = FLOAT, 2 = UINT8, 3 = INT8 — this is the ground
           # truth for whether the waveform input really stayed fp32.
           "graph_input_type": g.input[0].type.tensor_type.elem_type,
           "quantized_op_types": quantized,
           "fp32_fallback_op_types": fp32_left,
           "n_quantize_linear": n_q, "n_dequantize_linear": n_dq,
           "artifact_bytes": dst.stat().st_size,
           "n_nodes": len(ops)}
    Path(str(dst).replace(".onnx", "_ptq_log.json")).write_text(
        json.dumps(log, indent=2))
    print(f"int8 -> {dst}")
    print("fp32 fallback ops:", ", ".join(fp32_left) or "(none)")


if __name__ == "__main__":
    main()
