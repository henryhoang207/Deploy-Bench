"""Static int8 PTQ of an ONNX export (per-channel weights, per-tensor acts).
UNTESTED in sandbox. Calibration manifest: configs/calibration_500.txt
(500 balanced source-dev clips — generate in step 2.1 once data exists).

    python convert/ptq.py --onnx exports/aasist_fp32.onnx --calib-dir <wavs>
"""
import argparse
from pathlib import Path
import numpy as np


class WavCalibrationReader:
    """onnxruntime.quantization CalibrationDataReader over a wav folder."""

    def __init__(self, wav_dir: Path, input_samples: int, input_name: str,
                 limit: int = 500):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from score import load_wav, fix_length
        self.files = sorted(Path(wav_dir).glob("**/*.wav"))[:limit] + \
            sorted(Path(wav_dir).glob("**/*.flac"))[:limit]
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
    ap.add_argument("--input-samples", type=int, default=64600)
    a = ap.parse_args()
    from onnxruntime.quantization import quantize_static, QuantType, QuantFormat
    src = Path(a.onnx)
    dst = src.with_name(src.stem.replace("_fp32", "") + "_int8.onnx")
    quantize_static(
        str(src), str(dst),
        WavCalibrationReader(Path(a.calib_dir), a.input_samples, "waveform"),
        quant_format=QuantFormat.QDQ,
        per_channel=True,                      # per-channel weights
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8)       # per-tensor activations (default)
    print(f"int8 -> {dst}")


if __name__ == "__main__":
    main()
