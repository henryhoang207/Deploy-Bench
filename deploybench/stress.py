"""Channel stress suite: deterministic transcode/noise/IR condition generator.

    python -m deploybench.stress --in wavs/ --out stressed/ --conditions opus6k mp3_32
    python -m deploybench.stress --list

Design rules (protocol §5 + methodology):
- TRAIN_CONDITIONS (any head-retraining augmentation) and EVAL_CONDITIONS are
  DISJOINT; enforced here, never hand-edited.
- Deterministic: noise segment + gain chosen by hash(utt_id), not RNG state.
"""
from __future__ import annotations
import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path

TRAIN_CONDITIONS = {"opus16k", "mp3_128", "amr12.2"}      # usable for aug ONLY
# condition -> (ffmpeg encode args, container extension)
EVAL_CONDITIONS = {
    "opus6k":   (["-c:a", "libopus", "-b:a", "6k"], ".ogg"),
    "opus24k":  (["-c:a", "libopus", "-b:a", "24k"], ".ogg"),
    "amr4.75":  (["-ar", "8000", "-ac", "1", "-c:a", "libopencore_amrnb",
                  "-b:a", "4.75k"], ".amr"),
    "mp3_32":   (["-c:a", "libmp3lame", "-b:a", "32k"], ".mp3"),
    "encodec6k": None,                                     # handled separately
    "snr20": None, "snr10": None, "snr5": None,            # MUSAN additive noise
    "mic_ir": None,                                        # IR convolution
}
CODEC_CONDITIONS = {k for k, v in EVAL_CONDITIONS.items() if v is not None}


import functools


@functools.lru_cache(maxsize=None)
def has_encoder(name: str) -> bool:
    out = subprocess.run(["ffmpeg", "-encoders"], capture_output=True,
                         text=True).stdout
    return name in out


def _ffmpeg_roundtrip(src: Path, dst: Path, codec_args: list[str],
                      ext: str = ".ogg") -> None:
    """src -> codec (native container) -> back to 16 kHz mono PCM wav."""
    tmp = dst.with_suffix(".codec" + ext)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                    *codec_args, str(tmp)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp),
                    "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(dst)],
                   check=True)
    tmp.unlink()


def _det_seed(utt_id: str) -> int:
    return int(hashlib.sha256(utt_id.encode()).hexdigest()[:8], 16)


def _add_noise(src: Path, dst: Path, snr_db: float, musan_dir: Path) -> None:
    import numpy as np
    import soundfile as sf
    wave, sr = sf.read(str(src), dtype="float32", always_2d=False)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    noise_files = sorted(musan_dir.glob("noise/**/*.wav"))
    if not noise_files:
        raise FileNotFoundError(f"no MUSAN noise wavs under {musan_dir}")
    rng = np.random.default_rng(_det_seed(src.stem))
    nf = noise_files[rng.integers(len(noise_files))]
    noise, nsr = sf.read(str(nf), dtype="float32", always_2d=False)
    if noise.ndim > 1:
        noise = noise.mean(axis=1)
    if nsr != sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(nsr, sr)
        noise = resample_poly(noise, sr // g, nsr // g).astype("float32")
    if len(noise) < len(wave):
        noise = np.tile(noise, int(np.ceil(len(wave) / len(noise))))
    start = rng.integers(0, len(noise) - len(wave) + 1)
    noise = noise[start:start + len(wave)]
    pw = np.mean(wave ** 2) + 1e-12
    pn = np.mean(noise ** 2) + 1e-12
    noise *= np.sqrt(pw / (pn * 10 ** (snr_db / 10)))
    out = wave + noise
    peak = np.abs(out).max()
    if peak > 0.999:
        out = out / peak * 0.999
    sf.write(str(dst), out, sr)


def _conv_ir(src: Path, dst: Path, ir_dir: Path) -> None:
    import numpy as np
    import soundfile as sf
    from scipy.signal import fftconvolve
    wave, sr = sf.read(str(src), dtype="float32", always_2d=False)
    irs = sorted(ir_dir.glob("**/*.wav"))
    if not irs:
        raise FileNotFoundError(f"no IR wavs under {ir_dir}")
    rng = np.random.default_rng(_det_seed(src.stem))
    ir, _ = sf.read(str(irs[rng.integers(len(irs))]), dtype="float32",
                    always_2d=False)
    if ir.ndim > 1:
        ir = ir.mean(axis=1)
    out = fftconvolve(wave, ir / (np.abs(ir).max() + 1e-9))[:len(wave)]
    peak = np.abs(out).max()
    if peak > 0.999:
        out = out / peak * 0.999
    sf.write(str(dst), out.astype("float32"), sr)


def _encodec_roundtrip(src: Path, dst: Path) -> None:
    """EnCodec 6 kbps resynthesis (H5). Requires `pip install encodec` + torch."""
    import torch
    import soundfile as sf
    from encodec import EncodecModel
    from encodec.utils import convert_audio
    import torchaudio
    model = _encodec_roundtrip._model
    if model is None:
        model = EncodecModel.encodec_model_24khz()
        model.set_target_bandwidth(6.0)
        _encodec_roundtrip._model = model
    wav, sr = torchaudio.load(str(src))
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    with torch.no_grad():
        frames = model.encode(wav.unsqueeze(0))
        out = model.decode(frames)[0]
    out16 = torchaudio.functional.resample(out, model.sample_rate, 16000)
    sf.write(str(dst), out16.squeeze(0).numpy().T, 16000)


_encodec_roundtrip._model = None


def apply_condition(src: Path, dst: Path, condition: str,
                    musan_dir: Path | None = None,
                    ir_dir: Path | None = None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if condition == "clean":
        shutil.copy(src, dst)
    elif condition.startswith("snr"):
        _add_noise(src, dst, float(condition[3:]), musan_dir)
    elif condition == "mic_ir":
        _conv_ir(src, dst, ir_dir)
    elif condition == "encodec6k":
        _encodec_roundtrip(src, dst)
    elif condition in CODEC_CONDITIONS:
        args, ext = EVAL_CONDITIONS[condition]
        enc = args[args.index("-c:a") + 1] if "-c:a" in args else None
        if enc and not has_encoder(enc):
            raise RuntimeError(
                f"ffmpeg lacks encoder {enc!r} needed for {condition!r} — "
                "install a full ffmpeg build (see phase_0 step_02).")
        _ffmpeg_roundtrip(src, dst, args, ext)
    else:
        raise ValueError(f"unknown condition {condition!r} "
                         f"(train-only conditions {TRAIN_CONDITIONS} are barred here)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", help="input wav folder")
    ap.add_argument("--out", dest="out", help="output root (per-condition subdirs)")
    ap.add_argument("--conditions", nargs="+", default=list(EVAL_CONDITIONS))
    ap.add_argument("--musan", default=None)
    ap.add_argument("--ir", default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        print("eval:", ", ".join(EVAL_CONDITIONS))
        print("train-only (aug):", ", ".join(TRAIN_CONDITIONS))
        return
    src_files = sorted(Path(args.src).glob("**/*.wav")) + \
        sorted(Path(args.src).glob("**/*.flac"))
    manifest = []
    for cond in args.conditions:
        for f in src_files:
            dst = Path(args.out) / cond / (f.stem + ".wav")
            apply_condition(f, dst, cond,
                            musan_dir=Path(args.musan) if args.musan else None,
                            ir_dir=Path(args.ir) if args.ir else None)
            manifest.append((f.stem, cond, str(dst)))
        print(f"[{cond}] {len(src_files)} files")
    mpath = Path(args.out) / "manifest.csv"
    with open(mpath, "w") as f:
        f.write("utt_id,condition,path\n")
        for row in manifest:
            f.write(",".join(row) + "\n")
    print(f"manifest -> {mpath}")


if __name__ == "__main__":
    main()
