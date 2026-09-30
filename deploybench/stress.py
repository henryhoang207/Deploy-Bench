"""Channel stress suite: deterministic transcode/noise/IR condition generator.

    python -m deploybench.stress --in wavs/ --out stressed/ --conditions opus6k mp3_32
    python -m deploybench.stress --dataset asvspoof19_eval --ids ids.txt --out st/         --conditions clean encodec6k          # H5 path: labelled, scorable manifest
    python -m deploybench.stress --list

Scoring a stress set: score.py / score_onnx.py --dataset <source> --stress-manifest
<out>/manifest.csv --condition <cond>. Labels come from the source protocol
(iter_stressed); a folder-mode manifest has no source and is refused.

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
import os


@functools.lru_cache(maxsize=None)
def ffmpeg_exe() -> str:
    """$FFMPEG, else ffmpeg on PATH, else imageio-ffmpeg's bundled binary (sol1 has
    no system ffmpeg). Refuses if none exists."""
    exe = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if not exe:
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:                                       # noqa: BLE001
            exe = None
    if not exe:
        raise RuntimeError("no ffmpeg: set $FFMPEG, put ffmpeg on PATH, or install imageio-ffmpeg")
    return exe


@functools.lru_cache(maxsize=None)
def has_encoder(name: str) -> bool:
    out = subprocess.run([ffmpeg_exe(), "-hide_banner", "-encoders"], capture_output=True,
                         text=True).stdout
    return name in out


def _ffmpeg_roundtrip(src: Path, dst: Path, codec_args: list[str],
                      ext: str = ".ogg") -> None:
    """src -> codec (native container) -> back to 16 kHz mono PCM wav."""
    tmp = dst.with_suffix(".codec" + ext)
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
                    *codec_args, str(tmp)], check=True)
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(tmp),
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


MANIFEST_COLS = ("utt_id", "condition", "path", "source_dataset")
# Stressing audio that already carries a native per-trial condition (DF21's codecs) would
# stack two channels under one label; stress sources must be clean protocol audio.
NO_STRESS_SOURCES = {"asvspoof21_df_eval", "asvspoof21_df_eval_clean"}


def iter_stressed(manifest, dataset: str, condition: str, root=None):
    """Yield (utt_id, stressed_path, label) for the `condition` rows of a stress.py
    manifest, in manifest order, with labels joined from `dataset`'s protocol (the
    manifest carries no labels, so there is one source of truth). Refuses anything
    it cannot honour: a train-only or unknown condition, a manifest from another
    source, no rows for the condition, a duplicate or unknown utt_id, a missing file."""
    import csv
    from deploybench.data import protocol_iter
    if condition in TRAIN_CONDITIONS:
        raise ValueError(f"{condition!r} is a train-only augmentation condition; "
                         "it is barred from evaluation")
    if condition != "clean" and condition not in EVAL_CONDITIONS:
        raise ValueError(f"unknown stress condition {condition!r}")
    if dataset in NO_STRESS_SOURCES:
        raise ValueError(f"{dataset} carries native per-trial codecs; it is not a stress source")
    with open(manifest, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        if tuple(rd.fieldnames or ()) != MANIFEST_COLS:
            raise ValueError(f"{manifest}: columns {rd.fieldnames}, expected {MANIFEST_COLS} "
                             "(regenerate with `stress.py --dataset`; folder-mode manifests "
                             "have no source dataset and cannot be labelled)")
        rows = [r for r in rd if r["condition"] == condition]
    if not rows:
        raise ValueError(f"{manifest}: no rows for condition {condition!r}")
    src = {r["source_dataset"] for r in rows}
    if src != {dataset}:
        raise ValueError(f"{manifest}: rows come from {sorted(src)}, not {dataset!r}")
    labels = {u: lab for u, _p, lab in protocol_iter(dataset, root)}
    seen = set()
    for r in rows:
        u = r["utt_id"]
        if u in seen:
            raise ValueError(f"{manifest}: duplicate {u} under {condition!r}")
        seen.add(u)
        if u not in labels:
            raise ValueError(f"{manifest}: {u} is not in the {dataset} protocol")
        p = Path(r["path"])
        if not p.is_file():
            raise FileNotFoundError(f"{manifest}: stressed file missing: {p}")
        yield u, p, labels[u]


def _provenance(args, conditions, ids) -> dict:
    import datetime
    import hashlib as _h
    prov = {"created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "source_dataset": args.dataset, "n_utts": len(ids),
            "ids_sha256": _h.sha256("\n".join(ids).encode()).hexdigest(),
            "conditions": list(conditions), "musan": args.musan, "ir": args.ir}
    try:
        prov["ffmpeg"] = subprocess.run([ffmpeg_exe(), "-version"], capture_output=True,
                                        text=True).stdout.splitlines()[0]
    except Exception as e:                                      # noqa: BLE001
        prov["ffmpeg"] = f"unavailable: {e}"
    if "encodec6k" in conditions:
        import importlib.metadata as md
        prov["encodec"] = md.version("encodec")
    return prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", help="input wav folder (unlabelled; cannot be scored)")
    ap.add_argument("--dataset", default=None,
                    help="stress this protocol's audio instead of a folder; the manifest "
                         "records it so the scorers can join labels (H5 path)")
    ap.add_argument("--ids", default=None,
                    help="with --dataset: utt_id list (one per line) selecting the subset")
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
    bad = [c for c in args.conditions if c != "clean" and c not in EVAL_CONDITIONS]
    if bad:
        raise SystemExit(f"not evaluation conditions: {bad} (train-only: {sorted(TRAIN_CONDITIONS)})")
    out = Path(args.out)
    mpath = out / "manifest.csv"
    if mpath.exists():
        raise SystemExit(f"{mpath} exists: a stress set is generated once; use a new --out")
    if bool(args.src) == bool(args.dataset):
        raise SystemExit("give exactly one of --in (folder) or --dataset (protocol)")
    if args.dataset:
        if args.dataset in NO_STRESS_SOURCES:
            raise SystemExit(f"{args.dataset} carries native per-trial codecs; not a stress source")
        from deploybench.data import protocol_iter
        index = {u: p for u, p, _lab in protocol_iter(args.dataset)}
        if args.ids:
            from deploybench.manifests import read_ids
            ids = read_ids(Path(args.ids))
            miss = [u for u in ids if u not in index]
            if miss:
                raise SystemExit(f"{len(miss)} ids not in {args.dataset} (first {miss[:3]})")
        else:
            ids = list(index)
        sources = [(u, Path(index[u])) for u in ids]
    else:
        sources = [(f.stem, f) for f in sorted(Path(args.src).glob("**/*.wav")) +
                   sorted(Path(args.src).glob("**/*.flac"))]
    manifest = []
    for cond in args.conditions:
        for u, f in sources:
            # "clean" is a byte copy, so it keeps the source container (a FLAC renamed
            # .wav would misdescribe the file); every other condition writes 16 kHz wav.
            dst = out / cond / (u + (f.suffix if cond == "clean" else ".wav"))
            apply_condition(f, dst, cond,
                            musan_dir=Path(args.musan) if args.musan else None,
                            ir_dir=Path(args.ir) if args.ir else None)
            manifest.append((u, cond, str(dst), args.dataset or ""))
        print(f"[{cond}] {len(sources)} files")
    import csv
    out.mkdir(parents=True, exist_ok=True)
    with open(mpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(MANIFEST_COLS)
        w.writerows(manifest)
    if args.dataset:
        import json
        (out / "manifest.json").write_text(json.dumps(
            _provenance(args, args.conditions, [u for u, _ in sources]), indent=2) + "\n")
    print(f"manifest -> {mpath}")


if __name__ == "__main__":
    main()
