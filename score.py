"""Unified scoring CLI.

    python score.py --model aasist --dataset asvspoof19_eval --out results/scores/
    python score.py --model dummy --wav-dir some/folder --out /tmp/x.csv  # any WAV folder

Reads dataset protocol via deploybench.data, batches fixed-length waveforms
(pad-repeat/center-crop per model spec), writes the project score schema.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np


def fix_length(wave: np.ndarray, n: int) -> np.ndarray:
    """Authors' eval convention (clovaai/aasist data_utils.pad; ghua-ac TSSDNet
    data prep): truncate long clips FROM THE START, repeat-tile short clips.
    Do NOT center-crop — it changes scores vs published eval pipelines."""
    if len(wave) >= n:
        return wave[:n]
    reps = int(np.ceil(n / max(len(wave), 1)))
    return np.tile(wave, reps)[:n]


def load_wav(path: Path, target_sr: int = 16000) -> np.ndarray:
    import soundfile as sf
    wave, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if sr != target_sr:
        # polyphase resample via scipy (matches desktop reference; the Android
        # app ships its own resampler, parity-checked in phase_2 step_03)
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, target_sr)
        wave = resample_poly(wave, target_sr // g, sr // g).astype("float32")
    return wave


def iter_utts(args):
    if args.wav_dir:
        for p in sorted(Path(args.wav_dir).glob("**/*")):
            if p.suffix.lower() in (".wav", ".flac"):
                yield p.stem, p, -1
    else:
        from deploybench.data import protocol_iter      # completed in step 1.2
        yield from protocol_iter(args.dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--wav-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--runtime", default="pytorch")
    ap.add_argument("--condition", default="clean")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from models.base import load_model, SPECS
    model = load_model(args.model, args.checkpoint, args.device)
    n = SPECS[args.model].input_samples

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, batch_ids, batch_waves, batch_labels = [], [], [], []

    def flush():
        if not batch_waves:
            return
        scores = model.score_batch(np.stack(batch_waves))
        for uid, lab, sc in zip(batch_ids, batch_labels, scores):
            rows.append((uid, lab, float(sc)))
        batch_ids.clear(); batch_waves.clear(); batch_labels.clear()

    for uid, path, label in iter_utts(args):
        batch_ids.append(uid)
        batch_labels.append(label)
        batch_waves.append(fix_length(load_wav(Path(path)), n))
        if len(batch_waves) >= args.batch:
            flush()
    flush()

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "label", "score", "model", "dataset",
                    "precision", "runtime", "condition", "seed"])
        for uid, lab, sc in rows:
            w.writerow([uid, lab, sc, args.model, args.dataset or "adhoc",
                        args.precision, args.runtime, args.condition, args.seed])
    print(f"wrote {len(rows)} scores -> {out}")


if __name__ == "__main__":
    main()
