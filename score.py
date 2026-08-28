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


def load_wav(path: Path, target_sr: int = 16000, _retries: int = 4) -> np.ndarray:
    import soundfile as sf
    # Bounded retry on TRANSIENT open failures only. libsndfile raises a bare
    # "System error" when Windows momentarily refuses the handle (AV scan, I/O
    # hiccup); a single one of these previously killed a 71k-clip eval sweep at
    # utt ~25,000 on a file that reads back fine. This does NOT change read
    # semantics: on success the bytes, dtype and resampling path are identical,
    # so score.py and score_onnx.py still share one preprocessing definition.
    # A genuinely unreadable file still raises, just after the last attempt.
    import time
    for attempt in range(_retries + 1):
        try:
            wave, sr = sf.read(str(path), dtype="float32", always_2d=False)
            break
        except Exception:                                          # noqa: BLE001
            if attempt == _retries:
                raise
            time.sleep(0.25 * (attempt + 1))
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


def read_manifest(path):
    """One utt_id per line, '#' comments ignored. Order is the manifest's, so
    the parity reference and the parity test score the SAME clips in the SAME
    order (phase_2 step 2.3)."""
    ids = []
    for line in open(path, encoding="utf-8"):
        line = line.split("#")[0].strip()
        if line:
            ids.append(line.split()[0])
    return ids


def iter_utts(args):
    if args.wav_dir:
        for p in sorted(Path(args.wav_dir).glob("**/*")):
            if p.suffix.lower() in (".wav", ".flac"):
                yield p.stem, p, -1
    else:
        from deploybench.data import protocol_iter      # completed in step 1.2
        if getattr(args, "manifest", None):
            want = read_manifest(args.manifest)
            index = {u: (u, p, l) for u, p, l in protocol_iter(args.dataset)}
            miss = [u for u in want if u not in index]
            if miss:
                raise SystemExit(
                    f"{len(miss)} manifest utt_ids absent from "
                    f"{args.dataset} (first: {miss[:3]})")
            for u in want:
                yield index[u]
        else:
            yield from protocol_iter(args.dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--wav-dir", default=None)
    ap.add_argument("--manifest", default=None,
                    help="file of utt_ids (one per line) selecting a subset of "
                         "--dataset, in manifest order — the parity manifest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--runtime", default="pytorch")
    ap.add_argument("--condition", default="clean")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N utterances (smoke tests only "
                         "— NEVER for a reproduction or parity run)")
    ap.add_argument("--progress", type=int, default=0,
                    help="print a progress line every N utterances (0 = off)")
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

    import time
    t0, seen = time.time(), 0
    for uid, path, label in iter_utts(args):
        if args.limit is not None and seen >= args.limit:
            break
        batch_ids.append(uid)
        batch_labels.append(label)
        batch_waves.append(fix_length(load_wav(Path(path)), n))
        seen += 1
        if len(batch_waves) >= args.batch:
            flush()
        if args.progress and seen % args.progress == 0:
            el = time.time() - t0
            print(f"  {seen} utts  {el:.0f}s  {seen / max(el, 1e-9):.1f} utt/s",
                  flush=True)
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
