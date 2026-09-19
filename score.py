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


#: files decoded through _flac_md5_verified in this process (reported by callers)
FLAC_FALLBACKS = 0


def _flac_md5_verified(path: Path):
    """Decode a FLAC that libsndfile rejects and accept the samples ONLY if they
    match the file's STREAMINFO MD5, i.e. are provably the encoder's input.
    DF21 class A (D34): libFLAC >= 1.4 rejects these files; libFLAC 1.3.4 and
    ffmpeg decode them bit-identically and the MD5 verifies. Class B fails the
    MD5 and raises here. int16 / 32768 equals soundfile's float32 bit-exactly
    (verified on 200 DF21 files). Returns (float32 wave, sample rate)."""
    import hashlib
    import subprocess
    import imageio_ffmpeg
    head = open(path, "rb").read(42)
    if head[:4] != b"fLaC" or (head[4] & 0x7F) != 0:
        raise ValueError(f"{path}: first metadata block is not STREAMINFO; "
                         "cannot verify a fallback decode")
    si = head[8:42]
    x = int.from_bytes(si[10:18], "big")
    sr, ch, bps = x >> 44, ((x >> 41) & 7) + 1, ((x >> 36) & 31) + 1
    md5 = si[18:34]
    if (ch, bps) != (1, 16):
        raise ValueError(f"{path}: fallback handles mono 16-bit FLAC only "
                         f"(got {ch} ch, {bps} bit)")
    if md5 == bytes(16):
        raise ValueError(f"{path}: STREAMINFO MD5 is unset; cannot verify")
    raw = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i",
                          str(path), "-f", "s16le", "-acodec", "pcm_s16le", "-"],
                         capture_output=True, check=True).stdout
    if hashlib.md5(raw).digest() != md5:
        raise ValueError(f"{path}: fallback decode does not match the STREAMINFO "
                         "MD5 (DF21 class B, D34): the decodable audio is not "
                         "the original. Refusing.")
    global FLAC_FALLBACKS
    FLAC_FALLBACKS += 1
    return np.frombuffer(raw, "<i2").astype("float32") / 32768.0, sr


def load_wav(path: Path, target_sr: int = 16000, _retries: int = 4) -> np.ndarray:
    import soundfile as sf
    # Bounded retry on TRANSIENT open failures only. libsndfile raises a bare
    # "System error" when Windows momentarily refuses the handle (AV scan, I/O
    # hiccup); a single one of these previously killed a 71k-clip eval sweep at
    # utt ~25,000 on a file that reads back fine. This does NOT change read
    # semantics: on success the bytes, dtype and resampling path are identical,
    # so score.py and score_onnx.py still share one preprocessing definition.
    # A genuinely unreadable file still raises, just after the last attempt.
    # A FLAC *decode* error (anything but SF_ERR_SYSTEM = 2) is deterministic,
    # not transient: go straight to the MD5-verified decoder instead of retrying.
    import time
    for attempt in range(_retries + 1):
        try:
            wave, sr = sf.read(str(path), dtype="float32", always_2d=False)
            break
        except sf.LibsndfileError as e:
            if str(path).lower().endswith(".flac") and getattr(e, "code", 2) != 2:
                wave, sr = _flac_md5_verified(Path(path))
                break
            if attempt == _retries:
                raise
            time.sleep(0.25 * (attempt + 1))
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

    from deploybench.data import native_conditions
    native = native_conditions(args.dataset) if args.dataset else None
    if native is not None and args.condition != "clean":
        raise SystemExit(
            f"{args.dataset} carries a native per-utterance condition that is "
            f"written to the `condition` column; refusing --condition "
            f"{args.condition!r} rather than overwrite it")

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
            cond = native[uid] if native is not None else args.condition
            w.writerow([uid, lab, sc, args.model, args.dataset or "adhoc",
                        args.precision, args.runtime, cond, args.seed])
    print(f"wrote {len(rows)} scores -> {out}")
    print(f"FLAC files decoded via the MD5-verified fallback (D34): {FLAC_FALLBACKS}")


if __name__ == "__main__":
    main()
