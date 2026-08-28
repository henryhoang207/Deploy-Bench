"""Materialise the parity manifest as PCM16 WAV for the Android app.

WHY THIS EXISTS: the corpora ship FLAC, but android/WavIo.kt is a minimal PCM16
WAV reader (no FLAC decoder on device, deliberately -- a decoder would be one
more uncontrolled difference in the P3 comparison). Stage P3 needs the SAME 1,000
clips readable on both sides, so they are transcoded once, here, on the desktop.

LOSSLESSNESS IS THE WHOLE POINT. 19LA FLAC is 16-bit, so this reads and writes
int16 and never touches float: no scaling, no dithering, no rounding. Every clip
is read back and compared sample-for-sample, and the script fails if a single
sample differs. If the transcode were lossy, a P3 decision flip could not be
attributed to the device.

The desktop reference side keeps reading the ORIGINAL FLAC via score.py -- that
is correct, and is exactly what makes the check meaningful: soundfile's int16
-> float32 conversion divides by 32768, and WavIo.kt's `v / 32768f` matches it
bit for bit, so both sides reach identical float samples by different routes.

    python scripts/make_device_wavs.py --manifest configs/parity_manifest_tssdnet.txt \
        --flac-dir %DATA_ROOT%/LA/ASVspoof2019_LA_eval/flac --out-dir device_wavs
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 16000


def read_manifest(path: Path) -> list:
    utts = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            utts.append(line.split()[0])
    return utts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--flac-dir", default=None,
                    help="defaults to $DATA_ROOT/LA/ASVspoof2019_LA_eval/flac")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    flac_dir = a.flac_dir
    if flac_dir is None:
        root = os.environ.get("DATA_ROOT")
        if not root:
            print("ERROR: pass --flac-dir or set DATA_ROOT", file=sys.stderr)
            return 2
        flac_dir = Path(root) / "LA" / "ASVspoof2019_LA_eval" / "flac"
    flac_dir = Path(flac_dir)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    utts = read_manifest(Path(a.manifest))
    if a.limit:
        utts = utts[:a.limit]
    print(f"manifest : {a.manifest}  ({len(utts)} clips)")
    print(f"flac dir : {flac_dir}")
    print(f"out dir  : {out_dir}")

    missing, bad_sr, bad_rt, total_bytes = [], [], [], 0
    for i, utt in enumerate(utts, 1):
        src = flac_dir / f"{utt}.flac"
        if not src.exists():
            missing.append(utt)
            continue
        # int16 in, int16 out -- no float ever enters the path.
        data, sr = sf.read(str(src), dtype="int16", always_2d=False)
        if sr != TARGET_SR:
            bad_sr.append((utt, sr))
            continue
        if data.ndim > 1:                       # 19LA is mono; be explicit anyway
            bad_sr.append((utt, f"channels={data.shape[1]}"))
            continue
        dst = out_dir / f"{utt}.wav"
        sf.write(str(dst), data, sr, subtype="PCM_16")

        back, sr2 = sf.read(str(dst), dtype="int16", always_2d=False)
        if sr2 != sr or back.shape != data.shape or not np.array_equal(back, data):
            bad_rt.append(utt)
        total_bytes += dst.stat().st_size
        if i % 200 == 0:
            print(f"  {i}/{len(utts)}")

    written = len(utts) - len(missing) - len(bad_sr)
    print(f"\nwritten          : {written}")
    print(f"missing flac     : {len(missing)}")
    print(f"wrong sr/channels: {len(bad_sr)}")
    print(f"lossy round-trips: {len(bad_rt)}")
    print(f"total bytes      : {total_bytes:,}")
    for label, rows in (("MISSING", missing), ("BAD SR/CH", bad_sr), ("LOSSY", bad_rt)):
        for r in rows[:10]:
            print(f"  {label}: {r}")

    if missing or bad_sr or bad_rt:
        print("\nFAIL -- the device set is not a faithful copy of the manifest.")
        return 1
    print("\nOK -- all clips transcoded losslessly (sample-exact round-trip).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
