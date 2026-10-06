"""DEV-8 F1: leading / trailing silence statistics for LA19 dev, LA19 eval and In-the-Wild (CPU only).

Runs on a GPU server from a scratch directory. The harness is imported READ-ONLY from the deploybench clone
(bytecode writing is off; nothing in the clone is created or modified).

    source $SCRATCH/env.sh      # DATA_ROOT, db310 env
    nice -n 10 python -B f_vad_stats.py --out $SCRATCH/dfv_rev/vad/vad_stats.csv --procs 16

Energy VAD, fixed in DEVIATIONS.md DEV-8 before any run:
  16 kHz after the harness's own load_wav; 25 ms frames (400 samples), 10 ms hop (160), full frames only;
  E_t = 10 log10(mean(x^2) + 1e-12) dB; a frame is ACTIVE if E_t >= max_t E_t - T dB, T in {25, 35, 45};
  leading silence  = start of the first active frame (samples);
  trailing silence = samples after the end of the last active frame (end of frame = start + 400).
  A clip shorter than one frame has lead = trail = 0.
Output columns: dataset, utt_id, label, n_samples, lead25, trail25, lead35, trail35, lead45, trail45.
A clip that fails to load is FATAL (no silent skipping).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.dont_write_bytecode = True
REPO = os.environ.get("DEPLOYBENCH", str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

FRAME, HOP = 400, 160
THRS = (25, 35, 45)
DATASETS = ("asvspoof19_dev", "asvspoof19_eval", "itw")


def frame_energy_db(x: np.ndarray):
    n = len(x)
    if n < FRAME:
        return None
    c = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    nf = 1 + (n - FRAME) // HOP
    i = np.arange(nf) * HOP
    e = (c[i + FRAME] - c[i]) / FRAME
    return 10.0 * np.log10(e + 1e-12)


def bounds(e, n: int, T: float):
    """(lead, trail) in samples for threshold T dB."""
    if e is None:
        return 0, 0
    act = np.nonzero(e >= e.max() - T)[0]
    return int(act[0] * HOP), int(n - (act[-1] * HOP + FRAME))


def work(item):
    ds, uid, path, label = item
    from score import load_wav            # the harness reader (incl. resample to 16 kHz)
    x = load_wav(Path(path))
    e = frame_energy_db(x)
    row = [ds, uid, label, len(x)]
    for T in THRS:
        row.extend(bounds(e, len(x), T))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="smoke test only")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists: refusing to overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    from deploybench.data import protocol_iter
    items = []
    for ds in DATASETS:
        k = 0
        for uid, path, label in protocol_iter(ds):
            items.append((ds, uid, str(path), int(label)))
            k += 1
            if a.limit and k >= a.limit:
                break
        print(f"{ds}: {k} clips", flush=True)
    t0 = time.time()
    tmp = out.with_suffix(".partial")
    with Pool(a.procs) as pool, open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "utt_id", "label", "n_samples"] +
                   [f"{k}{T}" for T in THRS for k in ("lead", "trail")])
        for i, row in enumerate(pool.imap(work, items, chunksize=64), 1):
            w.writerow(row)
            if i % 10000 == 0:
                el = time.time() - t0
                print(f"  {i}/{len(items)}  {el:.0f}s  {i / el:.0f} clips/s", flush=True)
    os.replace(tmp, out)
    print(f"wrote {len(items)} rows -> {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
