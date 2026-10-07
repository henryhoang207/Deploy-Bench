"""E6 (PREREGISTRATION_ROUND2.md Sec. 9): leading / trailing non-speech of ASVspoof 5 dev and eval with the DEV-8 VAD
(deploybench.vad.bounds, T = 35 dB), to check on the released audio that the organisers' post-processing equalised it
between the classes (database paper Sec. 4.2). Audio statistics only. CPU, run from the clone with envs/db310:

    python -B scripts/sol1/round2/r2_as5_silence.py --out results/round2/as5_silence.csv --procs 24

Columns: dataset, utt_id, label, condition, n_samples, lead35, trail35. A clip that fails to load is FATAL.
"""
import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, ".")


def work(item):
    from deploybench import vad
    from score import load_wav
    ds, uid, path, label, cond = item
    x = load_wav(Path(path))
    lead, trail = vad.bounds(x)
    return [ds, uid, label, cond, len(x), lead, trail]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=24)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists: refusing to overwrite")
    from deploybench.data import asvspoof5_condition, asvspoof5_rows, protocol_iter
    items = []
    for part in ("dev", "eval"):
        cond = {r["utt"]: asvspoof5_condition(r) for r in asvspoof5_rows(part)}
        for u, p, lab in protocol_iter(f"asvspoof5_{part}"):
            items.append((f"asvspoof5_{part}", u, str(p), int(lab), cond[u]))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".partial")
    t0 = time.time()
    with Pool(a.procs) as pool, open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "utt_id", "label", "condition", "n_samples", "lead35", "trail35"])
        for i, row in enumerate(pool.imap(work, items, chunksize=256), 1):
            w.writerow(row)
            if i % 100000 == 0:
                print(f"  {i}/{len(items)}  {time.time() - t0:.0f}s", flush=True)
    os.replace(tmp, out)
    print(f"AS5_SILENCE_DONE {len(items)} rows -> {out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
