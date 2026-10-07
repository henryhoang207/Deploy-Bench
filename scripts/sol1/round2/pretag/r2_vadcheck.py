"""Pre-tag check: the harness VAD (deploybench.vad.bounds) reproduces DEV-8's recorded T = 35 dB bounds
(dfv_rev/vad/vad_stats.csv, from f_vad_stats.py) for every LA19 dev, LA19 eval and ITW clip. Audio statistics only.
Run from the deploybench clone with envs/db310:  python -B /mnt/data8tb/henry/xfer/r2_vadcheck.py"""
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, ".")
STATS = "/mnt/data8tb/henry/dfv_rev/vad/vad_stats.csv"


def work(item):
    from deploybench import vad
    from score import load_wav
    ds, uid, path, n0, lead, trail = item
    x = load_wav(Path(path))
    lb = vad.bounds(x)
    return (ds, uid, len(x) == n0, lb == (lead, trail), lb, (lead, trail))


def main():
    from deploybench.data import protocol_iter
    paths = {}
    for ds in ("asvspoof19_dev", "asvspoof19_eval", "itw"):
        for u, p, _ in protocol_iter(ds):
            paths[(ds, u)] = str(p)
    items = []
    with open(STATS, newline="") as f:
        for r in csv.DictReader(f):
            items.append((r["dataset"], r["utt_id"], paths[(r["dataset"], r["utt_id"])], int(r["n_samples"]),
                          int(r["lead35"]), int(r["trail35"])))
    bad_len = bad = 0
    ex = []
    with Pool(24) as pool:
        for ds, uid, ok_n, ok_b, got, want in pool.imap_unordered(work, items, chunksize=256):
            bad_len += not ok_n
            if not ok_b:
                bad += 1
                if len(ex) < 5:
                    ex.append((ds, uid, got, want))
    print(f"clips {len(items)}: length mismatches {bad_len}, bound mismatches {bad}; examples {ex}")
    print("VADCHECK_" + ("PASS" if bad == 0 and bad_len == 0 else "FAIL"))


if __name__ == "__main__":
    main()
