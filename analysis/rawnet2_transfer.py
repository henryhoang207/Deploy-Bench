"""Usage: python analysis/rawnet2_transfer.py

RawNet2 fp32 vs canonical int8 (r09 seed 2, int8-mixed K=16) on DF21 verified and ITW.
Metrics at the FROZEN 19LA-dev taus (no re-calibration), plus per-codec DF21 and
the check that the fresh fp32 DF21 clean scores equal the D33 gate scores."""
import csv
import json
import sys
from collections import defaultdict

import numpy as np

from pathlib import Path

R = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, R)
from deploybench.metrics import eer, far_at, frr_at, hter_at, ttg  # noqa: E402

TAU = json.load(open(f"{R}/results/phase_1_reproduction/tau_rawnet2.json"))["taus"]
MM = f"{R}/results/phase_2_conversion_app/main_matrix/rawnet2"
CS = f"{R}/results/phase_2_conversion_app/conversion_sensitivity/rawnet2/r09_seed2"


def load(p):
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    return {r["utt_id"]: (int(r["label"]), float(r["score"]), r["condition"]) for r in rows}, rows[0]


def block(name, f, q):
    assert set(f) == set(q), f"{name}: fp32 and int8 cover different trials"
    ids = sorted(f)
    y = np.array([f[u][0] for u in ids]); sf = np.array([f[u][1] for u in ids])
    sq = np.array([q[u][1] for u in ids])
    ef, eq = eer(sf, y)[0], eer(sq, y)[0]
    print(f"\n## {name}: n={len(ids)} (bona fide {int(y.sum())})")
    print(f"EER fp32 {ef*100:.3f}%  int8 {eq*100:.3f}%  dEER {(eq-ef)*100:+.3f} pp")
    for k in ("eer", "frr1", "frr5"):
        t = TAU[k]
        tf, tq = ttg(sf, y, t), ttg(sq, y, t)
        flips = int(((sf >= t) != (sq >= t)).sum())
        print(f"tau_{k:4s} fp32 HTER {hter_at(t,sf,y)*100:6.2f}% (FRR {frr_at(t,sf,y)*100:6.2f} FAR {far_at(t,sf,y)*100:6.2f}) TTG {tf['ttg']*100:+6.2f} | "
              f"int8 HTER {hter_at(t,sq,y)*100:6.2f}% (FRR {frr_at(t,sq,y)*100:6.2f} FAR {far_at(t,sq,y)*100:6.2f}) TTG {tq['ttg']*100:+6.2f} | flips {flips}")
    return ids, y, sf, sq


def main():
    for ds, fp, qp in (("ITW", f"{MM}/scores_fp32_pytorch_itw.csv", f"{CS}/scores_itw.csv"),
                       ("DF21 verified", f"{MM}/scores_fp32_pytorch_df21_verified.csv",
                        f"{CS}/scores_df21_verified.csv")):
        f, h = load(fp); q, hq = load(qp)
        print(f"\n[{ds}] fp32 file: {h['precision']}/{h['runtime']}  int8 file: {hq['precision']}/{hq['runtime']}")
        block(ds, f, q)
        if ds.startswith("DF21"):
            gate, _ = load(f"{R}/results/phase_1_reproduction/scores_df21clean_rawnet2.csv")
            d = max(abs(f[u][1] - gate[u][1]) for u in gate)
            print(f"fresh fp32 vs D33 gate scores on the {len(gate)} clean trials: max |d| {d:.2e}")
            block("DF21 clean only (sensitivity)", {u: f[u] for u in gate}, {u: q[u] for u in gate})
            by = defaultdict(list)
            for u in f:
                by[f[u][2]].append(u)
            print("\nper codec (verified surface): EER fp32 -> int8, HTER@tau_eer fp32 -> int8")
            for c in sorted(by):
                ids = by[c]; y = np.array([f[u][0] for u in ids])
                sf = np.array([f[u][1] for u in ids]); sq = np.array([q[u][1] for u in ids])
                t = TAU["eer"]
                print(f"  {c:10s} n={len(ids):6d}  EER {eer(sf,y)[0]*100:6.2f} -> {eer(sq,y)[0]*100:6.2f}  "
                      f"HTER {hter_at(t,sf,y)*100:6.2f} -> {hter_at(t,sq,y)*100:6.2f}")


if __name__ == "__main__":
    main()
