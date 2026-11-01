"""How much of TTG is the in-sample optimism of the oracle? (DEV-10; reviewer question 2)

    python analysis/review_r1/run_cv.py      # -> results/review_r1/cv.json (about 7 min)

For every detector x target at the frozen tau_EER: the registered in-sample TTG; the TTG against a leave-one-speaker-out
oracle (each speaker decided by the threshold that minimises HTER on all other speakers); against a speaker-disjoint
5-fold oracle (20 random fold assignments); and against a clip-level stratified 5-fold oracle (20 repeats), which keeps
speakers on both sides and so measures pure sampling optimism. The in-sample TTG is biased up, the cross-validated TTGs
are biased down (a threshold tuned on finitely many other speakers), so the two bracket the gap to the best threshold.
"""
import sys
import time

import numpy as np

from rvlib import NAMES, R, ROUND1, DETECTORS_R2, oracle, hter, targets_for, write
from cvlib import Surface, random_folds

REPS = 20
CELLS = {}
META = {"script": "analysis/review_r1/run_cv.py",
        "speaker_5fold": {"assignments": REPS, "rng": "numpy.random.default_rng(0), restarted for every cell"},
        "clip_5fold": {"repeats": REPS, "rng": "numpy.random.default_rng(1000 + r) for repeat r = 0..19",
                       "stratified": "by class"},
        "grid": "every unique target score plus one above the maximum; a quantile grid of min(80000, 1.2e7 / G) "
                "points where there are more unique scores"}
t0 = time.time()


def cells():
    for m in ROUND1 + DETECTORS_R2:
        for surf in targets_for(m):
            yield m, 0, surf
    for k in R.RUN_SEEDS + (R.PIPELINE_SEED,):
        yield "aasist_l_r2", k, "la19_eval"


def run_cell(m, seed, surf):
    ids, y, s = R.load(m, surf, seed)
    tau = R.taus(m, seed)["eer"]
    g, names = R.clusters(surf, ids)
    G = len(names)
    sf = Surface(s, y, g, G, max_m=min(80_000, int(1.2e7 / G)))
    h_star = hter(s, y, tau)
    h_in, _ = oracle(s, y)
    ones = np.ones((1, G))
    h_grid_in = float(sf.oracle(ones)[0])
    h_loso, fr_l, fa_l = sf.loso_oracle()
    rng = np.random.default_rng(0)
    spk5 = []
    for _ in range(REPS):
        f = random_folds(G, 5, rng)
        spk5.append(float(sf.cv_oracle(ones, f, 5)[0][0]))
    clip5 = []
    for r in range(REPS):
        rr = np.random.default_rng(1000 + r)
        pseudo = np.empty(len(y), dtype=int)
        for cls in (0, 1):
            idx = np.nonzero(y == cls)[0]
            pseudo[idx] = rr.permutation(np.arange(len(idx)) % 5)
        sc = Surface(s, y, pseudo, 5, max_m=80_000)
        clip5.append(float(sc.cv_oracle(np.ones((1, 5)), np.arange(5), 5)[0][0]))
    spk5, clip5 = np.array(spk5), np.array(clip5)
    cell = {"model": m, "seed": seed, "target": surf, "G": G, "n": int(len(y)), "grid_exact": sf.exact,
            "hter_star": h_star, "oracle_in": h_in, "oracle_in_grid": h_grid_in,
            "ttg_in": h_star - h_in, "ttg_loso": h_star - h_loso, "loso_frr": fr_l, "loso_far": fa_l,
            "ttg_spk5_mean": float(h_star - spk5.mean()), "ttg_spk5_range": [float(h_star - spk5.max()), float(h_star - spk5.min())],
            "ttg_clip5_mean": float(h_star - clip5.mean()),
            "oracle_loso": h_loso, "oracle_spk5_mean": float(spk5.mean()), "oracle_clip5_mean": float(clip5.mean())}
    return cell


print(f"{'model':12s} {'tgt':10s} {'G':>4s} {'HTER*':>6s} {'orIn':>6s} {'TTGin':>6s} {'TTGcl5':>6s} {'TTGloso':>7s} "
      f"{'TTGspk5 mean [range]':>24s} {'share in/loso':>13s}")
for m, seed, surf in cells():
    c = run_cell(m, seed, surf)
    key = f"{m}|s{seed}|{surf}"
    CELLS[key] = c
    lab = NAMES.get(m, f"AASIST-L r{seed}")
    print(f"{lab:12s} {surf[:10]:10s} {c['G']:4d} {100 * c['hter_star']:6.2f} {100 * c['oracle_in']:6.2f} "
          f"{100 * c['ttg_in']:6.2f} {100 * c['ttg_clip5_mean']:6.2f} {100 * c['ttg_loso']:7.2f} "
          f"{100 * c['ttg_spk5_mean']:8.2f} [{100 * c['ttg_spk5_range'][0]:5.2f}, {100 * c['ttg_spk5_range'][1]:5.2f}] "
          f"{c['ttg_in'] / c['hter_star']:5.2f}/{c['ttg_loso'] / c['hter_star']:5.2f}"
          f"{'' if c['grid_exact'] else '  (grid)'}  [{time.time() - t0:.0f}s]", flush=True)
    if abs(c["oracle_in_grid"] - c["oracle_in"]) > (1e-12 if c["grid_exact"] else 2e-4):
        sys.exit(f"grid oracle {c['oracle_in_grid']} differs from exact {c['oracle_in']} for {key}")

write("cv", META, CELLS)
print(f"done in {time.time() - t0:.0f}s")
