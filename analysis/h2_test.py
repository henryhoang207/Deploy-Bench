"""H2, the registered confirmatory test (PREREGISTRATION.md; methodology Part C).

    python analysis/h2_test.py [--out results/analysis/h2_test.json]

H2: "On In-the-Wild at tau* (EER policy), FRR > FAR for the majority of the 4 ranked
models; family test: per-model paired bootstrap FRR vs FAR." Methodology Part C:
1,000 bootstrap resamples, Holm-Bonferroni within the confirmatory family, alpha 0.05,
"majority" = majority of the 4 ranked models (>= 3; XLS-R excluded).

Details the registration leaves open, fixed here by reusing the repo's existing
deploybench.metrics.paired_bootstrap conventions (not tuned to these data):
  - resampling is stratified by class (bona fide and spoof resampled separately, each
    to its own size), so every resample has the original class counts;
  - FRR and FAR are computed on the SAME resample ("paired");
  - one-sided p = fraction of resamples with FRR - FAR <= 0 (H0: FRR <= FAR);
  - numpy default_rng(seed=0), B = 1000;
  - tau* = each model's frozen tau_eer from 19LA dev (tau_<model>.json); never re-tuned.
Disclosure: the descriptive FRR/FAR on ITW (analysis/itw_fp32_descriptives.py) were
seen before this script was written.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
from deploybench.metrics import far_at, frr_at, holm_bonferroni  # noqa: E402

RANKED = ("aasist", "aasist_l", "rawnet2", "tssdnet")
B, SEED, ALPHA = 1000, 0, 0.05


def frr_far_bootstrap(scores, labels, tau, n_boot=B, seed=SEED):
    """Vectorised class-stratified bootstrap of FRR - FAR at a fixed tau.
    SPOOF if score < tau: FRR = P(bona fide score < tau), FAR = P(spoof score >= tau)."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    rej_b = (s[y == 1] < tau).astype(np.int64)      # bona fide wrongly rejected
    acc_s = (s[y == 0] >= tau).astype(np.int64)     # spoof wrongly accepted
    rng = np.random.default_rng(seed)
    nb, ns = len(rej_b), len(acc_s)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        frr = rej_b[rng.integers(0, nb, nb)].mean()
        far = acc_s[rng.integers(0, ns, ns)].mean()
        diffs[i] = frr - far
    return diffs


def load(model):
    p = R / "results" / "phase_2_conversion_app" / "main_matrix" / model / "scores_fp32_pytorch_itw.csv"
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    bad = {(r["model"], r["dataset"], r["precision"], r["runtime"]) for r in rows} - \
        {(model, "itw", "fp32", "pytorch")}
    if bad or len(rows) != 31779:
        raise SystemExit(f"{p}: expected 31,779 {model}/itw/fp32/pytorch rows, got {len(rows)} {bad}")
    return (np.array([float(r["score"]) for r in rows]),
            np.array([int(r["label"]) for r in rows]), p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(R / "results" / "analysis" / "h2_test.json"))
    a = ap.parse_args()
    per, pvals = {}, {}
    for m in RANKED:
        s, y, p = load(m)
        tau = json.loads((R / "results" / "phase_1_reproduction" / f"tau_{m}.json").read_text())["taus"]["eer"]
        frr, far = frr_at(tau, s, y), far_at(tau, s, y)
        d = frr_far_bootstrap(s, y, tau)
        pvals[m] = float(np.mean(d <= 0))
        per[m] = {"tau_eer": tau, "frr": frr, "far": far, "delta": frr - far,
                  "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                  "p_one_sided": pvals[m], "n_bona": int(y.sum()), "n_spoof": int((y == 0).sum()),
                  "scores": str(p.relative_to(R))}
    reject = holm_bonferroni(pvals, ALPHA)
    n_sup = sum(reject.values())
    out = {"hypothesis": "H2", "family": list(RANKED), "B": B, "seed": SEED, "alpha": ALPHA,
           "correction": "Holm-Bonferroni within the H2 family",
           "per_model": {m: dict(per[m], holm_reject_h0=reject[m]) for m in RANKED},
           "n_models_frr_gt_far": n_sup, "majority_required": 3,
           "h2_supported": n_sup >= 3}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for m in RANKED:
        r = out["per_model"][m]
        print(f"{m:9s} FRR {r['frr']*100:6.2f}%  FAR {r['far']*100:6.2f}%  "
              f"FRR-FAR {r['delta']*100:+6.2f} pp  95% CI [{r['ci95'][0]*100:+.2f}, {r['ci95'][1]*100:+.2f}]  "
              f"p={r['p_one_sided']:.3f}  Holm reject H0: {r['holm_reject_h0']}")
    print(f"\nFRR > FAR (Holm, alpha {ALPHA}) in {n_sup} of 4 ranked models; "
          f"majority needs 3 -> H2 {'SUPPORTED' if out['h2_supported'] else 'NOT SUPPORTED'}")


if __name__ == "__main__":
    main()
