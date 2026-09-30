"""H5, the confirmatory test, as specified in DEVIATIONS.md DEV-2 (committed before any
EnCodec audio existed).

    python analysis/h5_test.py [--dir results/phase_3_stress/h5] [--out results/analysis/h5_test.json]

H5: "EnCodec-6k stress increases FRR more than FAR relative to clean, for a majority of
the 4 ranked models." Per model at the frozen tau_eer: dFRR = FRR(encodec6k) - FRR(clean),
dFAR = FAR(encodec6k) - FAR(clean), statistic dFRR - dFAR. Class-stratified PAIRED bootstrap
(the same resampled clip ids in both conditions), B = 1000, seed 0, one-sided
p = P(dFRR - dFAR <= 0). Holm-Bonferroni in the H5 family, alpha 0.05; supported if >= 3 of 4.
Inputs: <dir>/scores_<model>_{clean,encodec6k}.csv (fp32 PyTorch via --stress-manifest).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
from deploybench.manifests import ids_digest, read_ids  # noqa: E402
from deploybench.metrics import holm_bonferroni  # noqa: E402

RANKED = ("aasist", "aasist_l", "rawnet2", "tssdnet")
B, SEED, ALPHA = 1000, 0, 0.05
PARITY = R / "configs" / "parity_manifest_tssdnet.txt"


def load(p: Path, model: str, cond: str, ids):
    rows = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        key = (r["model"], r["dataset"], r["precision"], r["runtime"], r["condition"])
        if key != (model, "asvspoof19_eval", "fp32", "pytorch", cond):
            raise SystemExit(f"{p}: row {r['utt_id']} is {key}")
        if r["utt_id"] in rows:
            raise SystemExit(f"{p}: duplicate {r['utt_id']}")
        rows[r["utt_id"]] = (int(r["label"]), float(r["score"]))
    if set(rows) != set(ids):
        raise SystemExit(f"{p}: covers {len(rows)} clips, not exactly the {len(ids)} parity ids")
    return np.array([rows[u][0] for u in ids]), np.array([rows[u][1] for u in ids])


def delta_stat(y, sc, se, tau, idx_b, idx_s):
    """dFRR - dFAR on the given bona fide / spoof index arrays (SPOOF if score < tau)."""
    dfrr = np.mean(se[idx_b] < tau) - np.mean(sc[idx_b] < tau)
    dfar = np.mean(se[idx_s] >= tau) - np.mean(sc[idx_s] >= tau)
    return dfrr, dfar


def paired_bootstrap(y, sc, se, tau, n_boot=B, seed=SEED):
    ib, isp = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for i in range(n_boot):
        rb = ib[rng.integers(0, len(ib), len(ib))]
        rs = isp[rng.integers(0, len(isp), len(isp))]
        dfrr, dfar = delta_stat(y, sc, se, tau, rb, rs)
        out[i] = dfrr - dfar
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(R / "results" / "phase_3_stress" / "h5"))
    ap.add_argument("--out", default=str(R / "results" / "analysis" / "h5_test.json"))
    a = ap.parse_args()
    ids = read_ids(PARITY)
    if not ids_digest(ids).startswith("2a3c8964"):
        raise SystemExit("parity manifest ids changed; DEV-2 fixed sha256 2a3c8964...")
    per, pvals = {}, {}
    for m in RANKED:
        yc, sc = load(Path(a.dir) / f"scores_{m}_clean.csv", m, "clean", ids)
        ye, se = load(Path(a.dir) / f"scores_{m}_encodec6k.csv", m, "encodec6k", ids)
        if not np.array_equal(yc, ye):
            raise SystemExit(f"{m}: labels differ between conditions")
        tau = json.loads((R / "results" / "phase_1_reproduction" / f"tau_{m}.json").read_text())["taus"]["eer"]
        ib, isp = np.where(yc == 1)[0], np.where(yc == 0)[0]
        dfrr, dfar = delta_stat(yc, sc, se, tau, ib, isp)
        d = paired_bootstrap(yc, sc, se, tau)
        pvals[m] = float(np.mean(d <= 0))
        per[m] = {"tau_eer": tau,
                  "frr_clean": float(np.mean(sc[ib] < tau)), "frr_encodec": float(np.mean(se[ib] < tau)),
                  "far_clean": float(np.mean(sc[isp] >= tau)), "far_encodec": float(np.mean(se[isp] >= tau)),
                  "dfrr": float(dfrr), "dfar": float(dfar), "stat": float(dfrr - dfar),
                  "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                  "p_one_sided": pvals[m]}
    reject = holm_bonferroni(pvals, ALPHA)
    n = sum(reject.values())
    out = {"hypothesis": "H5", "spec": "DEVIATIONS.md DEV-2", "B": B, "seed": SEED, "alpha": ALPHA,
           "per_model": {m: dict(per[m], holm_reject_h0=reject[m]) for m in RANKED},
           "n_models": n, "majority_required": 3, "h5_supported": n >= 3}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for m in RANKED:
        r = out["per_model"][m]
        print(f"{m:9s} FRR {r['frr_clean']*100:5.1f} -> {r['frr_encodec']*100:5.1f}  "
              f"FAR {r['far_clean']*100:5.1f} -> {r['far_encodec']*100:5.1f}  "
              f"dFRR-dFAR {r['stat']*100:+6.1f} pp  CI [{r['ci95'][0]*100:+.1f}, {r['ci95'][1]*100:+.1f}]  "
              f"p={r['p_one_sided']:.3f}  Holm reject: {r['holm_reject_h0']}")
    print(f"\ndFRR > dFAR (Holm, alpha {ALPHA}) in {n} of 4; majority needs 3 -> "
          f"H5 {'SUPPORTED' if out['h5_supported'] else 'NOT SUPPORTED'}")


if __name__ == "__main__":
    main()
