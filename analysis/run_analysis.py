"""Hypothesis-driven analysis: database.parquet -> tables + figure data.

    python analysis/run_analysis.py --db results/database.parquet --out analysis/out/
    python analysis/run_analysis.py --db <db> --out <dir> --selftest   # simulated db

Produces (CSV, one per paper table/figure):
  ttg_table.csv        TTG + FRR/FAR decomposition per model x condition x tau-policy
  qts_table.csv        QTS per model x precision x eval condition (H3)
  sensitivity.csv      safe-band width per model x condition
  h1_points.csv        oracle-EER vs TTG point cloud + spearman (exploratory)
  hypothesis_verdicts.md
Cohort discipline (methodology C1): rankings only within cohort.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deploybench import metrics as M
from deploybench.tau import calibrate_tau

COHORT = {"aasist": "cohort19", "aasist_l": "cohort19", "rawnet2": "cohort19",
          "rawgat_st": "cohort19", "lcnn_lfcc": "cohort19", "tssdnet": "cohort19",
          "wavlm_trunc": "cohort_ssl", "wavlm_mhfa": "cohort_ssl",
          "xlsr_aasist": "cohort_ssl", "dummy": "cohort19"}
def is_dev(ds):  # tau-source rows
    return ds.str.endswith("_dev")


def analyze(db: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    ttg_rows, qts_rows, sens_rows = [], [], []
    for (model, precision, runtime, seed), g in db.groupby(
            ["model", "precision", "runtime", "seed"]):
        dev = g[is_dev(g.dataset)]
        if dev.empty:
            continue
        taus = calibrate_tau(dev.score.values, dev.label.values)
        ev = g[~is_dev(g.dataset)].copy()
        ev["cell"] = ev.dataset + ":" + ev.condition
        for cond, gc in ev.groupby("cell"):
            s, l = gc.score.values, gc.label.values
            for policy, tau in taus.items():
                r = M.ttg(s, l, tau)
                ci = M.bootstrap_ci(s, l, lambda a, b: M.hter_at(tau, a, b),
                                    n_boot=200)
                ttg_rows.append(dict(model=model, cohort=COHORT.get(model, "?"),
                                     precision=precision, runtime=runtime,
                                     seed=seed, condition=cond, policy=policy,
                                     tau=tau, **r,
                                     hter_lo=ci["lo"], hter_hi=ci["hi"],
                                     oracle_eer=M.eer(s, l)[0]))
            sens_rows.append(dict(model=model, precision=precision,
                                  condition=cond, seed=seed,
                                  **M.threshold_sensitivity(s, l, taus["eer"])))
    ttg = pd.DataFrame(ttg_rows)
    ttg.to_csv(outdir / "ttg_table.csv", index=False)
    pd.DataFrame(sens_rows).to_csv(outdir / "sensitivity.csv", index=False)

    # ---- QTS (H3): fp32 tau vs same-cell quantized tau, on quantized scores
    for (model, runtime, seed), g in db.groupby(["model", "runtime", "seed"]):
        devs = {p: d for p, d in g[is_dev(g.dataset)].groupby("precision")}
        if "fp32" not in devs:
            continue
        tau32 = calibrate_tau(devs["fp32"].score.values,
                              devs["fp32"].label.values)["eer"]
        for prec in [p for p in devs if p != "fp32"]:
            tauq = calibrate_tau(devs[prec].score.values,
                                 devs[prec].label.values)["eer"]
            gq = g[(g.precision == prec) & (~is_dev(g.dataset))].copy()
            gq["cell"] = gq.dataset + ":" + gq.condition
            for cond, gc in gq.groupby("cell"):
                r = M.qts(gc.score.values, gc.label.values, tau32, tauq)
                qts_rows.append(dict(model=model, runtime=runtime, seed=seed,
                                     precision=prec, condition=cond, **r))
    qts = pd.DataFrame(qts_rows)
    qts.to_csv(outdir / "qts_table.csv", index=False)

    # ---- H1 point cloud (EXPLORATORY: effect size + CI only)
    h1 = None
    if not ttg.empty:
        pts = ttg[(ttg.policy == "eer")]
        h1 = pts[["model", "cohort", "precision", "condition", "seed",
                  "oracle_eer", "ttg"]].copy()
        h1.to_csv(outdir / "h1_points.csv", index=False)

    # ---- verdicts skeleton
    lines = ["# Hypothesis verdicts (auto-generated skeleton)\n"]
    if not ttg.empty:
        frr_dom = (ttg.frr_fixed > ttg.far_fixed).mean()
        lines.append(f"- H2 (FRR-dominant): FRR>FAR in {frr_dom:.0%} of "
                     f"(model x condition x policy) cells.")
    if not qts.empty:
        lines.append(f"- H3 (QTS): median QTS = {qts.qts.median():.4f}, "
                     f"max = {qts.qts.max():.4f} "
                     f"({(qts.qts > 0.01).mean():.0%} of cells > 1pt).")
    if h1 is not None and len(h1) > 10:
        sp = M.spearman_ci(h1.oracle_eer.values, h1.ttg.values, n_boot=200)
        lines.append(f"- H1 (exploratory): rho={sp['rho']:.2f} "
                     f"CI[{sp['lo']:.2f},{sp['hi']:.2f}] n={sp['n']} "
                     "(clustered by model — interpret with care; no confirm/refute).")
    (outdir / "hypothesis_verdicts.md").write_text("\n".join(lines) + "\n")
    print(f"analysis -> {outdir} ({len(ttg)} ttg rows, {len(qts)} qts rows)")
    return ttg, qts


def selftest_db() -> pd.DataFrame:
    """Simulated database with planted effects: shift on 'itw' (TTG>0),
    score rescale under int8 (QTS>0)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from deploybench.simulate import gaussian_scores
    rows = []
    for model, sep in [("goodmodel", 2.4), ("weakmodel", 1.2)]:
        for prec, (a, b) in [("fp32", (1, 0)), ("int8", (0.6, -0.4))]:
            for ds, cond, shift in [("asvspoof19_dev", "clean", 0),
                                     ("asvspoof19_eval", "clean", 0),
                                     ("itw", "clean", -1.2)]:
                s, l = gaussian_scores(1500, 1500, mu_b=sep / 2 + shift,
                                       mu_s=-sep / 2 + shift,
                                       seed=hash((model, prec, ds)) % 2**31)
                s = a * s + b
                for si, (sc, la) in enumerate(zip(s, l)):
                    rows.append(dict(utt_id=f"u{si}", label=int(la), score=float(sc),
                                     model=model, dataset=ds, precision=prec,
                                     runtime="ort", condition=cond, seed=0))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db")
    ap.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    df = selftest_db() if a.selftest else pd.read_parquet(a.db)
    analyze(df, Path(a.out))
