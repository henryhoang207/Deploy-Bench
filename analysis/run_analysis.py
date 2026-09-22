"""Hypothesis-driven analysis: database.parquet -> tables + figure data.

    python analysis/run_analysis.py --db results/database.parquet --out analysis/out/
    python analysis/run_analysis.py --db <db> --out <dir> --selftest   # simulated db

Produces (CSV, one per paper table/figure):
  ttg_table.csv        TTG + FRR/FAR decomposition per model x condition x tau-policy
  qts_table.csv        QTS per model x precision x eval condition (H3)
  sensitivity.csv      safe-band width per model x condition
  h1_points.csv        oracle-EER vs TTG point cloud + spearman (exploratory)
  hypothesis_verdicts.md
Ranking discipline (methodology C2, D16/D30): the roster is single-cohort as of
D30, but xlsr_aasist remains an UNRANKED server reference -- exclude it from any
ordering claim and from "majority of models" counts (see UNRANKED below).

Thresholds (review finding B3, fixed 2026-09-22): every tau comes from the
frozen tau_<model>.json that analysis/repro_report.py wrote, never from
re-calibration here. Drift guard: the dev score file the freeze used must still
hash to the recorded sha256 and reproduce the frozen values exactly. ASVspoof19
LA dev is the only dev set accepted (D30). The one re-calibration left is the QTS
counterfactual: tau re-derived on a QUANTIZED model's own dev scores, compared
with the frozen fp32 tau. It is never used as a deployed threshold.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deploybench import metrics as M
from deploybench.tau import calibrate_tau

# D30 froze the roster at 5, all Cohort-19, so the two-cohort ranking stratum no
# longer exists. What survives is the RANKING rule: xlsr_aasist is an unranked
# server reference (D16) and must not appear in any ordering claim.
COHORT = {"aasist": "cohort19", "aasist_l": "cohort19", "rawnet2": "cohort19",
          "tssdnet": "cohort19", "xlsr_aasist": "cohort19", "dummy": "cohort19"}
UNRANKED = {"xlsr_aasist"}


DEV_SET = "asvspoof19_dev"         # the ONLY tau surface (D16 as collapsed by D30)
TAU_DIR = Path(__file__).resolve().parent.parent / "results" / "phase_1_reproduction"


def is_dev(ds):
    """tau-source rows: exactly ASVspoof19 LA dev. Any other *_dev set is
    refused in load_frozen_taus, so two dev surfaces can never be pooled."""
    return ds == DEV_SET


def load_frozen_taus(db: pd.DataFrame, tau_dir: Path) -> dict:
    """{model: {eer, frr1, frr5}} from tau_<model>.json (B3). Drift guard: the
    dev score file the freeze was calibrated on (scores_dev_<model>.csv beside
    the tau file, stored byte-exact in git) must still hash to the recorded
    dev_scores_sha256 AND reproduce the frozen taus exactly. The guard needs
    nothing from the database, whose grid cells are ort/executorch only."""
    import hashlib
    import json
    other = sorted(set(db.dataset[db.dataset.str.endswith("_dev")]) - {DEV_SET})
    if other:
        raise SystemExit(f"dev sets other than {DEV_SET} in the database: {other}. "
                         "D30 allows one tau surface; refusing to pool.")
    taus = {}
    for model in sorted(db.model.unique()):
        p = Path(tau_dir) / f"tau_{model}.json"
        src = Path(tau_dir) / f"scores_dev_{model}.csv"
        if not p.exists() or not src.exists():
            raise SystemExit(f"{p} or {src} missing: no verifiable frozen tau for "
                             f"{model}. Freeze it with analysis/repro_report.py first.")
        frozen = json.loads(p.read_text(encoding="utf-8-sig"))
        raw = src.read_bytes()
        if hashlib.sha256(raw).hexdigest() != frozen["dev_scores_sha256"]:
            raise SystemExit(f"{src} no longer hashes to the dev_scores_sha256 "
                             f"recorded at freeze time in {p}. Refusing.")
        import io
        ref = pd.read_csv(io.BytesIO(raw))
        if len(ref) != frozen["n_dev"] or set(ref.dataset) != {DEV_SET}:
            raise SystemExit(f"{src}: {len(ref)} rows on {sorted(set(ref.dataset))}, "
                             f"freeze says {frozen['n_dev']} on {DEV_SET}. Refusing.")
        again = calibrate_tau(ref.score.values, ref.label.values)
        drift = {k: (v, again.get(k)) for k, v in frozen["taus"].items()
                 if again.get(k) is None or abs(v - again[k]) > 1e-12}
        if drift:
            raise SystemExit(f"{model}: frozen tau drifts from the database's dev "
                             f"rows {drift}. Investigate; never re-freeze silently.")
        taus[model] = frozen["taus"]
    return taus


def analyze(db: pd.DataFrame, outdir: Path, tau_dir: Path = TAU_DIR):
    outdir.mkdir(parents=True, exist_ok=True)
    frozen = load_frozen_taus(db, tau_dir)
    ttg_rows, qts_rows, sens_rows = [], [], []
    for (model, precision, runtime, seed), g in db.groupby(
            ["model", "precision", "runtime", "seed"]):
        taus = frozen[model]              # the deployed thresholds, every cell
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

    # ---- QTS (H3): frozen fp32 tau vs the COUNTERFACTUAL tau re-derived on the
    # quantized model's own dev scores (what re-calibrating would have given).
    # tau32 is the deployed threshold; tauq is never deployed.
    for (model, runtime, seed, prec), g in db.groupby(
            ["model", "runtime", "seed", "precision"]):
        dev = g[is_dev(g.dataset)]
        if prec == "fp32" or dev.empty:
            continue
        tau32 = frozen[model]["eer"]
        tauq = calibrate_tau(dev.score.values, dev.label.values)["eer"]
        gq = g[~is_dev(g.dataset)].copy()
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
    import zlib
    from deploybench.simulate import gaussian_scores
    rows = []
    for model, sep in [("goodmodel", 2.4), ("weakmodel", 1.2)]:
        # fp32 is the PyTorch reference (the rows a tau freeze uses); int8 is ORT
        for prec, runtime, (a, b) in [("fp32", "pytorch", (1, 0)),
                                      ("int8", "ort", (0.6, -0.4))]:
            for ds, cond, shift in [("asvspoof19_dev", "clean", 0),
                                     ("asvspoof19_eval", "clean", 0),
                                     ("itw", "clean", -1.2)]:
                # crc32, not hash(): str hashing is salted per process
                s, l = gaussian_scores(1500, 1500, mu_b=sep / 2 + shift,
                                       mu_s=-sep / 2 + shift,
                                       seed=zlib.crc32(f"{model}|{prec}|{ds}".encode()))
                s = a * s + b
                for si, (sc, la) in enumerate(zip(s, l)):
                    rows.append(dict(utt_id=f"u{si}", label=int(la), score=float(sc),
                                     model=model, dataset=ds, precision=prec,
                                     runtime=runtime, condition=cond, seed=0))
    return pd.DataFrame(rows)


def selftest_freeze(db: pd.DataFrame, tau_dir: Path) -> Path:
    """Write scores_dev_<model>.csv + tau_<model>.json for a simulated db the
    way repro_report.py freezes them (tau from the file, file hash recorded)."""
    import hashlib
    import io
    import json
    tau_dir.mkdir(parents=True, exist_ok=True)
    for model, g in db.groupby("model"):
        ref = g[is_dev(g.dataset) & (g.precision == "fp32") & (g.runtime == "pytorch")]
        src = tau_dir / f"scores_dev_{model}.csv"
        ref.to_csv(src, index=False)
        back = pd.read_csv(io.BytesIO(src.read_bytes()))
        (tau_dir / f"tau_{model}.json").write_text(json.dumps(
            {"model": model, "n_dev": len(back),
             "dev_scores_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
             "taus": calibrate_tau(back.score.values, back.label.values)}))
    return tau_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau-dir", default=str(TAU_DIR),
                    help="directory holding the frozen tau_<model>.json files")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        df = selftest_db()
        analyze(df, Path(a.out), selftest_freeze(df, Path(a.out) / "taus"))
    else:
        analyze(pd.read_parquet(a.db), Path(a.out), Path(a.tau_dir))
