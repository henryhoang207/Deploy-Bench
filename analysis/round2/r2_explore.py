"""Round-2 exploratory analyses E1-E10 (PREREGISTRATION_ROUND2.md Sec. 9). Effect sizes and intervals only.

    python analysis/round2/r2_explore.py [--only E1,E7] [--b 1000]   # -> results/round2/analysis/r2_explore.{json,txt}

Every section skips (and reports) a model or surface whose scores or taus do not exist yet, so it can be run as data
arrives; the final run is the one reported. Definitions follow DEV-8 (DFV_Conference/revision/analysis):
  miss  = share of target clips between tau* and the target-oracle threshold (decisions that differ)
  band  = share of target clips in the threshold range where HTER stays within 1 pp of its minimum
  tie interval = DEV-8 B3 (r2lib.tie_interval) on the model's source dev
  window = bona fide priors p with C(p) = [p FRR + (1 - p) FAR] / min(p, 1 - p) < 1 (DEV-8 E1)
E9 calibrates on source dev with an exact, unpenalised logistic regression (DEV-9 amendment 2), which learns the dev
class prior; the prior log-odds of the dev set is subtracted so that the calibrated score is a log-likelihood ratio before
the Bayes threshold -logit(pi) is applied.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r2lib as R  # noqa: E402
from deploybench.metrics import det_curve, eer, far_at, frr_at  # noqa: E402
from deploybench.tau import tau_at_far  # noqa: E402

OUT = []
SURFACES = ("la19_eval", "itw", "df21_verified", "mlaad5", "as5_eval")
LOGIT_GRID = np.arange(-4.0, 4.0001, 0.25)
BUDGETS = (0.10, 0.05, 0.20)


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def have(model, surface, seed=0):
    try:
        R.load(model, surface, seed)
        R.taus(model, seed)
        return True
    except SystemExit:
        return False


def source_dev(model):
    from models.base import SPECS
    return "as5_dev" if model in SPECS and SPECS[model].cohort == "cohort_as5" else "la19_dev"


# --------------------------------------------------------------------------- building blocks (unit-tested)
def miss_band(s, y, tau):
    """DEV-8 D: rank-based miss and 1-pp band at tau, pooled and per class; also tau_oracle."""
    tg, fr, fa = det_curve(s, y)
    hh = 0.5 * (fr + fa)
    hmin = float(hh.min())
    t_or = float(tg[int(np.argmin(hh))])
    okb = tg[hh <= hmin + 0.01]
    blo, bhi = float(okb.min()), float(okb.max())
    isb = y == 1
    inb = (s >= blo) & (s < bhi)
    lo, hi = sorted((tau, t_or))
    sel = (s >= lo) & (s < hi)
    return {"tau_oracle": t_or,
            "band": {"pooled": float(inb.mean()), "bona": float(inb[isb].mean()), "spoof": float(inb[~isb].mean())},
            "miss": {"pooled": float(sel.mean()), "bona": float(sel[isb].mean()), "spoof": float(sel[~isb].mean())}}


def three_zone_edges(pooled_scores, tau, budget):
    """Review band of `budget` of the pooled scores, centred in rank on tau: edges at the quantiles F(tau) -/+ budget/2."""
    s = np.asarray(pooled_scores, float)
    q = float(np.mean(s < tau))
    return float(np.quantile(s, max(0.0, q - budget / 2))), float(np.quantile(s, min(1.0, q + budget / 2)))


def three_zone_apply(s, y, lo, hi):
    """score < lo -> spoof (auto-reject); score >= hi -> bona fide (auto-accept); otherwise review."""
    isb = y == 1
    rev = (s >= lo) & (s < hi)
    return {"review_pooled": float(rev.mean()), "review_bona": float(rev[isb].mean()),
            "review_spoof": float(rev[~isb].mean()),
            "bona_auto_rejected": float((s[isb] < lo).mean()), "spoof_auto_accepted": float((s[~isb] >= hi).mean())}


def bayes_thresholds(s_dev, y_dev):
    """Logistic calibration on dev -> raw-score Bayes threshold for each logit(prior) in LOGIT_GRID (None if the fitted
    slope is not positive). Calibrated LLR(s) = a s + b - logit(pi_dev); decide bona fide if LLR >= -logit(pi).
    DEV-9 amendment 2: the exact unpenalised maximum-likelihood fit (scikit-learn), not deploybench.metrics.platt_fit,
    whose fixed-step gradient descent does not converge in its 200 iterations and settles on a wrong fit at large score
    scales (tests/test_round2.py)."""
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e12, max_iter=100_000, tol=1e-10).fit(np.asarray(s_dev, float)[:, None],
                                                                      np.asarray(y_dev, int))
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    pi_dev = float(np.mean(y_dev))
    lo_dev = np.log(pi_dev / (1 - pi_dev))
    if not a > 0:
        return a, b, None
    return a, b, {float(L): float((-L + lo_dev - b) / a) for L in LOGIT_GRID}


def norm_cost(p, frr, far):
    return (p * frr + (1 - p) * far) / min(p, 1 - p)


def window(frr, far):
    lo = far / (far + 1.0 - frr) if (far + 1.0 - frr) > 0 else 1.0
    hi = (1.0 - far) / (1.0 - far + frr) if (1.0 - far + frr) > 0 else 0.0
    return (float(lo), float(hi)) if hi > lo else None


# --------------------------------------------------------------------------- E1 / E8 / E10
def e1(models, B):
    out = {}
    say("== E1 operating behaviour (five policies; TTG with a speaker-cluster 95% interval, B = %d) + E8 + E10" % B)
    for m in models:
        for surf in SURFACES:
            if surf == "as5_eval" and m != "as5_aasist":
                continue
            if not have(m, surf):
                continue
            ids, y, s = R.load(m, surf)
            t = R.taus(m)
            _, yd, sd = R.load(m, source_dev(m))
            lo_t, hi_t = R.tie_interval(sd, yd, t["eer"])
            g, names = R.clusters(surf, ids)
            C = R.cluster_counts(len(names), B, 0)
            h_or = R.Oracle(s, y, g).run(C)
            cell = {"n": int(len(y)), "eer": eer(s, y)[0], "clusters": len(names), "policies": {},
                    "tie": {"tau": [lo_t, hi_t], "frr": [frr_at(lo_t, s, y), frr_at(hi_t, s, y)],
                            "far": [far_at(lo_t, s, y), far_at(hi_t, s, y)]}}
            for pol in R.POLICIES5:
                p = R.point(s, y, t[pol])
                fr_r, fa_r = R.cluster_rates(s, y, g, len(names), t[pol], C)
                ttg_r = 0.5 * (fr_r + fa_r) - h_or
                mb = miss_band(s, y, t[pol])
                cell["policies"][pol] = {**p, "ttg_ci95": R.pct(ttg_r), "apcer": p["far"], "bpcer": p["frr"],
                                         "window": window(p["frr"], p["far"]), **mb}
            # E8: BPCER at APCER budgets fixed on dev (tau_far1/5) and at the target oracle
            cell["bpcer_at_apcer"] = {}
            for bud, pol in ((0.01, "far1"), (0.05, "far5")):
                tt = tau_at_far(s, y, bud)
                cell["bpcer_at_apcer"][str(bud)] = {"dev_budget": cell["policies"][pol]["frr"],
                                                    "target_oracle": frr_at(tt, s, y)}
            out[f"{m}|{surf}"] = cell
            pe = cell["policies"]["eer"]
            say(f"{m:13s} {surf:13s} EER {100 * cell['eer']:5.1f}  tau_eer: FRR {100 * pe['frr']:5.1f} FAR {100 * pe['far']:5.1f} "
                f"TTG {100 * pe['ttg']:5.2f} [{100 * pe['ttg_ci95'][0]:.2f}, {100 * pe['ttg_ci95'][1]:.2f}] share "
                f"{pe['transfer_share']:.2f}  tie FRR {100 * cell['tie']['frr'][1]:.1f}-{100 * cell['tie']['frr'][0]:.1f}  "
                f"frr1->FRR {100 * cell['policies']['frr1']['frr']:5.1f}  far5->FAR {100 * cell['policies']['far5']['far']:5.1f}")
    return out


# --------------------------------------------------------------------------- E2 / E3
def runs_table(model, seeds, surfaces=("la19_eval", "itw")):
    rows = {}
    for k in seeds:
        if not have(model, "la19_dev", k):
            continue
        _, yd, sd = R.load(model, "la19_dev", k)
        t = R.taus(model, k)
        r = {"dev_eer": eer(sd, yd)[0]}
        for surf in surfaces:
            if have(model, surf, k):
                _, y, s = R.load(model, surf, k)
                for pol in ("eer", "frr1", "far5"):
                    p = R.point(s, y, t[pol])
                    r[f"{surf}|{pol}"] = {kk: p[kk] for kk in ("frr", "far", "hter", "hter_oracle", "ttg",
                                                              "transfer_share", "eer")}
        rows[k] = r
    return rows


def e2_e3():
    out = {}
    say("\n== E2 seed spread (LA19 eval / ITW at tau_eer) and E3 last-epoch comparison")
    ref = {}
    for surf in ("la19_eval", "itw"):
        _, y, s = R.load("aasist_l", surf)
        p = R.point(s, y, R.taus("aasist_l")["eer"])
        ref[surf] = p
    out["released_aasist_l"] = {k: {kk: v[kk] for kk in ("frr", "far", "hter", "ttg", "eer")} for k, v in ref.items()}
    for model in ("aasist_l_r2", "aasist_l_trim", "aasist_l_r2_last", "aasist_l_trim_last"):
        seeds = R.RUN_SEEDS + ((R.PIPELINE_SEED,) if model.startswith("aasist_l_r2") else ())
        rows = runs_table(model, seeds)
        out[model] = {str(k): v for k, v in rows.items()}
        if not rows:
            continue
        for surf in ("la19_eval", "itw"):
            vals = [v[f"{surf}|eer"]["ttg"] for k, v in rows.items() if k in R.RUN_SEEDS and f"{surf}|eer" in v]
            frr = [v[f"{surf}|eer"]["frr"] for k, v in rows.items() if k in R.RUN_SEEDS and f"{surf}|eer" in v]
            if vals:
                say(f"{model:19s} {surf:9s} TTG over seeds 1-5: mean {100 * np.mean(vals):.2f} pp, range "
                    f"{100 * min(vals):.2f}-{100 * max(vals):.2f}; FRR range {100 * min(frr):.1f}-{100 * max(frr):.1f}%"
                    f"  (released AASIST-L: TTG {100 * ref[surf]['ttg']:.2f}, FRR {100 * ref[surf]['frr']:.1f}%)")
        dev = [v["dev_eer"] for k, v in rows.items() if k in R.RUN_SEEDS]
        say(f"{model:19s} dev EER over seeds 1-5: {', '.join(f'{100 * d:.2f}' for d in dev)} %")
    return out


# --------------------------------------------------------------------------- E4
def e4(B):
    out = {}
    say("\n== E4 channel effects at tau_eer (dFRR, dFAR vs clean; speaker-cluster 95% intervals, B = %d)" % B)
    conds = list(R.STRESS_NEW) + ["encodec6k"]
    entries = [(m, 0) for m in R.ROUND1 + R.DETECTORS_R2] + \
              [("aasist_l_r2", k) for k in R.RUN_SEEDS] + [("aasist_l_trim", k) for k in R.RUN_SEEDS]
    for m, k in entries:
        if not have(m, "stress_clean", k):
            continue
        ids, y, sc = R.load(m, "stress_clean", k)
        tau = R.taus(m, k)["eer"]
        g, names = R.clusters("stress_clean", ids)
        C = R.cluster_counts(len(names), B, 0)
        row = {}
        for c in conds:
            if m in R.ROUND1 and c == "encodec6k":
                continue                       # round 1's EnCodec result is H5 (reported in round 1)
            if not have(m, f"stress_{c}", k):
                continue
            idx, yx, sx = R.load(m, f"stress_{c}", k)
            pos = {u: i for i, u in enumerate(idx)}
            se = sx[np.array([pos[u] for u in ids])]
            pf, pa, dfrr, dfar = R.paired_cluster_test(y, sc, se, g, len(names), tau, C)
            row[c] = {"dfrr": pf, "dfar": pa, "dfrr_ci95": R.pct(dfrr), "dfar_ci95": R.pct(dfar)}
        out[f"{m}|s{k}"] = row
        say(f"{m:13s} s{k:<4d} " + "  ".join(f"{c}: {100 * v['dfrr']:+5.1f}/{100 * v['dfar']:+5.1f}" for c, v in row.items()))
    return out


# --------------------------------------------------------------------------- E5 / E6
def by_group(model, surface, groups: dict):
    """FRR/FAR at tau_eer, EER and TTG per group of utt_ids (language, codec condition)."""
    ids, y, s = R.load(model, surface)
    tau = R.taus(model)["eer"]
    out = {}
    for name, keep in groups.items():
        k = np.array([u in keep for u in ids])
        if k.sum() == 0 or len(set(y[k])) < 2:
            continue
        p = R.point(s[k], y[k], tau)
        out[name] = {"n": int(k.sum()), "frr": p["frr"], "far": p["far"], "eer": p["eer"], "ttg": p["ttg"]}
    return out


def e5():
    import pandas as pd
    out = {}
    say("\n== E5 MLAAD-5 by language (tau_eer); sample-rate confound: every spoof 22.05 kHz, 98.1% of bona fide 16 kHz")
    df = pd.read_csv(R.meta_file("mlaad5"), dtype=str, usecols=["utt_id", "lang"])
    groups = {lang: set(g["utt_id"]) for lang, g in df.groupby("lang")}
    for m in R.DETECTORS_R2:
        if not have(m, "mlaad5"):
            continue
        out[m] = by_group(m, "mlaad5", groups)
        say(f"{m:12s} " + "  ".join(f"{k}: FRR {100 * v['frr']:4.1f} FAR {100 * v['far']:4.1f}" for k, v in out[m].items()))
    return out


def e6():
    out = {}
    say("\n== E6 ASVspoof 5 eval by condition (as5_aasist, tau from AS5 dev; C00 = no codec, the channel-matched subset)")
    if not have("as5_aasist", "as5_eval"):
        say("   (as5_aasist / as5_eval not scored yet)")
        return out
    from deploybench.data import asvspoof5_condition, asvspoof5_rows
    groups = {}
    for r in asvspoof5_rows("eval"):
        groups.setdefault(asvspoof5_condition(r), set()).add(r["utt"])
    out = by_group("as5_aasist", "as5_eval", dict(sorted(groups.items())))
    for k, v in out.items():
        say(f"   {k}: n {v['n']:6d}  EER {100 * v['eer']:5.1f}  FRR {100 * v['frr']:5.1f}  FAR {100 * v['far']:5.1f}  "
            f"TTG {100 * v['ttg']:5.2f}")
    sil = R.R2 / "as5_silence.csv"
    if sil.exists() or sil.with_name(sil.name + ".gz").exists():
        import pandas as pd
        df = pd.read_csv(sil if sil.exists() else sil.with_name(sil.name + ".gz"))
        out["silence"] = {}
        for (ds, lab), g in df.groupby(["dataset", "label"]):
            lead, trail = g["lead35"] / 16000, g["trail35"] / 16000
            out["silence"][f"{ds}|{'bona' if lab == 1 else 'spoof'}"] = {
                "n": int(len(g)), "lead_median_s": float(lead.median()), "trail_median_s": float(trail.median()),
                "lead_q25_q75_s": [float(lead.quantile(0.25)), float(lead.quantile(0.75))]}
        for k, v in out["silence"].items():
            say(f"   silence {k}: leading median {v['lead_median_s']:.2f} s (IQR {v['lead_q25_q75_s'][0]:.2f}-"
                f"{v['lead_q25_q75_s'][1]:.2f}), trailing median {v['trail_median_s']:.2f} s")
    return out


# --------------------------------------------------------------------------- E7
def e7(models):
    out = {}
    say("\n== E7 three-zone policies (review band from source dev; primary budget 10% centred on tau_eer)")
    for m in models:
        if not have(m, source_dev(m)):
            continue
        _, yd, sd = R.load(m, source_dev(m))
        t = R.taus(m)
        for surf in SURFACES:
            if not have(m, surf):
                continue
            _, y, s = R.load(m, surf)
            cell = {}
            for bud in BUDGETS:
                lo, hi = three_zone_edges(sd, t["eer"], bud)
                mb = miss_band(s, y, t["eer"])
                lo_o, hi_o = three_zone_edges(s, mb["tau_oracle"], bud)
                lo_f, hi_f = three_zone_edges(s, t["eer"], bud)          # label-free: target pooled scores
                cell[str(bud)] = {"dev": three_zone_apply(s, y, lo, hi), "edges_dev": [lo, hi],
                                  "oracle": three_zone_apply(s, y, lo_o, hi_o),
                                  "label_free": three_zone_apply(s, y, lo_f, hi_f)}
            collapse = not t["frr1"] < t["far1"]
            lo_e, hi_e = (t["eer"], t["eer"]) if collapse else (t["frr1"], t["far1"])
            cell["error_budget"] = {"collapse": collapse, **three_zone_apply(s, y, lo_e, hi_e)}
            out[f"{m}|{surf}"] = cell
            d = cell["0.1"]["dev"]
            say(f"{m:13s} {surf:13s} review {100 * d['review_pooled']:5.1f}% (bona {100 * d['review_bona']:5.1f}) "
                f"auto-rej bona {100 * d['bona_auto_rejected']:5.1f}%  auto-acc spoof {100 * d['spoof_auto_accepted']:5.1f}% | "
                f"label-free review {100 * cell['0.1']['label_free']['review_pooled']:4.1f}% auto-rej bona "
                f"{100 * cell['0.1']['label_free']['bona_auto_rejected']:5.1f}% | error-budget rule "
                f"{'COLLAPSES' if collapse else 'band ok'}")
    return out


# --------------------------------------------------------------------------- E9
def e9(models):
    out = {}
    say("\n== E9 actual vs minimum normalised cost over priors (Platt on source dev, prior-corrected; logit(pi) -4..4)")
    for m in models:
        if not have(m, source_dev(m)):
            continue
        _, yd, sd = R.load(m, source_dev(m))
        a, b, th = bayes_thresholds(sd, yd)
        for surf in SURFACES:
            if not have(m, surf):
                continue
            _, y, s = R.load(m, surf)
            tg, fr, fa = det_curve(s, y)
            rows = []
            for L in LOGIT_GRID:
                p = 1 / (1 + np.exp(-L))
                cmin = float(min(np.min((p * fr + (1 - p) * fa) / min(p, 1 - p)), 1.0))
                if th is None:
                    rows.append((float(L), None, cmin))
                    continue
                tau = th[float(L)]
                rows.append((float(L), float(norm_cost(p, frr_at(tau, s, y), far_at(tau, s, y))), cmin))
            out[f"{m}|{surf}"] = {"platt": [a, b], "rows": rows}
            mid = [r for r in rows if r[0] == 0.0][0]
            say(f"{m:13s} {surf:13s} platt a={a:.3f} b={b:.3f}; at pi=0.5 act {mid[1] if mid[1] is None else round(mid[1], 3)} "
                f"min {mid[2]:.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="E1,E2,E4,E5,E6,E7,E9")
    ap.add_argument("--b", type=int, default=1000)
    a = ap.parse_args()
    want = set(a.only.split(","))
    models = list(R.DETECTORS_R2)
    res = {}
    if "E1" in want:
        res["E1_E8_E10"] = e1(models, a.b)
    if "E2" in want:
        res["E2_E3"] = e2_e3()
    if "E4" in want:
        res["E4"] = e4(a.b)
    if "E5" in want:
        res["E5"] = e5()
    if "E6" in want:
        res["E6"] = e6()
    if "E7" in want:
        res["E7"] = e7(models + list(R.ROUND1))
    if "E9" in want:
        res["E9"] = e9(models + list(R.ROUND1))
    R.write("r2_explore", res, OUT)


if __name__ == "__main__":
    main()
