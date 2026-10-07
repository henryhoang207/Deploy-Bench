"""Round-2 registered tests (PREREGISTRATION_ROUND2.md, Sec. 6). Written, tested and committed BEFORE any round-2
score existed (tag round2-freeze). Run once every registered score file, tau file and results/round2/gates.json
exist; it refuses otherwise.

    python analysis/round2/r2_registered.py            # -> results/round2/analysis/r2_registered.{json,txt}

R2-H1  The policy sets the direction: on ITW, FRR > FAR at tau_frr1 and FAR > FRR at tau_far5, for each round-2
       detector (as5_aasist, pascu_lr, xlsr_mamba). Speaker-cluster bootstrap (ITW labels), B = 10,000, seed 0,
       p = (1 + #{stat <= 0}) / (B + 1), Holm over the six tests at .05. A detector supports H1 if both of its tests
       reject; H1 is supported if at least 2 of 3 detectors do. A detector that failed its gate or has no scores
       counts as not supporting.
R2-H2  The LA19-eval transfer gap survives retraining: TTG(tau_eer) on LA19 eval > 1 pp for the aasist_l_r2 runs
       (seeds 1-5). Speaker-cluster bootstrap (LA19-eval speakers), the oracle recomputed in every replicate, B =
       10,000, seed 0, H0: TTG <= 1 pp, Holm over five; supported if at least 4 of 5 reject (a missing run does not).
R2-H3  Removing the silence cue from training, aasist_l_trim vs aasist_l_r2 (seeds 1-5 each), exact one-sided
       permutation tests of the difference in run means (252 splits of 10 runs): (a) manipulation check, not in the
       family: LA19-dev EER higher when trimmed; (b) ITW FRR at tau_eer lower when trimmed; (c) ITW EER lower when
       trimmed. Holm over (b), (c). Fewer than 3 runs in an arm: not evaluable.
R2-H4  H5 generalised to eight channel conditions: for each condition c in STRESS_NEW and each of the four primary
       round-1 models, dFRR - dFAR > 0 at tau_eer relative to the clean copy of the 1,000-clip manifest; paired
       speaker-cluster bootstrap (67 LA19-eval speakers), B = 10,000, seed 0, Holm over the four models within c;
       c supports H4 if at least 3 of 4 reject. The components dFRR > 0 and dFAR < 0 are reported with Holm too.
R2-RL  Reading rule for the ICME revision's limitation (a), descriptive and fixed in advance, per round-2 detector on
       ITW at tau_eer: ITW EER >= 20% -> "discrimination-limited"; else "transfer-limited" if the transfer share is
       >= 0.5 or the tie interval spans >= 10 pp of ITW FRR; "well-transferred" if the share is < 0.2 and the span
       < 5 pp; otherwise "intermediate".
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r2lib as R  # noqa: E402

B, SEED, ALPHA = 10_000, 0, 0.05
TTG_MARGIN = 0.01
OUT = []


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def gates():
    f = R.R2 / "gates.json"
    if not f.exists():
        R.die("results/round2/gates.json missing: run r2_gates.py first")
    return json.loads(f.read_text(encoding="utf-8"))


def _try(fn, *a):
    try:
        return fn(*a)
    except SystemExit as e:
        return e


# --------------------------------------------------------------------------- R2-H1
def h1(gt):
    pv, res = {}, {}
    for m in R.DETECTORS_R2:
        if not gt.get(m, {}).get("passed", False):
            res[m] = {"evaluable": False, "reason": f"gate: {gt.get(m, {}).get('summary', 'no gate record')}"}
            continue
        got = _try(R.load, m, "itw")
        t = _try(R.taus, m)
        if isinstance(got, SystemExit) or isinstance(t, SystemExit):
            res[m] = {"evaluable": False, "reason": str(got if isinstance(got, SystemExit) else t)}
            continue
        ids, y, s = got
        g, names = R.clusters("itw", ids)
        G = len(names)
        C = R.cluster_counts(G, B, SEED)
        fr1, fa1 = R.cluster_rates(s, y, g, G, t["frr1"], C)
        fr5, fa5 = R.cluster_rates(s, y, g, G, t["far5"], C)
        a, b = fr1 - fa1, fa5 - fr5
        pv[f"{m}|frr1"], pv[f"{m}|far5"] = R.pval_le0(a), R.pval_le0(b)
        p1, p5 = R.point(s, y, t["frr1"]), R.point(s, y, t["far5"])
        res[m] = {"evaluable": True, "clusters": G,
                  "frr1": {"frr_pct": 100 * p1["frr"], "far_pct": 100 * p1["far"],
                           "frr_minus_far_pp": 100 * (p1["frr"] - p1["far"]), "ci95_pp": [100 * x for x in R.pct(a)],
                           "p": pv[f"{m}|frr1"]},
                  "far5": {"frr_pct": 100 * p5["frr"], "far_pct": 100 * p5["far"],
                           "far_minus_frr_pp": 100 * (p5["far"] - p5["frr"]), "ci95_pp": [100 * x for x in R.pct(b)],
                           "p": pv[f"{m}|far5"]}}
    rej = R.holm(pv, ALPHA) if pv else {}
    n_sup = 0
    for m in R.DETECTORS_R2:
        if res[m]["evaluable"]:
            res[m]["frr1"]["holm_reject"] = bool(rej[f"{m}|frr1"])
            res[m]["far5"]["holm_reject"] = bool(rej[f"{m}|far5"])
            res[m]["supports"] = res[m]["frr1"]["holm_reject"] and res[m]["far5"]["holm_reject"]
        else:
            res[m]["supports"] = False
        n_sup += int(res[m]["supports"])
    out = {"B": B, "seed": SEED, "per_model": res, "n_supporting": n_sup, "supported": n_sup >= 2}
    say(f"== R2-H1  the policy sets the direction (ITW; B = {B}, seed {SEED}, Holm over {len(pv)})")
    for m in R.DETECTORS_R2:
        r = res[m]
        if not r["evaluable"]:
            say(f"{m:12s} NOT EVALUABLE ({r['reason']}) -> counts as not supporting")
            continue
        say(f"{m:12s} tau_frr1: FRR {r['frr1']['frr_pct']:5.1f} FAR {r['frr1']['far_pct']:5.1f} "
            f"[{r['frr1']['ci95_pp'][0]:+.1f}, {r['frr1']['ci95_pp'][1]:+.1f}] p={r['frr1']['p']:.4f} "
            f"Holm {r['frr1']['holm_reject']} | tau_far5: FRR {r['far5']['frr_pct']:5.1f} FAR {r['far5']['far_pct']:5.1f} "
            f"[{r['far5']['ci95_pp'][0]:+.1f}, {r['far5']['ci95_pp'][1]:+.1f}] p={r['far5']['p']:.4f} "
            f"Holm {r['far5']['holm_reject']} -> supports: {r['supports']}")
    say(f"-> {n_sup} of 3 detectors support; R2-H1 {'SUPPORTED' if out['supported'] else 'NOT SUPPORTED'} (needs 2)")
    return out


# --------------------------------------------------------------------------- R2-H2
def h2():
    pv, res = {}, {}
    for k in R.RUN_SEEDS:
        got = _try(R.load, "aasist_l_r2", "la19_eval", k)
        t = _try(R.taus, "aasist_l_r2", k)
        if isinstance(got, SystemExit) or isinstance(t, SystemExit):
            res[k] = {"evaluable": False, "reason": str(got if isinstance(got, SystemExit) else t)}
            continue
        ids, y, s = got
        tau = t["eer"]
        g, names = R.clusters("la19_eval", ids)
        G = len(names)
        C = R.cluster_counts(G, B, SEED)
        fr, fa = R.cluster_rates(s, y, g, G, tau, C)
        ttg = 0.5 * (fr + fa) - R.Oracle(s, y, g).run(C)
        pv[k] = R.pval_le0(ttg - TTG_MARGIN)
        p = R.point(s, y, tau)
        res[k] = {"evaluable": True, "clusters": G, "ttg_pp": 100 * p["ttg"], "ci95_pp": [100 * x for x in R.pct(ttg)],
                  "p": pv[k], "frr_pct": 100 * p["frr"], "far_pct": 100 * p["far"], "hter_pct": 100 * p["hter"],
                  "eer_pct": 100 * p["eer"], "transfer_share": p["transfer_share"]}
    rej = R.holm({str(k): v for k, v in pv.items()}, ALPHA) if pv else {}
    n_rej = 0
    for k in R.RUN_SEEDS:
        res[k]["holm_reject"] = bool(rej.get(str(k), False)) if res[k]["evaluable"] else False
        n_rej += int(res[k]["holm_reject"])
    out = {"B": B, "seed": SEED, "margin_pp": 100 * TTG_MARGIN, "per_run": {str(k): v for k, v in res.items()},
           "n_rejected": n_rej, "supported": n_rej >= 4}
    say(f"\n== R2-H2  LA19-eval TTG > 1 pp across retraining (aasist_l_r2 seeds {R.RUN_SEEDS}; B = {B})")
    for k in R.RUN_SEEDS:
        r = res[k]
        if not r["evaluable"]:
            say(f"seed {k}: NOT EVALUABLE ({r['reason']})")
            continue
        say(f"seed {k}: TTG {r['ttg_pp']:5.2f} pp [{r['ci95_pp'][0]:.2f}, {r['ci95_pp'][1]:.2f}] p={r['p']:.4f} "
            f"Holm {r['holm_reject']} | EER {r['eer_pct']:.2f}% HTER {r['hter_pct']:.2f}% share {r['transfer_share']:.2f}")
    say(f"-> {n_rej} of 5 runs reject H0 (TTG <= 1 pp); R2-H2 {'SUPPORTED' if out['supported'] else 'NOT SUPPORTED'} (needs 4)")
    return out


# --------------------------------------------------------------------------- R2-H3
def perm_greater(a, b):
    """Exact one-sided permutation test of mean(a) > mean(b) over all relabellings; p counts the observed split."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    v = np.concatenate([a, b])
    n, N = len(a), len(v)
    obs = a.mean() - b.mean()
    cnt = tot = 0
    for idx in combinations(range(N), n):
        m = np.zeros(N, bool)
        m[list(idx)] = True
        st = v[m].mean() - v[~m].mean()
        cnt += st >= obs - 1e-12
        tot += 1
    return float(obs), cnt / tot, tot


def _run_quantities(model, k):
    from deploybench.metrics import eer, frr_at
    _, yd, sd = R.load(model, "la19_dev", k)
    _, yi, si = R.load(model, "itw", k)
    t = R.taus(model, k)["eer"]
    return {"dev_eer": eer(sd, yd)[0], "itw_frr": frr_at(t, si, yi), "itw_eer": eer(si, yi)[0]}


def h3():
    q = {"aasist_l_r2": {}, "aasist_l_trim": {}}
    missing = []
    for model in q:
        for k in R.RUN_SEEDS:
            got = _try(_run_quantities, model, k)
            if isinstance(got, SystemExit):
                missing.append(f"{model}/s{k}: {got}")
            else:
                q[model][k] = got
    na, nb = len(q["aasist_l_trim"]), len(q["aasist_l_r2"])
    say(f"\n== R2-H3  silence-trimmed training (aasist_l_trim, n = {na}) vs aasist_l_r2 (n = {nb}); exact permutation")
    if na < 3 or nb < 3:
        say("-> NOT EVALUABLE (fewer than 3 runs in an arm): " + "; ".join(missing))
        return {"evaluable": False, "missing": missing}
    T = lambda key, model: [q[model][k][key] for k in sorted(q[model])]  # noqa: E731
    res = {}
    obs, p, n = perm_greater(T("dev_eer", "aasist_l_trim"), T("dev_eer", "aasist_l_r2"))
    res["a_dev_eer_higher_when_trimmed"] = {"diff_pp": 100 * obs, "p": p, "n_splits": n}
    obs, p, n = perm_greater(T("itw_frr", "aasist_l_r2"), T("itw_frr", "aasist_l_trim"))
    res["b_itw_frr_lower_when_trimmed"] = {"diff_pp": 100 * obs, "p": p, "n_splits": n}
    obs, p, n = perm_greater(T("itw_eer", "aasist_l_r2"), T("itw_eer", "aasist_l_trim"))
    res["c_itw_eer_lower_when_trimmed"] = {"diff_pp": 100 * obs, "p": p, "n_splits": n}
    rej = R.holm({"b": res["b_itw_frr_lower_when_trimmed"]["p"], "c": res["c_itw_eer_lower_when_trimmed"]["p"]}, ALPHA)
    res["b_itw_frr_lower_when_trimmed"]["holm_reject"] = bool(rej["b"])
    res["c_itw_eer_lower_when_trimmed"]["holm_reject"] = bool(rej["c"])
    out = {"evaluable": True, "runs": {m: {str(k): v for k, v in q[m].items()} for m in q}, "tests": res,
           "missing": missing}
    for name, r in res.items():
        say(f"{name:34s} diff {r['diff_pp']:+6.2f} pp  p = {r['p']:.4f} ({r['n_splits']} splits)"
            + (f"  Holm {r['holm_reject']}" if "holm_reject" in r else "  (manipulation check)"))
    return out


# --------------------------------------------------------------------------- R2-H4
def h4():
    out = {"B": B, "seed": SEED, "conditions": {}}
    say(f"\n== R2-H4  H5 generalised: dFRR - dFAR > 0 at tau_eer, four primary models, per condition (B = {B})")
    n_sup = 0
    for c in R.STRESS_NEW:
        pv = {"stat": {}, "dfrr": {}, "dfar": {}}
        res = {}
        for m in R.PRIMARY4:
            gc = _try(R.load, m, "stress_clean")
            gx = _try(R.load, m, f"stress_{c}")
            if isinstance(gc, SystemExit) or isinstance(gx, SystemExit):
                res[m] = {"evaluable": False, "reason": str(gc if isinstance(gc, SystemExit) else gx)}
                continue
            ids, y, sc = gc
            idx, yx, sx = gx
            pos = {u: i for i, u in enumerate(idx)}
            if set(pos) != set(ids):
                R.die(f"{m}/{c}: utt_ids differ from the clean copy")
            o = np.array([pos[u] for u in ids])
            se, ye = sx[o], yx[o]
            if not np.array_equal(ye, y):
                R.die(f"{m}/{c}: labels differ from the clean copy")
            tau = R.taus(m)["eer"]
            g, names = R.clusters(f"stress_{c}", ids)
            G = len(names)
            C = R.cluster_counts(G, B, SEED)
            pfrr, pfar, dfrr, dfar = R.paired_cluster_test(y, sc, se, g, G, tau, C)
            st = dfrr - dfar
            pv["stat"][m], pv["dfrr"][m], pv["dfar"][m] = R.pval_le0(st), R.pval_le0(dfrr), R.pval_le0(-dfar)
            res[m] = {"evaluable": True, "clusters": G, "dfrr_pp": 100 * pfrr, "dfar_pp": 100 * pfar,
                      "stat_ci95_pp": [100 * x for x in R.pct(st)], "dfrr_ci95_pp": [100 * x for x in R.pct(dfrr)],
                      "dfar_ci95_pp": [100 * x for x in R.pct(dfar)],
                      "p_stat": pv["stat"][m], "p_dfrr": pv["dfrr"][m], "p_dfar": pv["dfar"][m]}
        holm = {k: (R.holm(v, ALPHA) if v else {}) for k, v in pv.items()}
        n_rej = 0
        for m in R.PRIMARY4:
            if res[m]["evaluable"]:
                for k in pv:
                    res[m][f"holm_{k}"] = bool(holm[k][m])
                n_rej += int(res[m]["holm_stat"])
        sup = n_rej >= 3
        n_sup += int(sup)
        out["conditions"][c] = {"per_model": res, "n_rejected": n_rej, "supported": sup}
        say(f"{c:8s} " + "  ".join(
            (f"{m}: {res[m]['dfrr_pp']:+5.1f}/{res[m]['dfar_pp']:+5.1f} p={res[m]['p_stat']:.4f}"
             f"{'*' if res[m]['holm_stat'] else ''}") if res[m]["evaluable"] else f"{m}: n/a"
            for m in R.PRIMARY4) + f"  -> {n_rej}/4, {'supported' if sup else 'not supported'}")
    out["n_conditions_supported"] = n_sup
    say(f"-> R2-H4 holds for {n_sup} of {len(R.STRESS_NEW)} conditions (per condition: at least 3 of 4 models)")
    return out


# --------------------------------------------------------------------------- R2-RL
def reading_rule(gt):
    out = {}
    say("\n== R2-RL  reading rule for limitation (a) (ITW, tau_eer; descriptive, fixed in advance)")
    for m in R.DETECTORS_R2:
        if not gt.get(m, {}).get("passed", False):
            out[m] = {"category": "not evaluable (gate)"}
            say(f"{m:12s} not evaluable (gate)")
            continue
        src = "as5_dev" if m == "as5_aasist" else "la19_dev"
        got_d, got_i, t = _try(R.load, m, src), _try(R.load, m, "itw"), _try(R.taus, m)
        if any(isinstance(x, SystemExit) for x in (got_d, got_i, t)):
            out[m] = {"category": "not evaluable (missing scores or taus)"}
            say(f"{m:12s} not evaluable (missing scores or taus)")
            continue
        _, yd, sd = got_d
        _, yi, si = got_i
        lo, hi = R.tie_interval(sd, yd, t["eer"])
        from deploybench.metrics import frr_at
        span = 100 * abs(frr_at(lo, si, yi) - frr_at(hi, si, yi))
        p = R.point(si, yi, t["eer"])
        eer_pct, share = 100 * p["eer"], p["transfer_share"]
        if eer_pct >= 20:
            cat = "discrimination-limited"
        elif share >= 0.5 or span >= 10:
            cat = "transfer-limited"
        elif share < 0.2 and span < 5:
            cat = "well-transferred"
        else:
            cat = "intermediate"
        out[m] = {"itw_eer_pct": eer_pct, "transfer_share": share, "tie_tau": [lo, hi], "tie_itw_frr_span_pp": span,
                  "category": cat}
        say(f"{m:12s} ITW EER {eer_pct:5.1f}%  share {share:.2f}  tie-interval FRR span {span:5.1f} pp -> {cat}")
    return out


def main():
    gt = gates()
    res = {"R2-H1": h1(gt), "R2-H2": h2(), "R2-H3": h3(), "R2-H4": h4(), "R2-RL": reading_rule(gt)}
    R.write("r2_registered", res, OUT)


if __name__ == "__main__":
    main()
