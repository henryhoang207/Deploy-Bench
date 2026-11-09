"""The last-epoch contrast of R2-H3 with its exact permutation test (DEV-11 A; verification report, S1(b)).

    python analysis/review_r1/run_lastepoch.py   # -> results/review_r1/lastepoch.json

Exploratory and post-hoc. The result had been computed before DEV-11 was logged (2026-11-08, when the report's figure was
checked); this script is the committed form of that computation, and DEV-11 says so.

R2-H3's exact one-sided permutation test (r2_registered.perm_greater: 252 splits of ten runs, no RNG), applied to the
last-epoch weights of seeds 1-5 in each arm, aasist_l_trim_last against aasist_l_r2_last. The pipeline-check run of the
untrimmed arm (seed 1234) enters nothing. Tests: lower ITW FRR at tau_EER and lower ITW EER in the trimmed arm, Holm at
.05 over the two, as in R2-H3's family; the manipulation check is not part of it. Differences are printed trimmed minus
untrimmed, so the FRR line is negative (r2_registered.json's diff_pp is untrimmed minus trimmed).

Two checks must pass, or the script refuses and writes nothing:
  1. The same code on the pre-specified weights (aasist_l_trim against aasist_l_r2) reproduces R2-H3 in
     results/round2/analysis/r2_registered.json: differences within 1e-9, p, the number of splits and the Holm decisions
     exactly.
  2. Every last-epoch run's ITW FRR and EER equals E2_E3 in results/round2/analysis/r2_explore.json within 1e-9.

Importing this module computes nothing: the work is in main(), so that tests can replace R.load, R.taus, the committed
result files (REGISTERED, EXPLORE) and rvlib.OUT.
"""
from __future__ import annotations

import json

import numpy as np

from rvlib import R, write          # rvlib also puts analysis/round2 on the path, for r2_registered
import r2_registered as RR          # noqa: E402  (perm_greater and _run_quantities are used as they are, not re-implemented)

SEEDS = R.RUN_SEEDS                 # 1-5; the pipeline-check run (R.PIPELINE_SEED = 1234) is never read
ARMS = {"trimmed": "aasist_l_trim_last", "untrimmed": "aasist_l_r2_last"}     # DEV-11 A
PRESPEC = {"trimmed": "aasist_l_trim", "untrimmed": "aasist_l_r2"}            # the weights of R2-H3, for the self-check
FAMILY = ("b_itw_frr_lower_when_trimmed", "c_itw_eer_lower_when_trimmed")      # R2-H3 (b) and (c), the Holm family
QUANTITY = {FAMILY[0]: "itw_frr", FAMILY[1]: "itw_eer"}
HYPOTHESIS = {FAMILY[0]: "ITW FRR at tau_EER lower when trimmed", FAMILY[1]: "ITW EER lower when trimmed"}
ALPHA = RR.ALPHA                    # 0.05
N_SPLITS = 252                      # C(10, 5)
TOL = 1e-9
REGISTERED = R.R2 / "analysis" / "r2_registered.json"
EXPLORE = R.R2 / "analysis" / "r2_explore.json"


def read_committed(path):
    if not path.is_file():
        R.die(f"{path} is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def _need(obj, *path):
    """obj[path[0]][path[1]]..., refusing (and naming the path) if a key is missing."""
    for i, key in enumerate(path):
        if not isinstance(obj, dict) or key not in obj:
            R.die(f"the committed result file has no {' / '.join(str(p) for p in path[: i + 1])}")
        obj = obj[key]
    return obj


def collect(model):
    """{seed: {"itw_frr", "itw_eer"}} for seeds 1-5 of one model: the run quantities of r2_registered, as h3() takes them
    (ITW FRR at the run's frozen tau_EER, ITW EER). The run's LA19-dev EER, which only the manipulation check uses, is
    not kept."""
    out = {}
    for k in SEEDS:
        q = RR._run_quantities(model, k)
        out[k] = {"itw_frr": float(q["itw_frr"]), "itw_eer": float(q["itw_eer"])}
    return out


def _values(arm, key):
    return [arm[k][key] for k in sorted(arm)]


def contrast(trimmed, untrimmed):
    """The two family tests, written as h3() writes them: perm_greater(untrimmed, trimmed), i.e. H1: the untrimmed mean
    is greater. Per test: `diff_pp_registered_sign` = 100 x (mean untrimmed - mean trimmed), the sign of
    r2_registered.json; `diff_pp` = trimmed minus untrimmed, the sign the paper prints; the one-sided p; its exact count
    out of n_splits."""
    out = {}
    for name in FAMILY:
        key = QUANTITY[name]
        obs, p, n = RR.perm_greater(_values(untrimmed, key), _values(trimmed, key))
        count = int(round(p * n))
        if n != N_SPLITS or count / n != p:
            R.die(f"{name}: expected {N_SPLITS} splits and p = count / {N_SPLITS}, got p = {p!r} over {n} splits")
        out[name] = {"diff_pp_registered_sign": 100 * obs, "diff_pp": -100 * obs, "p": p, "count": count,
                     "n_splits": n}
    return out


def holm(tests):
    """Holm at alpha .05 over exactly the two family tests, with the names h3() gives them."""
    rej = R.holm({"b": tests[FAMILY[0]]["p"], "c": tests[FAMILY[1]]["p"]}, ALPHA)
    return {FAMILY[0]: bool(rej["b"]), FAMILY[1]: bool(rej["c"])}


def check_registered(tests, decisions, registered):
    """Self-check: the pre-specified weights, through the same code, reproduce R2-H3 of r2_registered.json. Refuses on
    any mismatch (difference of the means beyond 1e-9; p, n_splits or the Holm decision not equal)."""
    ref = _need(registered, "R2-H3", "tests")
    rec = {}
    for name in FAMILY:
        want = {k: _need(ref, name, k) for k in ("diff_pp", "p", "n_splits", "holm_reject")}
        got = tests[name]
        gap = abs(got["diff_pp_registered_sign"] - want["diff_pp"])
        same = (gap <= TOL and got["p"] == want["p"] and got["n_splits"] == want["n_splits"]
                and decisions[name] == want["holm_reject"])
        if not same:
            R.die(f"self-check: the pre-specified arms do not reproduce R2-H3 / {name} of r2_registered.json "
                  f"(diff_pp {got['diff_pp_registered_sign']!r} against {want['diff_pp']!r}, p {got['p']!r} against "
                  f"{want['p']!r}, splits {got['n_splits']} against {want['n_splits']}, Holm {decisions[name]} against "
                  f"{want['holm_reject']})")
        rec[name] = {"abs_diff_pp": gap, "p_equal": True, "n_splits_equal": True, "holm_equal": True}
    return {"source": "results/round2/analysis/r2_registered.json", "key": "R2-H3 / tests",
            "arms": dict(PRESPEC), "tolerance": TOL, "tests": rec, "passed": True}


def check_explore(last, explore):
    """Second cross-check: each last-epoch run's ITW FRR and EER (itw|eer: frr, eer) equals E2_E3 of r2_explore.json
    within 1e-9. Refuses otherwise. Seed 1234, which E2_E3 also holds, is not read."""
    worst, n = 0.0, 0
    for arm, model in ARMS.items():
        for k, q in last[arm].items():
            cell = _need(explore, "E2_E3", model, str(k), "itw|eer")
            for key, field in (("itw_frr", "frr"), ("itw_eer", "eer")):
                gap = abs(q[key] - _need(cell, field))
                if not gap <= TOL:                                    # also refuses on NaN
                    R.die(f"cross-check: {model} seed {k} {key} = {q[key]!r} differs from E2_E3 / itw|eer / {field} = "
                          f"{cell[field]!r} in r2_explore.json")
                worst, n = max(worst, gap), n + 1
    return {"source": "results/round2/analysis/r2_explore.json", "key": "E2_E3 / <model> / <seed> / itw|eer / frr, eer",
            "values_checked": n, "tolerance": TOL, "max_abs_diff": worst, "passed": True}


def build(last, tests, decisions, self_check, cross_check):
    meta = {"script": "analysis/review_r1/run_lastepoch.py", "deviation": "DEV-11 A",
            "status": "exploratory and post-hoc; the result had been computed before DEV-11 was logged",
            "arms": dict(ARMS), "seeds": list(SEEDS),
            "excluded_runs": {"aasist_l_r2_last": [R.PIPELINE_SEED]},
            "excluded_reason": "the pipeline-check run (seed 1234) enters no test or mean",
            "quantities": "r2_registered._run_quantities: ITW FRR at the run's frozen tau_EER, and ITW EER",
            "family": list(FAMILY), "holm_alpha": ALPHA,
            "test": "r2_registered.perm_greater(untrimmed, trimmed): exact one-sided permutation test of the difference "
                    f"in run means over all {N_SPLITS} splits of ten runs; p counts the observed split; no RNG",
            "sign": "diff_pp_trimmed_minus_untrimmed = 100 x (mean trimmed - mean untrimmed); the diff_pp of "
                    "r2_registered.json is untrimmed minus trimmed",
            "self_check": self_check, "cross_check": cross_check}
    cells = {"runs": {model: {str(k): q for k, q in last[arm].items()} for arm, model in ARMS.items()},
             "means": {model: {key: float(np.mean(_values(last[arm], key))) for key in ("itw_frr", "itw_eer")}
                       for arm, model in ARMS.items()},
             "tests": {name: {"hypothesis": HYPOTHESIS[name], "diff_pp_trimmed_minus_untrimmed": tests[name]["diff_pp"],
                              "p": tests[name]["p"], "count": tests[name]["count"], "n_splits": tests[name]["n_splits"],
                              "holm_reject": decisions[name]} for name in FAMILY}}
    return meta, cells


def main():
    registered, explore = read_committed(REGISTERED), read_committed(EXPLORE)
    spec = {arm: collect(model) for arm, model in PRESPEC.items()}
    spec_tests = contrast(spec["trimmed"], spec["untrimmed"])
    self_check = check_registered(spec_tests, holm(spec_tests), registered)
    last = {arm: collect(model) for arm, model in ARMS.items()}
    cross_check = check_explore(last, explore)
    tests = contrast(last["trimmed"], last["untrimmed"])
    decisions = holm(tests)
    meta, cells = build(last, tests, decisions, self_check, cross_check)
    write("lastepoch", meta, cells)
    for name in FAMILY:
        t = tests[name]
        print(f"{HYPOTHESIS[name]:38s} trimmed - untrimmed {t['diff_pp']:+6.2f} pp  p = {t['count']}/{t['n_splits']} = "
              f"{t['p']:.3f}  Holm {'rejects' if decisions[name] else 'does not reject'}")
    print("self-check (R2-H3 reproduced) and E2_E3 cross-check passed; wrote lastepoch.json")


if __name__ == "__main__":
    main()
