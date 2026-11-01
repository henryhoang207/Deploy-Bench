"""Print every number of the review analyses (DEV-10) in the form the paper quotes, for the claim audit:

    python analysis/review_r1/rv_checks.py   # results/review_r1/*.json -> results/review_r1/rv_checks.txt

TTG and its intervals in percentage points with two decimals; HTER, FRR, FAR, total error rates and accuracies in percent
with one decimal; class shares with three decimals; p values with four. Reads only the JSON files that the run_*.py
scripts write.
"""
from __future__ import annotations

from rvlib import NAMES, OUT, R, read

SURF = {"la19_eval": "LA19 eval", "itw": "ITW", "df21_verified": "DF21 ver.", "mlaad5": "MLAAD-5", "as5_eval": "AS5 eval"}
POLICIES = (("static", "static"), ("logistic", "logistic"), ("znorm", "z-norm"), ("ratematch", "rate-match"),
            ("emquant", "EM-quantile"), ("gmm2", "GMM-2"), ("loso", "LOSO"))
TER = (("static", "static"), ("bayes_pi0.5", "equal-prior Bayes"), ("bayes_em", "EM Bayes"),
       ("bayes_bbse", "BBSE Bayes"), ("bayes_trueprior", "true-prior Bayes"), ("oracle", "TER-optimal"))
LINES = []


def say(s=""):
    LINES.append(s)


def pp(x):
    return f"{100 * x:.2f}"


def pc(x):
    return f"{100 * x:.1f}"


def iv(a):
    return f"[{pp(a[0])}, {pp(a[1])}]"


def pp3(x):                     # derived summary values, printed with a spare decimal so that they round either way
    return f"{100 * x:.3f}"


def label(m, seed=0):
    return f"AASIST-L run, seed {seed}" if m == "aasist_l_r2" else NAMES[m]


def check1():
    cv = read("cv")["cells"]
    say("== Check 1. In-sample versus cross-validated oracle (cv.json)")
    say("TTG in pp at the frozen tau_EER. Columns: detector | target | speakers | HTER at tau_EER (%) | TTG in-sample | "
        "clip 5-fold mean | leave-one-speaker-out | speaker 5-fold mean [range of 20] | transfer share in-sample / LOSO / "
        "speaker 5-fold")
    for c in cv.values():
        h = c["hter_star"]
        say(f"{label(c['model'], c['seed'])} | {SURF[c['target']]} | {c['G']} | {pc(h)} | {pp(c['ttg_in'])} | "
            f"{pp(c['ttg_clip5_mean'])} | {pp(c['ttg_loso'])} | {pp(c['ttg_spk5_mean'])} {iv(c['ttg_spk5_range'])} | "
            f"{c['ttg_in'] / h:.2f} / {c['ttg_loso'] / h:.2f} / {c['ttg_spk5_mean'] / h:.2f}")

    def drop(c):
        return c["ttg_in"] - min(c["ttg_clip5_mean"], c["ttg_loso"], c["ttg_spk5_mean"])

    def share_move(c):
        return max(abs(c["ttg_in"] - c[k]) for k in ("ttg_loso", "ttg_spk5_mean")) / c["hter_star"]

    core = [c for c in cv.values() if c["model"] in R.DETECTORS_R2 + ("xlsr_aasist",) or c["target"] == "la19_eval"]
    worst = max(core, key=drop)
    moved = max(core, key=share_move)
    say("summary:")
    say(f"  round-2 detectors and the reference on every target, and every LA19-eval cell ({len(core)} cells): largest "
        f"drop from the in-sample to a cross-validated TTG {pp3(drop(worst))} pp ({label(worst['model'], worst['seed'])}, "
        f"{SURF[worst['target']]}); largest move of the transfer share {share_move(moved):.3f} "
        f"({label(moved['model'], moved['seed'])}, {SURF[moved['target']]})")
    runs = [c for c in cv.values() if c["model"] == "aasist_l_r2" and c["seed"] in (2, 3, 4)]
    say(f"  AASIST-L runs, seeds 2 to 4: largest drop {pp3(max(drop(c) for c in runs))} pp")
    prim = [c for c in cv.values() if c["model"] in R.PRIMARY4 and c["target"] == "itw"]
    loso = [c["ttg_in"] - c["ttg_loso"] for c in prim]
    say(f"  round-1 primary models on ITW: drop to the leave-one-speaker-out TTG {pp3(min(loso))} to {pp3(max(loso))} pp; "
        f"negative leave-one-speaker-out TTG: {', '.join(label(c['model']) for c in prim if c['ttg_loso'] < 0) or 'none'}")
    say()


def check1b():
    h2 = read("h2cv")
    say("== Check 1b. R2-H2 with a cross-validated oracle in every bootstrap replicate (h2cv.json)")
    b = h2["meta"]["bootstrap"]
    say(f"Speaker bootstrap B = {b['B']}, seed {b['seed']}; margin {pp(h2['meta']['margin'])} pp. Columns: run | TTG "
        "in-sample [95%] | p | TTG 5-fold CV [95%] | p | fold seed | window columns")
    for k, c in h2["cells"].items():
        say(f"seed {k} | {pp(c['ttg_in'])} {iv(c['ci_in'])} | {c['p_in']:.4f} | {pp(c['ttg_cv'])} {iv(c['ci_cv'])} | "
            f"{c['p_cv']:.4f} | {c['fold_seed']} | {c['window_cols']}")
    for lab, key in (("in-sample (registered)", "in_sample"), ("cross-validated", "cv")):
        s = h2.get("summary", {}).get("holm", {}).get(key)
        if s:
            rej = ", ".join(k for k, v in sorted(s["rejected"].items()) if v)
            say(f"Holm, {lab}: {s['rejections']}/5 rejections ({'supported' if s['supported'] else 'not supported'}; "
                f"seeds {rej})")
    say()


def check1c():
    f1, a2 = read("h2cv_folds"), read("h2cv")["cells"]
    b, fo = f1["meta"]["bootstrap"], f1["meta"]["folds"]
    say("== Check 1c. R2-H2 with a cross-validated oracle over 10 fold assignments per run (DEV-10 F1; h2cv_folds.json)")
    say(f"Speaker bootstrap B = {b['B']}, seed {b['seed']}; margin {pp(f1['meta']['margin'])} pp; {fo['K']}-fold speaker "
        f"folds, assignment a of run k drawn by default_rng(k + 10 a). Columns: run | assignment | fold seed | TTG 5-fold CV "
        "[95%] | p | threshold search")
    for k, c in f1["cells"].items():
        for r in c["assignments"]:
            say(f"seed {k} | {r['assignment']} | {r['fold_seed']} | {pp(r['ttg_cv'])} {iv(r['ci_cv'])} | {r['p_cv']:.4f} | "
                f"{r['method']}")
    s = f1["summary"]
    say("Per run. Columns: run | TTG 5-fold CV over the assignments, min to max | lowest lower interval end | largest p | "
        "Holm on the largest p")
    for k, c in f1["cells"].items():
        t = [r["ttg_cv"] for r in c["assignments"]]
        say(f"seed {k} | {pp(min(t))} to {pp(max(t))} | {pp(c['min_ci_lo'])} | {c['max_p_cv']:.4f} | "
            f"{'rejects' if s['holm_on_largest_p']['rejected'][k] else 'does not reject'}")

    def same(r, c):
        return r["fold_seed"] == c["fold_seed"] and all(
            abs(x - z) <= 1e-12 for x, z in zip([r["ttg_cv"], *r["ci_cv"], r["p_cv"]], [c["ttg_cv"], *c["ci_cv"], c["p_cv"]]))

    h = s["holm_on_largest_p"]
    per_a = s["holm_per_assignment"]
    say("summary:")
    say(f"  Holm over the five runs on each run's largest p: {h['rejections']}/5 rejections, so the cross-validated support "
        f"is {'robust' if h['robust'] else 'not robust'} to the fold assignment (reading rule: at least 4)")
    say(f"  Holm across the five runs, per assignment index 0 to 9: {' / '.join(map(str, per_a['rejections']))} rejections; "
        f"indices with at least 4: {per_a['assignments_with_at_least_4']} of {len(per_a['rejections'])}")
    say(f"  assignments computed on the full grid because the window refused: {s['full_grid_assignments']}")
    say(f"  assignment 0 reproduces A2 (h2cv.json) in every run: "
        f"{'yes' if all(same(c['assignments'][0], a2[k]) for k, c in f1['cells'].items()) else 'NO'}")
    say()


def check2():
    dev = read("dev")["cells"]
    say("== Check 2. Dev-threshold estimators (dev.json)")
    say("Trials that pin tau_EER (dev bona fide below / dev spoof at or above), of the dev bona fide trials:")
    seen = set()
    for c in dev.values():
        if (c["model"], c["seed"]) not in seen:
            seen.add((c["model"], c["seed"]))
            say(f"  {label(c['model'], c['seed'])}: {c['dev_pins'][0]}/{c['dev_pins'][1]} of {c['nb_dev']}")
    say("TTG in pp. Columns: detector | target | at tau_EER | over tied thresholds, min to max | dev bootstrap median [95%] | "
        "KDE x0.5 / x1 / x2 | isotonic (PAV) | logistic (E9) | FRR/FAR (%) at tau_EER | FRR/FAR (%) at the best tied "
        "threshold   (* = threshold outside the dev-equivalent interval)")
    for c in dev.values():
        sm = c["smooth"]

        def v(k):
            return pp(sm[k]["ttg"]) + ("" if sm[k]["in_tie"] else "*")

        q = c["devboot_ttg_q"]
        f = c["frr_far_at_tie_min"]
        say(f"{label(c['model'], c['seed'])} | {SURF[c['target']]} | {pp(c['ttg_star'])} | {pp(c['ttg_tie_min'])} to "
            f"{pp(c['ttg_tie_max'])} | {pp(q[1])} [{pp(q[0])}, {pp(q[2])}] | {v('kde0.5')} / {v('kde1')} / {v('kde2')} | "
            f"{v('pav')} | {v('logistic')} | {pc(c['frr_star'])}/{pc(c['far_star'])} | {pc(f[0])}/{pc(f[1])}")
    say()


def check3():
    lad = read("ladder")["cells"]
    say("== Check 3. Threshold policies after the shift (ladder.json)")
    say("TTG in pp against the in-sample oracle. Columns: detector | target | " + " | ".join(n for _, n in POLICIES))
    for c in lad.values():
        p = c["policies"]
        say(f"{label(c['model'])} | {SURF[c['target']]} | "
            + " | ".join(pp(p[k]["ttg"]) if k in p else "-" for k, _ in POLICIES))
    say("FRR/FAR (%) per policy, and the oracle HTER (%):")
    for c in lad.values():
        p = c["policies"]
        say(f"{label(c['model'])} | {SURF[c['target']]} | "
            + " | ".join(f"{n} {pc(p[k]['frr'])}/{pc(p[k]['far'])}" for k, n in POLICIES if k in p)
            + f" | oracle HTER {pc(c['oracle_hter'])}")
    say("Bona fide share: true | EM | BBSE | GMM-2 weight of the upper component; total error rate (%) at the true "
        "class mix: " + " | ".join(n for _, n in TER))
    for c in lad.values():
        t = c["ter"]
        say(f"{label(c['model'])} | {SURF[c['target']]} | {c['pi_true']:.3f} | {c['pi_em']:.3f} | {c['pi_bbse']:.3f} | "
            f"{c['pi_gmm']:.3f} || " + " | ".join(f"{n} {pc(t[k])}" for k, n in TER))
    cv = read("cv")["cells"]
    say("Share of clips rejected at tau_EER (%) = bona fide share x FRR + spoof share x (1 - FAR); clip counts:")
    for c in lad.values():
        st = c["policies"]["static"]
        n = cv[f"{c['model']}|s0|{c['target']}"]["n"]
        rej = c["pi_true"] * st["frr"] + (1 - c["pi_true"]) * (1 - st["far"])
        say(f"{label(c['model'])} | {SURF[c['target']]} | {pc(rej)} = {c['pi_true']:.3f} x {pc(st['frr'])} + "
            f"{1 - c['pi_true']:.3f} x {pc(1 - st['far'])} | {round(c['pi_true'] * n)} bona fide of {n} clips")
    say()


def smalllabel():
    sl = read("smalllabel")
    say("== Small labelled audit (smalllabel.json)")
    say(f"Held-out TTG in pp, median [5th, 95th percentile] over {sl['meta']['splits']} speaker halvings. Columns: "
        "detector | target | static | " + " | ".join(f"k = {k}" for k in sl["meta"]["k_per_class"]))
    for key, c in sl["cells"].items():
        m, surf = key.split("|")
        say(f"{label(m)} | {SURF[surf]} | "
            + " | ".join(f"{pp(c[k][1])} [{pp(c[k][0])}, {pp(c[k][2])}]"
                         for k in ["static"] + [str(k) for k in sl["meta"]["k_per_class"]]))
    say()


def check4():
    it = read("intervals")["cells"]
    say("== Check 4. Interval methods near the zero boundary (intervals.json)")
    say("TTG in pp. Columns: detector | target | speakers | TTG | percentile (registered) | BCa | z0 | a | m-out-of-n | m")
    for key, c in it.items():
        m, surf = key.split("|")
        say(f"{label(m)} | {SURF[surf]} | {c['G']} | {pp(c['ttg'])} | {iv(c['pct'])} | {iv(c['bca'])} | "
            f"{c['z0']:+.2f} | {c['a']:+.3f} | {iv(c['m_of_n'])} | {c['m']}")
    say()


def check4b():
    mo, it = read("mofn"), read("intervals")["cells"]
    rules = list(mo["meta"]["m"])
    say("== Check 4b. Sensitivity of the m-out-of-n interval to m (DEV-10 F2; mofn.json)")
    say(f"B = {mo['meta']['B']}, default_rng(1) restarted for every cell and every m; rescaled basic interval, truncated at 0. "
        "TTG in pp. Columns: detector | target | speakers | TTG | "
        + " | ".join(f"m = round({r}): m [95%]" for r in rules) + " | over m: lower end min to max; upper end min to max")
    for key, c in mo["cells"].items():
        m, surf = key.split("|")
        lo, hi = [c[r]["m_of_n"][0] for r in rules], [c[r]["m_of_n"][1] for r in rules]
        say(f"{label(m)} | {SURF[surf]} | {c['G']} | {pp(c['ttg'])} | "
            + " | ".join(f"{c[r]['m']} {iv(c[r]['m_of_n'])}" for r in rules)
            + f" | {pp(min(lo))} to {pp(max(lo))}; {pp(min(hi))} to {pp(max(hi))}")
    same = all(c["G^(2/3)"]["m_of_n"] == it[k]["m_of_n"] and c["G^(2/3)"]["m"] == it[k]["m"]
               for k, c in mo["cells"].items())
    mixed = []                  # cells whose lower end is above 0 for some m and at 0 for another
    for key, c in mo["cells"].items():
        if len({c[r]["m_of_n"][0] > 0 for r in rules}) == 2:
            m, surf = key.split("|")
            mixed.append(f"{label(m)} {SURF[surf]}")
    short = [f"{key} at round({r}): {c[r]['finite_replicates']}" for key, c in mo["cells"].items() for r in rules
             if c[r]["finite_replicates"] < mo["meta"]["B"]]
    say("summary:")
    say(f"  m = round(G^(2/3)) reproduces E (intervals.json) in every cell: {'yes' if same else 'NO'}")
    say(f"  cells whose lower end is above 0 for some m and at 0 for another: {', '.join(mixed) or 'none'}")
    say(f"  replicates with both classes present, where fewer than B: {'; '.join(short) or 'none'}")
    say()


def check5():
    at = read("atc")
    s, strong = at["summary"], at["meta"]["strong_detectors"]
    say("== Check 5. Label-free accuracy estimation by ATC (DEV-10 F3; atc.json)")
    say("Confidence = |s - tau_EER|; ATC threshold t = the (1 - dev accuracy) quantile of the dev margins, in score units; "
        "success: |predicted - true| at most 5 pp. Accuracy in %, error in pp. Columns: detector | target | dev accuracy | "
        "t | predicted | true | error | criterion")
    for c in at["cells"].values():
        say(f"{label(c['model'])} | {SURF[c['target']]} | {pc(c['acc_dev'])} | {c['t']:.4f} | {pc(c['predicted'])} | "
            f"{pc(c['true'])} | {pc(c['error'])} | {'meets' if c['meets'] else 'fails'}")
    itw = [c["error"] for c in at["cells"].values() if c["target"] == "itw"]
    say("summary:")
    say(f"  cells meeting the criterion: {s['meeting']}/{s['cells']}; on ITW {s['itw_meeting']}/{s['itw_cells']}")
    say(f"  error on ITW, min to max: {pc(min(itw))} to {pc(max(itw))} pp; the four strong detectors on ITW: "
        + ", ".join(f"{label(m)} {pc(at['cells'][f'{m}|itw']['error'])}" for m in strong))
    say(f"  usable as a label-free monitor of this shift (every strong detector meets it on ITW): "
        f"{'yes' if s['usable'] else 'no'}")
    say()


def main():
    say("Analyses prompted by the round-1 review (DEVIATIONS.md, DEV-10). Exploratory, not registered.")
    say()
    for part in (check1, check1b, check1c, check2, check3, smalllabel, check4, check4b, check5):
        part()
    (OUT / "rv_checks.txt").write_text("\n".join(LINES), encoding="utf-8")
    print(f"wrote {(OUT / 'rv_checks.txt').relative_to(R.REPO)} ({len(LINES)} lines)")


if __name__ == "__main__":
    main()
