"""Tables for the analyses prompted by the round-1 review (DEVIATIONS.md, DEV-10, parts A-F: exploratory and post-review),
generated from the committed outputs in results/review_r1/ (never edited by hand).

    python analysis/review_r1/rv_paper_tables.py   # reads results/review_r1/{cv,dev,ladder,smalllabel,intervals,mofn,atc}.json;
                                                   # writes gen/rv_*.tex and gen/rvs_*.tex in the manuscript folder

$RV_GEN overrides the output folder; R2_DIGITS=comma groups thousands with a comma instead of the LaTeX thin space that
Springer Nature tables use, as in analysis/round2/r2_paper_tables.py.
rv_robust.tex       main text: the headline cells under the in-sample oracle, two cross-validated oracles and the dev-only
                    threshold estimators (P3.2, P4)
rv_ladder.tex       main text: the four strong detectors under every threshold policy, TTG (P6.1)
Supplement (rvs_*):
rvs_cv.tex          every cell: in-sample, clip 5-fold, leave-one-speaker-out and speaker 5-fold oracles, transfer shares (P3.3)
rvs_estimators.tex  every cell: TTG at the tied thresholds and under KDE, isotonic and logistic dev thresholds (P4.2, P4.6)
rvs_intervals.tex   every cell: percentile, BCa and m-out-of-n intervals (P5.2)
rvs_mofn.tex        every cell: the m-out-of-n interval for m = round(G^(1/2)), round(G^(2/3)) and round(G/2) (P5.3; F2)
rvs_ladder.tex      every cell: FRR, FAR, threshold and TTG per policy, in two panels of four policies (P6.2)
rvs_labelshift.tex  every cell: prior estimates against the true share, total error under five priors (P6.2)
rvs_smalllabel.tex  every cell: held-out TTG from k labelled clips per class (P6.2); ITW and LA19 eval from DEV-10 D, DF21 ver. added by DEV-11 B
rvs_atc.tex         every cell: accuracy at tau_eer predicted without target labels by ATC, against the true accuracy (P7.3; F3)
Each file is a tabular body; captions and notes stay in the manuscript. A missing input stops the run (rvlib.read).
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rvlib as V  # noqa: E402

REPO = V.REPO
GEN = Path(os.environ.get("RV_GEN", REPO / "paper" / "Thresholded_Once__Deployed_Elsewhere" / "gen"))
DIGITS = {"comma": ",", "thin": r"\,"}         # thousands separator in table cells; Springer Nature tables use "thin"
if os.environ.get("R2_DIGITS", "thin") not in DIGITS:
    raise SystemExit(f"REFUSED: R2_DIGITS must be one of {sorted(DIGITS)}")
SEP = DIGITS[os.environ.get("R2_DIGITS", "thin")]
NICE = {"as5_aasist": "AS5 AASIST", "pascu_lr": "Pascu LR", "xlsr_mamba": "XLSR-Mamba",
        "aasist": "AASIST", "aasist_l": "AASIST-L", "rawnet2": "RawNet2", "tssdnet": "Res-TSSDNet",
        "xlsr_aasist": "XLS-R+AASIST"}
SURF = {"la19_eval": "LA19 eval", "itw": "ITW", "df21_verified": "DF21 ver.", "mlaad5": "MLAAD-5", "as5_eval": "AS5 eval"}
MODELS = tuple(V.NAMES) + ("aasist_l_r2",)     # the five AASIST-L runs of round 2 follow the eight detectors
TARGETS = ("la19_eval", "itw", "df21_verified", "as5_eval")
STRONG = ("xlsr_aasist", "xlsr_mamba", "as5_aasist", "pascu_lr")   # the four detectors that separate ITW well
STAR = r"\rlap{$^{\ast}$}"                     # marks an estimate outside the dev-equivalent interval; overhangs so digits align
TRANSFER_SHARE_DECIMALS = 2                    # as in results/review_r1/rv_checks.txt (Check 1)
if set(NICE) != set(V.NAMES):
    raise SystemExit("REFUSED: NICE and rvlib.NAMES name different detectors")


def pct(x, d=1):
    """A fraction as a percentage (percent, or pp for a TTG) with d decimals; a value that rounds to zero prints unsigned."""
    if x is None:
        return "--"
    s = f"{100 * x:.{d}f}"
    return s[1:] if s.startswith("-") and not s.strip("-0.") else s


def dec(x, d=3):
    """A share, prior or score as it is, with d decimals."""
    s = f"{x:.{d}f}"
    return s[1:] if s.startswith("-") and not s.strip("-0.") else s


def iv(lo, hi, d=2):
    return f"[{pct(lo, d)}, {pct(hi, d)}]"


def span(lo, hi, d=2):
    """Range of two percentages; 'to' instead of '--' where an end is negative, so the minus sign stays unambiguous."""
    return f"{pct(lo, d)}{' to ' if min(lo, hi) < 0 else '--'}{pct(hi, d)}"


def grouped(n):
    """A count with its thousands separated by SEP from five digits up; a four-digit count has no separator (decision 10)."""
    return f"{n:,}".replace(",", SEP) if n >= 10000 else str(n)


def minus(line):
    """Typeset a negative number's sign as a minus, not a hyphen (comment lines untouched; never inside \\( \\))."""
    return line if line.startswith("%") else re.sub(r"(?<=[ /\[])-(?=\d)", "$-$", line)


def write(name, lines):
    GEN.mkdir(parents=True, exist_ok=True)
    with open(GEN / f"{name}.tex", "w", encoding="utf-8", newline="\n") as f:      # LF on every platform: the bytes are compared
        f.write("\n".join(minus(x) for x in lines) + "\n")
    print(f"wrote gen/{name}.tex")


def head(src, note):
    return [f"% generated by analysis/review_r1/rv_paper_tables.py from results/review_r1/{src}; do not edit by hand",
            f"% {note}"]


def parse(key):
    """(model, seed, target) from a cell key: 'model|s0|target' (cv, dev) or 'model|target' (the other files)."""
    p = key.split("|")
    if len(p) == 3:
        return p[0], int(p[1].lstrip("s")), p[2]
    return p[0], 0, p[1]


def ordered(cells):
    """The cells of a result file as (key, cell) pairs, in the paper's order: detectors, then run seed, then target."""
    def rank(kv):
        m, s, t = parse(kv[0])
        if m not in MODELS or t not in TARGETS:
            V.R.die(f"unexpected cell {kv[0]}")
        return MODELS.index(m), s, TARGETS.index(t)
    return sorted(cells.items(), key=rank)


def label(model, seed=0):
    """Detector name; a run of round 2 is named by its seed, and seed 1234, the pipeline check, carries a dagger as in r2_seeds."""
    if model == "aasist_l_r2":
        return f"{NICE['aasist_l']}, seed {seed}" + (r"$^\dagger$" if seed == 1234 else "")
    return NICE[model]


def agree(what, a, b, tol=1e-9):
    if abs(a - b) > tol:
        V.R.die(f"{what}: {a!r} and {b!r} differ by more than {tol}")


def groups_head(n_lead, groups):
    """A row of spanning group titles above the column labels, and its cmidrules. groups = [(title, width)]; a title of None
    leaves its columns blank."""
    cells, rules, col = [""] * n_lead, [], n_lead + 1
    for title, w in groups:
        if title is None:
            cells += [""] * w
        else:
            cells.append(rf"\multicolumn{{{w}}}{{c}}{{{title}}}")
            rules.append(rf"\cmidrule(lr){{{col}-{col + w - 1}}}")
        col += w
    return [" & ".join(cells) + r" \\", "".join(rules)]


def two(top, bottom):
    """A two-line column label."""
    return rf"\shortstack{{{top}\\{bottom}}}"


TAU_E = r"\(\tauE\)"                            # the manuscript's macro (main.tex and supplement.tex define it)


# --------------------------------------------------------------------------------------------------- main text
ROBUST = (("itw", (("xlsr_aasist", 0), ("xlsr_mamba", 0), ("as5_aasist", 0), ("pascu_lr", 0))),
          ("df21_verified", (("xlsr_aasist", 0),)),
          ("la19_eval", (("aasist", 0), ("aasist_l", 0), ("aasist_l_r2", 1), ("aasist_l_r2", 2), ("aasist_l_r2", 3),
                         ("aasist_l_r2", 4), ("aasist_l_r2", 5))))


def t_robust(cv, dev):
    """P3.2 / P4: one row per headline cell. In-sample TTG, the two cross-validated oracles, then the dev-only estimators."""
    L = head("cv.json, dev.json",
             "TTG in pp; Oracle: TTG at the frozen tau_eer against three oracles, in-sample (as the paper's TTG), leave-one-speaker-out "
             "(LOSO) and the mean [range] of 20 speaker 5-fold assignments (cv.json); other dev-only thresholds, always against "
             "the in-sample oracle (dev.json): min--max over the thresholds tied with tau_eer on dev, 95% interval over 1000 "
             "class-stratified dev resamples, KDE at Silverman bandwidth x1, isotonic (PAV), logistic (E9); rows grouped by "
             "target; the AASIST-L runs are the round-2 seeds of R2-H2")
    L += [r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\toprule"]
    L += groups_head(1, [("Oracle", 3), ("Other dev-only thresholds", 5)])
    L += [" & ".join(["Detector", "In-sample", "LOSO", two("Speaker 5-fold", "mean [range]"), two("Tied", "min--max"),
                      two("Dev bootstrap", "95\\% interval"), r"KDE \(\times1\)", "Isotonic", "Logistic"]) + r" \\",
          r"\midrule"]
    for gi, (t, rows) in enumerate(ROBUST):
        if gi:
            L.append(r"\addlinespace")
        L.append(rf"\multicolumn{{9}}{{@{{}}l}}{{\emph{{{SURF[t]}}}}} \\")
        for m, s in rows:
            k = f"{m}|s{s}|{t}"
            c, d = cv[k], dev[k]
            agree(f"TTG of {k} in cv.json and dev.json", c["ttg_in"], d["ttg_star"])
            sm, q = d["smooth"], d["devboot_ttg_q"]

            def est(e):
                if not e["in_tie"]:        # the note of Table 8 does not define the star; t_estimators still does
                    V.R.die(f"{k}: a dev-only estimate lies outside the dev-equivalent interval, so rv_robust would need "
                            "its star and the note's definition of it back")
                return pct(e["ttg"], 2)
            L.append(" & ".join([label(m, s), pct(c["ttg_in"], 2), pct(c["ttg_loso"], 2),
                                 f"{pct(c['ttg_spk5_mean'], 2)} {iv(*c['ttg_spk5_range'])}",
                                 span(d["ttg_tie_min"], d["ttg_tie_max"]), iv(q[0], q[2]),
                                 est(sm["kde1"]), est(sm["pav"]), est(sm["logistic"])]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("rv_robust", L)


# The ladder: columns are the policies of ladder.json, in the order of the paper.
LADDER = (("static", "Static"), ("logistic", "Logistic"), ("znorm", "z-norm"), ("ratematch", "Rate-match"),
          ("emquant", "EM-quantile"), ("gmm2", "GMM-2"), ("loso", "LOSO"))
LADDER_TARGETS = ("itw", "la19_eval", "df21_verified")
LADDER_K = ("50", "200")                        # small labelled audit: labelled clips per class


def t_ladder(lad, sl):
    """P6.1: TTG of each threshold policy for the four strong detectors on three targets (orientation: one row per detector
    and target, one column per policy, so that the policies read left to right as the ladder rungs)."""
    L = head("ladder.json, smalllabel.json",
             "TTG in pp against the in-sample oracle, except the last column; Static = the frozen tau_eer; Logistic = Bayes "
             "threshold of the dev-calibrated scores (E9); z-norm, rate-match, EM-quantile and GMM-2 use the unlabelled target "
             "scores; LOSO = threshold from the labelled other speakers; Static (held-out), k = 50 and k = 200 = median over 200 "
             "speaker halvings of the TTG on the held-out half, with k labelled clips per class from the other half, against "
             "that half's own oracle (smalllabel.json: ITW and LA19 eval from DEV-10 D, DF21 ver. added by DEV-11 B; -- marks "
             "a cell it lacks); Oracle = HTER of the in-sample oracle in %, its TTG being 0 by definition")
    if [int(k) for k in sl["meta"]["k_per_class"][:2]] != [int(k) for k in LADDER_K] or sl["meta"]["quantiles"] != [5, 50, 95]:
        V.R.die("smalllabel.json: k_per_class or quantiles differ from what the ladder table reads")
    L += [r"\begin{tabular}{@{}l" + "r" * (len(LADDER) + len(LADDER_K) + 2) + r"@{}}", r"\toprule"]
    L += groups_head(1, [("No target labels", 6), ("Target labels", 1), ("Audit on held-out halves", 1 + len(LADDER_K)),
                         (None, 1)])
    L += [" & ".join(["Detector"] + [n for _, n in LADDER[:-1]] + ["LOSO", two("Static", "(held-out)")]
                     + [rf"\(k={k}\)" for k in LADDER_K] + [two("Oracle", "HTER (\\%)")]) + r" \\", r"\midrule"]
    for gi, t in enumerate(LADDER_TARGETS):
        if gi:
            L.append(r"\addlinespace")
        L.append(rf"\multicolumn{{{len(LADDER) + len(LADDER_K) + 3}}}{{@{{}}l}}{{\emph{{{SURF[t]}}}}} \\")
        for m in STRONG:
            c = lad[f"{m}|{t}"]
            p = c["policies"]
            s = sl["cells"].get(f"{m}|{t}")        # None only for a cell smalllabel.json lacks (DF21 ver. before DEV-11 B)
            cells = [pct(p[k]["ttg"], 2) for k, _ in LADDER]
            cells += ["--"] * (1 + len(LADDER_K)) if s is None else [pct(s["static"][1], 2)] + [pct(s[k][1], 2) for k in LADDER_K]
            L.append(" & ".join([label(m)] + cells + [pct(c["oracle_hter"])]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("rv_ladder", L)


# ------------------------------------------------------------------------------------------------- supplement
def t_cv(cv):
    """P3.3: every cell of cv.json."""
    n = TRANSFER_SHARE_DECIMALS
    L = head("cv.json", "TTG in pp at the frozen tau_eer against four oracles: in-sample (as the paper's TTG), clip 5-fold (mean of 20 "
             "repeats, stratified by class), leave-one-speaker-out (LOSO), and speaker 5-fold (mean [range] of 20 assignments); "
             "share = TTG/HTER at tau_eer under the oracle of the column; speakers = number of speaker labels; HTER in %; "
             "dagger = seed 1234, the pipeline check of round 2, which enters no test or mean")
    L += [r"\begin{tabular}{@{}llrr" + "r" * 4 + "r" * 3 + r"@{}}", r"\toprule"]
    L += groups_head(4, [("TTG (pp)", 4), ("Share of HTER", 3)])
    L += [" & ".join(["Detector", "Target", "Speakers", "HTER", "In-sample", "Clip 5-fold", "LOSO",
                      two("Speaker 5-fold", "mean [range]"), "In-sample", "LOSO", "Speaker 5-fold"]) + r" \\", r"\midrule"]
    prev = None
    for key, c in ordered(cv):
        m, s, t = parse(key)
        if prev is not None and m != prev:
            L.append(r"\addlinespace")
        prev = m
        h = c["hter_star"]
        L.append(" & ".join([label(m, s), SURF[t], str(c["G"]), pct(h), pct(c["ttg_in"], 2), pct(c["ttg_clip5_mean"], 2),
                             pct(c["ttg_loso"], 2), f"{pct(c['ttg_spk5_mean'], 2)} {iv(*c['ttg_spk5_range'])}",
                             dec(c["ttg_in"] / h, n), dec(c["ttg_loso"] / h, n), dec(c["ttg_spk5_mean"] / h, n)]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("rvs_cv", L)


def t_estimators(dev):
    """P4.2 / P4.6: every cell of dev.json."""
    L = head("dev.json", "TTG in pp against the in-sample oracle; pins = dev bona fide trials below tau_eer / dev spoof trials "
             "at or above it; tied = min--max of the TTG over the thresholds tied with tau_eer on dev; dev bootstrap = 95% "
             "interval over 1000 class-stratified dev resamples with tau_eer re-estimated; KDE = per-class Gaussian kernels at "
             "x0.5, x1 and x2 of the Silverman bandwidth, threshold where the smoothed FRR and FAR are equal; isotonic = PAV, "
             "threshold midway between the dev scores where the posterior crosses the dev prior; logistic = E9's fit; "
             "* = estimate outside the dev-equivalent interval (in_tie false)")
    L += [r"\begin{tabular}{@{}llr" + "r" * 8 + r"@{}}", r"\toprule"]
    L += groups_head(3, [(None, 3), ("KDE bandwidth", 3), (None, 2)])
    L += [" & ".join(["Detector", "Target", "Pins", rf"TTG at {TAU_E}", two("Tied", "min--max"),
                      two("Dev bootstrap", "95\\% interval"), r"\(\times0.5\)", r"\(\times1\)", r"\(\times2\)", "Isotonic",
                      "Logistic"]) + r" \\", r"\midrule"]
    prev = None
    for key, c in ordered(dev):
        m, s, t = parse(key)
        if prev is not None and m != prev:
            L.append(r"\addlinespace")
        prev = m
        sm, q = c["smooth"], c["devboot_ttg_q"]

        def est(k):
            return pct(sm[k]["ttg"], 2) + ("" if sm[k]["in_tie"] else STAR)
        L.append(" & ".join([label(m, s), SURF[t], f"{grouped(c['dev_pins'][0])}/{grouped(c['dev_pins'][1])}",
                             pct(c["ttg_star"], 2), span(c["ttg_tie_min"], c["ttg_tie_max"]), iv(q[0], q[2]),
                             est("kde0.5"), est("kde1"), est("kde2"), est("pav"), est("logistic")]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("rvs_estimators", L)


# Column specifications of the interval table: (label, getter), and the titles that span them. The sensitivity of the
# m-out-of-n interval to m (P5.3) has its own table, rvs_mofn, so that this one stays narrow enough to read.
INTERVAL_COLS = (
    ("Speakers", lambda c: str(c["G"])),
    ("TTG", lambda c: pct(c["ttg"], 2)),
    ("95\\% CI", lambda c: iv(*c["pct"])),
    ("95\\% CI", lambda c: iv(*c["bca"])),
    (r"\(z_0\)", lambda c: f"{c['z0']:+.2f}"),
    (r"\(a\)", lambda c: f"{c['a']:+.3f}"),
    ("95\\% CI", lambda c: iv(*c["m_of_n"])),
    (r"\(m\)", lambda c: str(c["m"])),
)
INTERVAL_GROUPS = ((None, 2), ("Percentile", 1), ("BCa", 3), ("m-out-of-n", 2))      # (title, width) over INTERVAL_COLS


def t_intervals(it):
    """P5.2: every cell of intervals.json."""
    assert sum(w for _, w in INTERVAL_GROUPS) == len(INTERVAL_COLS)
    L = head("intervals.json", "TTG in pp with three 95% intervals: percentile = the speaker-cluster bootstrap of the main results "
             "(B = 1000); BCa = bias-corrected and accelerated on the same replicates, with bias constant z0 and acceleration a "
             "(leave-one-speaker-out jackknife); m-out-of-n = rescaled basic interval from 2000 draws of m = round(G^(2/3)) "
             "speakers drawn with replacement, truncated at 0; speakers = G")
    L += [r"\begin{tabular}{@{}ll" + "r" * len(INTERVAL_COLS) + r"@{}}", r"\toprule"]
    L += groups_head(2, list(INTERVAL_GROUPS))
    L += [" & ".join(["Detector", "Target"] + [name for name, _ in INTERVAL_COLS]) + r" \\", r"\midrule"]
    prev = None
    for key, c in ordered(it):
        m, s, t = parse(key)
        if prev is not None and m != prev:
            L.append(r"\addlinespace")
        prev = m
        L.append(" & ".join([label(m, s), SURF[t]] + [fn(c) for _, fn in INTERVAL_COLS]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("rvs_intervals", L)


MOFN = (("G^(1/2)", r"\(m=\mathrm{round}(G^{1/2})\)"), ("G^(2/3)", r"\(m=\mathrm{round}(G^{2/3})\)"),
        ("G/2", r"\(m=\mathrm{round}(G/2)\)"))      # mofn.json's rules for m, in increasing m


def t_mofn(mo, it):
    """P5.3: every cell of mofn.json (DEV-10 F2). Its m = round(G^(2/3)) column must reproduce intervals.json."""
    if list(mo["meta"]["m"]) != [r for r, _ in MOFN]:
        V.R.die("mofn.json: the rules for m differ from what the table reads")
    L = head("mofn.json, intervals.json", f"TTG in pp with the m-out-of-n 95% interval under three rules for m: rescaled "
             f"basic interval from {mo['meta']['B']} draws of m speakers drawn with replacement, truncated at 0; G = speakers; "
             "m = round(G^(2/3)) is the m-out-of-n interval of rvs_intervals")
    L += [r"\begin{tabular}{@{}llrr" + "rr" * len(MOFN) + r"@{}}", r"\toprule"]
    L += groups_head(4, [(title, 2) for _, title in MOFN])
    L += [" & ".join(["Detector", "Target", "Speakers", "TTG"] + [r"\(m\)", "95\\% CI"] * len(MOFN)) + r" \\", r"\midrule"]
    prev = None
    for key, c in ordered(mo["cells"]):
        m, s, t = parse(key)
        e = it[key]
        agree(f"TTG of {key} in mofn.json and intervals.json", c["ttg"], e["ttg"])
        if c["G^(2/3)"]["m"] != e["m"]:
            V.R.die(f"{key}: m = round(G^(2/3)) is {c['G^(2/3)']['m']} in mofn.json but {e['m']} in intervals.json")
        for lo_hi in (0, 1):
            agree(f"m-out-of-n interval of {key} in mofn.json and intervals.json", c["G^(2/3)"]["m_of_n"][lo_hi],
                  e["m_of_n"][lo_hi], tol=0)
        if prev is not None and m != prev:
            L.append(r"\addlinespace")
        prev = m
        cells = []
        for r, _ in MOFN:
            cells += [str(c[r]["m"]), iv(*c[r]["m_of_n"])]
        L.append(" & ".join([label(m, s), SURF[t], str(c["G"]), pct(c["ttg"], 2)] + cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("rvs_mofn", L)


# Two panels of four policies each; every policy shows FRR and FAR (%), the threshold and the TTG (pp).
PANELS = ((("static", "Static"), ("logistic", "Logistic"), ("znorm", "z-norm"), ("ratematch", "Rate-match")),
          (("emquant", "EM-quantile"), ("gmm2", "GMM-2"), ("loso", "LOSO"), ("oracle", "Oracle")))


def t_s_ladder(lad):
    """P6.2: every cell of ladder.json."""
    L = head("ladder.json", "FRR and FAR in %, threshold tau in score units and TTG in pp for each threshold policy, in two "
             "panels; Static = the frozen tau_eer; Logistic = Bayes threshold of the dev-calibrated scores (E9); z-norm, "
             "rate-match, EM-quantile and GMM-2 use the unlabelled target scores; LOSO = per-speaker thresholds chosen on the "
             "other speakers, pooled over speakers, so it has no single tau; Oracle = in-sample HTER minimum, its TTG being 0")
    cols = "ll" + "rrrr" * 4
    L += [rf"\begin{{tabular}}{{@{{}}{cols}@{{}}}}", r"\toprule"]
    for pi, panel in enumerate(PANELS):
        if pi:
            L.append(r"\midrule")
        L += groups_head(2, [(name, 4) for _, name in panel])
        L += [" & ".join(["Detector", "Target"] + ["FRR", "FAR", r"\(\tau\)", "TTG"] * len(panel)) + r" \\", r"\midrule"]
        prev = None
        for key, c in ordered(lad):
            m, s, t = parse(key)
            if prev is not None and m != prev:
                L.append(r"\addlinespace")
            prev = m
            cells = []
            for k, _ in panel:
                p = c["policies"][k]
                cells += [pct(p["frr"]), pct(p["far"]), dec(p["tau"], 2) if "tau" in p else "--", pct(p["ttg"], 2)]
            L.append(" & ".join([label(m, s), SURF[t]] + cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("rvs_ladder", L)


TER = (("static", "Static"), ("bayes_pi0.5", r"\(\pi=0.5\)"), ("bayes_em", "EM prior"), ("bayes_bbse", "BBSE prior"),
       ("bayes_trueprior", "True prior"), ("oracle", "Oracle"))


def t_labelshift(lad):
    """P6.2: every cell of ladder.json, label-shift columns."""
    L = head("ladder.json", "bona fide share of the target: true, estimated by EM (Saerens et al. 2002), by BBSE (dev FRR and "
             "FAR at tau_eer) and by the weight of the upper component of the two-component Gaussian mixture (GMM-2); total "
             "error rate in % at the true class mix with the threshold of the dev-calibrated scores that is Bayes-optimal at "
             "the prior of the column heading (pi = 0.5, the EM prior, the BBSE prior, the true prior), against the frozen "
             "tau_eer (Static) and the threshold that minimises total error on the target (Oracle)")
    L += [r"\begin{tabular}{@{}ll" + "r" * 4 + "r" * len(TER) + r"@{}}", r"\toprule"]
    L += groups_head(2, [("Bona fide share", 4), ("Total error (\\%)", len(TER))])
    L += [" & ".join(["Detector", "Target", "True", "EM", "BBSE", "GMM-2"] + [n for _, n in TER]) + r" \\", r"\midrule"]
    prev = None
    for key, c in ordered(lad):
        m, s, t = parse(key)
        if prev is not None and m != prev:
            L.append(r"\addlinespace")
        prev = m
        L.append(" & ".join([label(m, s), SURF[t], dec(c["pi_true"]), dec(c["pi_em"]), dec(c["pi_bbse"]), dec(c["pi_gmm"])]
                            + [pct(c["ter"][k]) for k, _ in TER]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("rvs_labelshift", L)


def t_smalllabel(sl):
    """P6.2: every cell of smalllabel.json."""
    meta = sl["meta"]
    if meta["quantiles"] != [5, 50, 95]:
        V.R.die("smalllabel.json: quantiles differ from 5, 50, 95")
    ks = [str(k) for k in meta["k_per_class"]]
    L = head("smalllabel.json", f"held-out TTG in pp, median [5th, 95th percentile] over {meta['splits']} speaker halvings; "
             "the labelled clips (k per class, drawn without replacement) come from one half of the speakers and the TTG is "
             "scored on the other half against that half's own in-sample oracle; static = the frozen tau_eer on the same "
             "held-out halves")
    L += [r"\begin{tabular}{@{}ll" + "r" * (1 + len(ks)) + r"@{}}", r"\toprule"]
    L += groups_head(2, [(None, 1), ("Threshold from \\(k\\) labelled clips per class", len(ks))])
    L += [" & ".join(["Detector", "Target", "Static"] + [rf"\(k={k}\)" for k in ks]) + r" \\", r"\midrule"]
    prev = None
    for key, c in ordered(sl["cells"]):
        m, s, t = parse(key)
        if prev is not None and m != prev:
            L.append(r"\addlinespace")
        prev = m
        L.append(" & ".join([label(m, s), SURF[t]] + [f"{pct(c[k][1], 2)} {iv(c[k][0], c[k][2])}" for k in ["static"] + ks])
                 + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("rvs_smalllabel", L)


def t_atc(at):
    """P7.3 (DEV-10 F3): every cell of atc.json. Accuracy is the share of target clips decided correctly at tau_eer."""
    if at["meta"]["criterion"] != "|predicted - true| <= 0.05":
        V.R.die("atc.json: the success criterion differs from the 5 pp the table states")
    L = head("atc.json", "accuracy at tau_eer in %: on the source dev set, predicted on the target without its labels by ATC "
             "(Garg et al. 2022; confidence = the margin |s - tau_eer|, threshold t in score units such that the share of dev "
             "margins above t equals the dev accuracy) and true on the target; error = predicted - true in pp; the "
             "pre-specified success criterion is an absolute error of at most 5 pp")
    L += [r"\begin{tabular}{@{}ll" + "r" * 5 + r"l@{}}", r"\toprule"]
    L += groups_head(2, [(None, 2), ("Target accuracy (\\%)", 2), (None, 2)])
    L += [" & ".join(["Detector", "Target", two("Dev", "accuracy (\\%)"), r"\(t\)", "Predicted", "True", two("Error", "(pp)"),
                      two("Within", "5 pp")]) + r" \\", r"\midrule"]
    prev = None
    for key, c in ordered(at["cells"]):
        m, s, t = parse(key)
        if prev is not None and m != prev:
            L.append(r"\addlinespace")
        prev = m
        L.append(" & ".join([label(m, s), SURF[t], pct(c["acc_dev"]), dec(c["t"], 2), pct(c["predicted"]), pct(c["true"]),
                             pct(c["error"]), "yes" if c["meets"] else "no"]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("rvs_atc", L)


def main():
    cv, dev, lad = V.read("cv")["cells"], V.read("dev")["cells"], V.read("ladder")["cells"]
    sl, it, mo, at = V.read("smalllabel"), V.read("intervals")["cells"], V.read("mofn"), V.read("atc")
    for k, c in lad.items():                          # the files must describe the same TTG at the same threshold
        m, t = k.split("|")
        agree(f"static TTG of {k} in ladder.json and cv.json", c["policies"]["static"]["ttg"], cv[f"{m}|s0|{t}"]["ttg_in"])
    t_robust(cv, dev)
    t_ladder(lad, sl)
    t_cv(cv)
    t_estimators(dev)
    t_intervals(it)
    t_mofn(mo, it)
    t_s_ladder(lad)
    t_labelshift(lad)
    t_smalllabel(sl)
    t_atc(at)


if __name__ == "__main__":
    main()
