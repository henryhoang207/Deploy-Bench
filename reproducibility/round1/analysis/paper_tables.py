"""LaTeX table bodies for the revised paper and its supplement, generated from the result files (DEV-8).

    python paper_tables.py      # -> revision/paper/gen/*.tex   (each file is one complete tabular environment)

Nothing is typed in by hand: every number is read from revision/results/*.json or from the committed score-derived JSON files
through revlib (whose gate has already reproduced the committed numbers to 1e-12). The manuscript \\input{}s these files.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import revlib as R

GEN = Path(os.environ.get("DFV_GEN", R.REV / "paper" / "gen"))
RES = R.RESULTS
NICE = {"aasist": "AASIST", "aasist_l": "AASIST-L", "rawnet2": "RawNet2", "tssdnet": "Res-TSSDNet", "xlsr_aasist": r"XLS-R (ref.)"}
M5 = R.MODELS
RANKED = R.RANKED
TARGETS = ("19la_eval", "itw", "df21_verified")
TNAME = {"19la_eval": "LA19 eval: seen corpus, unseen attacks", "itw": "In-the-Wild", "df21_verified": "DF21, verified subset"}
POL = ("eer", "frr1", "frr5", "far1", "far5")

B = json.loads((RES / "b_thresholds.json").read_text(encoding="utf-8"))
A = json.loads((RES / "a_cluster.json").read_text(encoding="utf-8"))
E = json.loads((RES / "e_cost.json").read_text(encoding="utf-8"))
C5 = json.loads((RES / "a5_cheapcheck.json").read_text(encoding="utf-8"))
G = json.loads((RES / "g_docs.json").read_text(encoding="utf-8"))
FS = json.loads((RES / "f_vad_stats_analysis.json").read_text(encoding="utf-8"))
FT = json.loads((RES / "f_vad_scores_analysis.json").read_text(encoding="utf-8"))
H5 = json.loads((R.COMMITTED / "h5_test.json").read_text(encoding="utf-8"))
DET = B["deterministic"]


def num(x, nd=1, plus=False):
    """Number in text mode with a typographic minus (and an explicit plus if asked)."""
    s = f"{x:+.{nd}f}" if plus else f"{x:.{nd}f}"
    return s.replace("-", "$-$").replace("+", "$+$")


def ci(lo, hi, nd=1, plus=False):
    return f"[{num(lo, nd, plus)}, {num(hi, nd, plus)}]"


def pval(p, B_):
    floor = 1.0 / (B_ + 1)
    if p <= floor * 1.0001:
        return rf"$\leq${floor:.4f}".replace("0.", ".", 1) if B_ >= 10000 else r"$\leq$.001"
    return f"{p:.3f}".lstrip("0") if p < 1 else "1.00"


def write(name, text):
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / name).write_text(text, encoding="utf-8", newline="\n")


# ---------------------------------------------------------------- main text: Table II (roster and dev thresholds)
def tab_roster():
    pub = {"aasist": ("0.83", "0.83"), "aasist_l": ("0.99", "0.99"), "rawnet2": ("4.60", r"5.64\textsuperscript{a}"),
           "tssdnet": ("1.62", "1.66"), "xlsr_aasist": ("0.15", r"---\textsuperscript{b}")}
    params = {"aasist": "298K", "aasist_l": "85K", "rawnet2": "17.6M", "tssdnet": "349K", "xlsr_aasist": "315M"}
    rows = []
    for m in M5:
        d = DET[m]["dev"]
        rows.append(rf"{NICE[m]} & {params[m]} & {pub[m][0]} / {pub[m][1]} & {d['eer_pct']:.3f} & {num(d['taus']['eer'], 2)} & "
                    rf"{d['n_bona_below_tau_eer']} / {d['n_spoof_at_or_above_tau_eer']} \\")
    t = (r"\begin{tabular}{@{}lrcrrc@{}}" "\n" r"\toprule" "\n"
         r"Model & Params & Eval EER & Dev EER & $\tauE$ & Dev trials \\" "\n" r"\midrule" "\n"
         + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    write("tab_roster.tex", t)


# ---------------------------------------------------------------- main text: Table III (operating behaviour at tau_EER)
def tab_operating():
    rows = []
    for t in TARGETS:
        n_cl = A["a3_e2"]["ttg"][t]["clusters"]
        unit = {"19la_eval": "speakers", "itw": "speaker labels", "df21_verified": "speakers"}[t]
        rows.append(rf"\multicolumn{{9}}{{@{{}}l}}{{\emph{{{TNAME[t]}}} ({n_cl} {unit})}} \\")
        for m in M5:
            c = DET[m]["targets"][t]
            p = c["policies"]["eer"]
            lo, hi = A["a3_e2"]["ttg"][t]["cells"][f"{m}|eer"]["cluster_ci95_pp"]
            d = 2 if t == "19la_eval" else 1
            rows.append(rf"{NICE[m]} & {c['eer_pct']:.{d}f} & {p['frr_pct']:.{d}f} & {p['far_pct']:.{d}f} & {p['hter_pct']:.{d}f} & {c['oracle_hter_pct']:.{d}f} & "
                        rf"{p['ttg_pp']:.2f} & {ci(lo, hi, 2)} & {100 * p['transfer_share']:.0f} \\")
        if t != TARGETS[-1]:
            rows.append(r"\midrule")
    t_ = (r"\begin{tabular}{@{}lrrrrrrlr@{}}" "\n" r"\toprule" "\n"
          r"Model & EER & FRR & FAR & HTER & Oracle & TTG & 95\% CI (speaker bootstrap) & TTG/HTER \\" "\n" r"\midrule" "\n"
          + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    write("tab_operating.tex", t_)


# ---------------------------------------------------------------- main text: Table IV (confirmatory tests, both resampling units)
def tab_tests():
    a1, a2 = A["a1_h2"], A["a2_h5"]
    rows = [r"\multicolumn{6}{@{}l}{\emph{H2: ITW at $\tauE$, FRR$-$FAR (pp)}} \\"]
    h2c = json.loads((R.COMMITTED / "h2_test.json").read_text(encoding="utf-8"))["per_model"]
    h5c = H5["per_model"]
    for m in RANKED:
        r = a1["per_model"][m]
        rows.append(rf"{NICE[m]} & {num(r['delta_pp'], 1, True)} & {ci(*r['registered_ci95_pp'], 1, True)} & "
                    rf"{'yes' if h2c[m]['holm_reject_h0'] else 'no'} & "
                    rf"{ci(*r['ci95_pp'], 1, True)} & {'yes' if r['holm_reject_h0'] else 'no'} \\")
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{6}{@{}l}{\emph{H5: clean $\to$ EnCodec at $\tauE$, $\Delta$FRR$-\Delta$FAR (pp)}} \\")
    for m in RANKED:
        r = a2["per_model"][m]
        rows.append(rf"{NICE[m]} & {num(r['stat_pp'], 1, True)} & {ci(*r['registered_ci95_pp'], 1, True)} & "
                    rf"{'yes' if h5c[m]['holm_reject_h0'] else 'no'} & "
                    rf"{ci(*r['stat_ci95_pp'], 1, True)} & {'yes' if r['holm_stat'] else 'no'} \\")
    t_ = (r"\begin{tabular}{@{}lrcccc@{}}" "\n" r"\toprule" "\n"
          r" & Estimate & \multicolumn{2}{c}{Clips (registered)} & \multicolumn{2}{c}{Speakers (this revision)} \\" "\n"
          r"\cmidrule(lr){3-4}\cmidrule(l){5-6}" "\n"
          r"Model & (pp) & 95\% CI & Rej. & 95\% CI & Rej. \\" "\n" r"\midrule" "\n"
          + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    write("tab_tests.tex", t_)


if __name__ == "__main__":
    R.gate(verbose=False)
    tab_roster()
    tab_operating()
    tab_tests()
    print("wrote main-text tables to", GEN.name)
