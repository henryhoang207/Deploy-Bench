"""Round-2 tables for the T-BIOM manuscript, generated from the committed analysis outputs (never edited by hand):

    python analysis/round2/r2_paper_tables.py      # reads results/round2/{gates.json, analysis/r2_registered.json,
                                                   #   analysis/r2_explore.json}; writes paper/tbiom/gen/r2_*.tex
r2_gates.tex      the three gates
r2_operating.tex  round-2 detectors at tau_eer on every target (E1)
r2_tests.tex      R2-H1 to R2-H4 and the reading rule
r2s_*.tex         supplement tables (seeds, channels, three-zone, languages, ASVspoof 5 conditions)
A table whose input is missing is not written (the manuscript then shows a red placeholder).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r2lib as R  # noqa: E402

GEN = R.REPO / "paper" / "tbiom" / "gen"
NICE = {"as5_aasist": "AASIST (ASVspoof 5)", "pascu_lr": "XLS-R 2B + LR", "xlsr_mamba": "XLSR-Mamba",
        "aasist": "AASIST", "aasist_l": "AASIST-L", "rawnet2": "RawNet2", "tssdnet": "Res-TSSDNet",
        "xlsr_aasist": "XLS-R+AASIST"}
SURF = {"la19_eval": "LA19 eval", "itw": "ITW", "df21_verified": "DF21 ver.", "mlaad5": "MLAAD-5", "as5_eval": "AS5 eval"}
COND = {"opus6k": "Opus 6k", "opus24k": "Opus 24k", "amr4.75": "AMR 4.75k", "mp3_32": "MP3 32k", "snr20": "Noise 20 dB",
        "snr10": "Noise 10 dB", "snr5": "Noise 5 dB", "mic_ir": "Phone IR", "encodec6k": "EnCodec 6k"}


def load(rel):
    p = R.R2 / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pct(x, d=1):
    return "--" if x is None else f"{100 * x:.{d}f}"


def write(name, lines):
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote gen/{name}.tex")


def t_gates(g):
    L = [r"\begin{tabular}{@{}llrrc@{}}", r"\toprule", r"Model & Gate & Ours & Reference & Pass \\", r"\midrule"]
    a = g["as5_aasist"]
    L.append(rf"AASIST (ASVspoof 5) & AS5 dev EER (\%) & {a['eer_pct']:.2f} & 15.2 & {'yes' if a['passed'] else 'no'} \\")
    x = g["xlsr_mamba"]
    L.append(rf"XLSR-Mamba & DF21 EER, same {x['n_overlap']:,} trials (\%) & {x['eer_ours_pct']:.2f} & "
             rf"{x['eer_theirs_pct']:.2f} & {'yes' if x['passed'] else 'no'} \\")
    p = g["pascu_lr"]
    L.append(rf"XLS-R 2B + LR & LA19-eval EER, train+dev fit (\%) & {p['eer_pct']:.2f} & 0.6 & {'yes' if p['passed'] else 'no'} \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("r2_gates", L)


def t_operating(e1):
    L = [r"\begin{tabular}{@{}llrrrrrrrr@{}}", r"\toprule",
         r"Model & Target & EER & FRR & FAR & HTER & Oracle & TTG [95\% CI] & Share & Tie-interval FRR \\", r"\midrule"]
    for m in R.DETECTORS_R2:
        for s in SURF:
            c = e1.get(f"{m}|{s}")
            if not c:
                continue
            p = c["policies"]["eer"]
            tie = c["tie"]["frr"]
            L.append(rf"{NICE[m]} & {SURF[s]} & {pct(c['eer'])} & {pct(p['frr'])} & {pct(p['far'])} & {pct(p['hter'])} & "
                     rf"{pct(p['hter_oracle'])} & {pct(p['ttg'], 2)} [{pct(p['ttg_ci95'][0], 2)}, {pct(p['ttg_ci95'][1], 2)}] & "
                     rf"{100 * p['transfer_share']:.0f} & {pct(min(tie))}--{pct(max(tie))} \\")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("r2_operating", L)


def t_tests(r):
    L = [r"\begin{tabular}{@{}lll@{}}", r"\toprule", r"Hypothesis & Per unit & Outcome \\", r"\midrule"]
    h1 = r["R2-H1"]
    per = "; ".join(
        (f"{NICE[m]}: {pct(v['frr1']['frr_minus_far_pp'] / 100, 1)}/{pct(v['far5']['far_minus_frr_pp'] / 100, 1)}"
         + (r"$^\ast$" if v.get("supports") else "")) if v.get("evaluable") else f"{NICE[m]}: n/a"
        for m, v in h1["per_model"].items())
    L.append(rf"R2-H1 policy sets direction & {per} & {h1['n_supporting']}/3, {'supported' if h1['supported'] else 'not supported'} \\")
    h2 = r["R2-H2"]
    per = ", ".join(f"s{k}: {v['ttg_pp']:.2f}" + (r"$^\ast$" if v.get("holm_reject") else "") if v.get("evaluable")
                    else f"s{k}: n/a" for k, v in h2["per_run"].items())
    L.append(rf"R2-H2 TTG $>$ 1\,pp (pp) & {per} & {h2['n_rejected']}/5, {'supported' if h2['supported'] else 'not supported'} \\")
    h3 = r["R2-H3"]
    if h3.get("evaluable"):
        t = h3["tests"]
        per = (f"dev EER {t['a_dev_eer_higher_when_trimmed']['diff_pp']:+.2f} (p={t['a_dev_eer_higher_when_trimmed']['p']:.3f}); "
               f"ITW FRR {-t['b_itw_frr_lower_when_trimmed']['diff_pp']:+.2f} (p={t['b_itw_frr_lower_when_trimmed']['p']:.3f}); "
               f"ITW EER {-t['c_itw_eer_lower_when_trimmed']['diff_pp']:+.2f} (p={t['c_itw_eer_lower_when_trimmed']['p']:.3f})")
        out = "; ".join(k[0] + (" yes" if v.get("holm_reject") else " no") for k, v in t.items() if "holm_reject" in v)
        L.append(rf"R2-H3 trimmed $-$ untrimmed & {per} & {out} \\")
    else:
        L.append(r"R2-H3 trimmed $-$ untrimmed & -- & not evaluable \\")
    h4 = r["R2-H4"]
    per = ", ".join(f"{COND[c]} {v['n_rejected']}/4" for c, v in h4["conditions"].items())
    L.append(rf"R2-H4 channels & {per} & {h4['n_conditions_supported']}/8 conditions \\")
    rl = r["R2-RL"]
    per = "; ".join(f"{NICE[m]}: {v['category']}" for m, v in rl.items())
    L.append(rf"R2-RL reading rule & {per} & -- \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("r2_tests", L)


def main():
    g = load("gates.json")
    if g:
        t_gates(g)
    ex = load("analysis/r2_explore.json")
    if ex and "E1_E8_E10" in ex:
        t_operating(ex["E1_E8_E10"])
    rg = load("analysis/r2_registered.json")
    if rg:
        t_tests(rg)


if __name__ == "__main__":
    main()
