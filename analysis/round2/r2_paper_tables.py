"""Round-2 tables for the Paper 1 manuscript (Speech Communication, D59), generated from the committed analysis
outputs (never edited by hand):

    python analysis/round2/r2_paper_tables.py      # reads results/round2/{gates.json, analysis/r2_registered.json,
                                                   #   analysis/r2_explore.json}; writes paper/speechcomm/gen/r2_*.tex
#   ($R2_GEN overrides the output folder)
r2_gates.tex        the three gates
r2_operating.tex    round-2 detectors at tau_eer on every target (E1)
r2_tests.tex        R2-H1 to R2-H4 and the reading rule
r2_seeds.tex        the eleven AASIST-L runs and the released checkpoint (E2, E3)
r2_channels.tex     channel effects at tau_eer, dFRR/dFAR in pp (E4; round-1 rows are R2-H4's point estimates)
r2_mlaad.tex        MLAAD-5 by language at tau_eer (E5)
r2_as5.tex          ASVspoof 5 eval by codec condition (E6)
r2_threezone.tex    three-zone review on ITW at the 10% dev budget (E7); r2s_threezone.tex: every target
r2_calibration.tex  actual / minimum normalised cost at pi = 0.5 after logistic calibration on source dev (E9)
A table whose input is missing is not written (the manuscript then shows a red placeholder).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r2lib as R  # noqa: E402

GEN = Path(os.environ.get("R2_GEN", R.REPO / "paper" / "speechcomm" / "gen"))
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


MODELS8 = ("as5_aasist", "pascu_lr", "xlsr_mamba", "aasist", "aasist_l", "rawnet2", "tssdnet", "xlsr_aasist")
CONDS9 = ("opus6k", "opus24k", "amr4.75", "mp3_32", "snr20", "snr10", "snr5", "mic_ir", "encodec6k")


def head(src, note):
    return [f"% generated by analysis/round2/r2_paper_tables.py from results/round2/{src}; do not edit by hand",
            f"% {note}"]


def t_seeds(e23, gates):
    runs = (gates or {}).get("runs", {})
    L = head("analysis/r2_explore.json (E2_E3), gates.json (runs)",
             "rates in %, TTG in pp, at tau_eer from each run's own LA19 dev; SWA = recipe snapshots averaged; "
             "last = last-epoch weights (E3; identical to swa for the trimmed arm, so its Last columns are --); "
             "the trimmed arm never reached dev EER <= 1%, so R2-H3 compares SWA models with last-epoch models")
    L += [r"\begin{tabular}{@{}llrrrrrrrrrr@{}}", r"\toprule",
          r"Arm & Seed & SWA & Dev EER & LA19 EER & LA19 TTG & ITW EER & ITW FRR & ITW TTG & Last: LA19 TTG & "
          r"Last: ITW TTG & Last: ITW FRR \\", r"\midrule"]
    rel = e23["released_aasist_l"]
    L.append(rf"Released AASIST-L & -- & -- & -- & {pct(rel['la19_eval']['eer'], 2)} & {pct(rel['la19_eval']['ttg'], 2)} & "
             rf"{pct(rel['itw']['eer'])} & {pct(rel['itw']['frr'])} & {pct(rel['itw']['ttg'], 2)} & -- & -- & -- \\")
    for arm, name in (("aasist_l_r2", "As released"), ("aasist_l_trim", "Trimmed")):
        L.append(r"\addlinespace")
        rows = e23.get(arm, {})
        last = e23.get(arm + "_last", {})
        for k, v in rows.items():
            n_swa = runs.get(f"{arm}_s{k}", {}).get("n_swa_update")
            lt = last.get(k, {})
            l_la = pct(lt["la19_eval|eer"]["ttg"], 2) if arm == "aasist_l_r2" and "la19_eval|eer" in lt else "--"
            l_it = pct(lt["itw|eer"]["ttg"], 2) if arm == "aasist_l_r2" and "itw|eer" in lt else "--"
            l_fr = pct(lt["itw|eer"]["frr"]) if arm == "aasist_l_r2" and "itw|eer" in lt else "--"
            seed = f"{k}" + (r"$^\dagger$" if k == "1234" else "")
            L.append(rf"{name} & {seed} & {'--' if n_swa is None else n_swa} & {pct(v['dev_eer'], 2)} & "
                     rf"{pct(v['la19_eval|eer']['eer'], 2)} & {pct(v['la19_eval|eer']['ttg'], 2)} & "
                     rf"{pct(v['itw|eer']['eer'])} & {pct(v['itw|eer']['frr'])} & {pct(v['itw|eer']['ttg'], 2)} & "
                     rf"{l_la} & {l_it} & {l_fr} \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("r2_seeds", L)


def t_channels(e4):
    L = head("analysis/r2_explore.json (E4)", "dFRR/dFAR in pp at tau_eer against the clean copy of the 1,000-clip set; "
             "arm rows: mean over seeds 1-5; round-1 rows equal R2-H4's point estimates; -- = not scored")
    L += [r"\begin{tabular}{@{}l" + "r" * len(CONDS9) + r"@{}}", r"\toprule",
          "Model & " + " & ".join(COND[c] for c in CONDS9) + r" \\", r"\midrule"]

    def sg(x):                                 # signed whole pp; a value that rounds to zero prints as 0, not -0
        v = round(100 * x)
        return "0" if v == 0 else f"{v:+d}"

    def cell(d):
        return "--" if d is None else f"{sg(d['dfrr'])}/{sg(d['dfar'])}"
    for m in MODELS8:
        r = e4.get(f"{m}|s0")
        if r:
            L.append(NICE[m] + " & " + " & ".join(cell(r.get(c)) for c in CONDS9) + r" \\")
        if m == "xlsr_mamba":
            L.append(r"\addlinespace")
    L.append(r"\addlinespace")
    for arm, name in (("aasist_l_r2", "AASIST-L runs, as released"), ("aasist_l_trim", "AASIST-L runs, trimmed")):
        rs = [e4[f"{arm}|s{k}"] for k in (1, 2, 3, 4, 5) if f"{arm}|s{k}" in e4]
        if not rs:
            continue
        cells = []
        for c in CONDS9:
            v = [r[c] for r in rs if c in r]
            cells.append("--" if not v else f"{sg(sum(x['dfrr'] for x in v) / len(v))}/"
                                            f"{sg(sum(x['dfar'] for x in v) / len(v))}")
        L.append(name + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("r2_channels", L)


def t_mlaad(e5):
    langs = ("de", "en", "fr", "it", "pl")
    L = head("analysis/r2_explore.json (E5)", "FRR/FAR in % at tau_eer by language; every MLAAD-5 spoof is 22.05 kHz and "
             "98.1% of its bona fide clips are 16 kHz (sample-rate confound)")
    L += [r"\begin{tabular}{@{}l" + "r" * len(langs) + r"@{}}", r"\toprule",
          "Model & " + " & ".join(langs) + r" \\", r"\midrule"]
    for m in R.DETECTORS_R2:
        r = e5.get(m)
        if r:
            L.append(NICE[m] + " & " + " & ".join(
                (f"{pct(r[g]['frr'], 0)}/{pct(r[g]['far'], 0)}" if g in r else "--") for g in langs) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("r2_mlaad", L)


def t_as5(e6):
    L = head("analysis/r2_explore.json (E6)", "AASIST trained on ASVspoof 5, tau_eer from AS5 dev; rates in %, TTG in pp; "
             "C00 = no codec")
    L += [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule", r"Condition & Trials & EER & FRR & FAR & TTG \\", r"\midrule"]
    for c, v in e6.items():
        if not c.startswith("C"):
            continue
        L.append(rf"{c} & {v['n']:,} & {pct(v['eer'])} & {pct(v['frr'])} & {pct(v['far'])} & {pct(v['ttg'], 2)} \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("r2_as5", L)


def t_threezone(e7, surfaces, name):
    L = head("analysis/r2_explore.json (E7)", "10% dev review budget centred in rank on tau_eer; % of target clips; "
             "label-free: the same 10% re-anchored on the target's unlabelled scores; error-budget rule: review band "
             "[tau_frr1, tau_far1), which collapses when the two cross on near-separable dev")
    L += [r"\begin{tabular}{@{}llrrrrrrl@{}}", r"\toprule",
          r"Model & Target & Review & Review (genuine) & Genuine auto-rej. & Spoof auto-acc. & "
          r"Label-free review & Label-free genuine auto-rej. & Error-budget rule \\", r"\midrule"]
    for m in MODELS8:
        for surf in surfaces:
            c = e7.get(f"{m}|{surf}")
            if not c:
                continue
            d, f, eb = c["0.1"]["dev"], c["0.1"]["label_free"], c["error_budget"]
            ebs = "collapses" if eb["collapse"] else f"review {pct(eb['review_pooled'])}"
            L.append(rf"{NICE[m]} & {SURF[surf]} & {pct(d['review_pooled'])} & {pct(d['review_bona'])} & "
                     rf"{pct(d['bona_auto_rejected'])} & {pct(d['spoof_auto_accepted'])} & {pct(f['review_pooled'])} & "
                     rf"{pct(f['bona_auto_rejected'])} & {ebs} \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write(name, L)


def t_calibration(e9):
    surfs = ("la19_eval", "itw", "df21_verified", "mlaad5", "as5_eval")
    L = head("analysis/r2_explore.json (E9)", "actual/minimum normalised cost at pi = 0.5 after an exact logistic "
             "calibration on source dev (DEV-9 amendment 2), prior-corrected; 1 = the better trivial system")
    L += [r"\begin{tabular}{@{}l" + "r" * len(surfs) + r"@{}}", r"\toprule",
          "Model & " + " & ".join(SURF[x] for x in surfs) + r" \\", r"\midrule"]
    for m in MODELS8:
        cells = []
        for x in surfs:
            c = e9.get(f"{m}|{x}")
            mid = [r for r in c["rows"] if r[0] == 0.0][0] if c else None
            cells.append("--" if not mid or mid[1] is None else f"{mid[1]:.2f}/{mid[2]:.2f}")
        if any(c != "--" for c in cells):
            L.append(NICE[m] + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("r2_calibration", L)


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
    if ex:
        if "E2_E3" in ex:
            t_seeds(ex["E2_E3"], g)
        if "E4" in ex:
            t_channels(ex["E4"])
        if "E5" in ex:
            t_mlaad(ex["E5"])
        if "E6" in ex:
            t_as5(ex["E6"])
        if "E7" in ex:
            t_threezone(ex["E7"], ("itw",), "r2_threezone")
            t_threezone(ex["E7"], tuple(SURF), "r2s_threezone")
        if "E9" in ex:
            t_calibration(ex["E9"])


if __name__ == "__main__":
    main()
