"""LaTeX table bodies for the supplement (DEV-8), generated from the result files: nothing is typed in by hand.

    python supp_tables.py      # -> revision/paper/gen/s_*.tex   (each file is one complete tabular environment)
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

import revlib as R
from paper_tables import (A, B, C5, DET, E, FS, FT, G, H5, M5, NICE, POL, RANKED, RES, TARGETS, TNAME, ci, num, write)

EX = json.loads((RES / "extra_facts.json").read_text(encoding="utf-8"))
H14 = json.loads((R.COMMITTED / "h1_h4.json").read_text(encoding="utf-8"))
H2C = json.loads((R.COMMITTED / "h2_test.json").read_text(encoding="utf-8"))
PN = {"eer": r"$\tauE$", "frr1": r"$\tauF{1}$", "frr5": r"$\tauF{5}$", "far1": r"$\tauA{1}$", "far5": r"$\tauA{5}$"}
SHORT = {"aasist": "AASIST", "aasist_l": "AASIST-L", "rawnet2": "RawNet2", "tssdnet": "Res-TSSDNet", "xlsr_aasist": "XLS-R"}


def tabular(spec, header, rows):
    return "\\begin{tabular}{" + spec + "}\n\\toprule\n" + header + "\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}"


# ---- S-A: thresholds, margins, dev bootstrap (B1, B2)
def s_thresholds():
    rows = []
    for m in M5:
        d = DET[m]["dev"]
        t = d["taus"]
        lo, hi = B["dev_bootstrap"][m]["taus"]["eer"]
        rows.append(rf"{NICE[m]} & {d['eer_pct']:.3f} & " + " & ".join(num(t[p], 2) for p in POL) + rf" & {num(d['margin'], 2)} & {ci(lo, hi, 2)} \\")
    hdr = r"Model & Dev EER (\%) & " + " & ".join(PN[p] for p in POL) + r" & Margin & Dev-bootstrap 95\% CI of $\tauE$ \\"
    write("s_tab_thresholds.tex", tabular(r"@{}lrrrrrrrl@{}", hdr, rows))


# ---- S-B: EER-equivalent interval and its effect; eval-set threshold (B3, B4); triage shares
def s_interval():
    rows = []
    for m in M5:
        iv = DET[m]["eer_equivalent_interval"]
        cells = [rf"{NICE[m]} & {num(iv['tau_lo'], 2)} & {num(iv['tau_eer'], 2)} & {num(iv['tau_hi'], 2)}"]
        for t in ("itw", "df21_verified"):
            c = DET[m]["targets"][t]
            reg = c["policies"]["eer"]
            cells.append(f"{c['tau_lo']['frr_pct']:.1f}/{c['tau_lo']['far_pct']:.1f} & {reg['frr_pct']:.1f}/{reg['far_pct']:.1f} & "
                         f"{c['tau_hi']['frr_pct']:.1f}/{c['tau_hi']['far_pct']:.1f}")
        tr = EX["triage"][m]["itw"]
        cells.append(f"{100 * tr['genuine']:.1f}/{100 * tr['spoof']:.1f}")
        rows.append(" & ".join(cells) + r" \\")
    hdr = (r" & \multicolumn{3}{c}{Threshold} & \multicolumn{3}{c}{ITW: FRR/FAR (\%)} & \multicolumn{3}{c}{DF21: FRR/FAR (\%)} & ITW clips in \\" "\n"
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}" "\n"
           r"Model & lo & registered & hi & lo & registered & hi & lo & registered & hi & interval (\%) \\")
    write("s_tab_interval.tex", tabular(r"@{}lrrrrrrrrrr@{}", hdr, rows))


def s_evalthr():
    rows = []
    for m in M5:
        e = DET[m]["eval_threshold"]
        cells = [rf"{NICE[m]} & {e['eer_eval_pct']:.2f} & {num(e['tau'], 2)} & {num(DET[m]['dev']['taus']['eer'], 2)}"]
        for t in ("itw", "df21_verified"):
            c = DET[m]["targets"][t]
            reg = c["policies"]["eer"]
            cells.append(f"{c['tau_eval']['frr_pct']:.1f}/{c['tau_eval']['far_pct']:.1f} & {reg['frr_pct']:.1f}/{reg['far_pct']:.1f}")
        rows.append(" & ".join(cells) + r" \\")
    hdr = (r" & & \multicolumn{2}{c}{Threshold} & \multicolumn{2}{c}{ITW: FRR/FAR (\%)} & \multicolumn{2}{c}{DF21: FRR/FAR (\%)} \\" "\n"
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}" "\n"
           r"Model & LA19 eval EER (\%) & eval-set & dev & eval-set $\tau$ & dev $\tau$ & eval-set $\tau$ & dev $\tau$ \\")
    write("s_tab_evalthr.tex", tabular(r"@{}lrrrrrrr@{}", hdr, rows))


# ---- S-C: all policies on every target (C)
def s_policies():
    rows = []
    for t in TARGETS:
        rows.append(rf"\multicolumn{{13}}{{@{{}}l}}{{\emph{{{TNAME[t]}}}}} \\")
        for m in M5:
            c = DET[m]["targets"][t]
            cells = [NICE[m], f"{c['oracle_hter_pct']:.2f}", f"{c['target_oracle_frr_at_far']['0.05']['frr_pct']:.1f}"]
            for p in POL:
                q = c["policies"][p]
                cells.append(f"{q['frr_pct']:.1f}/{q['far_pct']:.1f}")
                cells.append(f"{q['ttg_pp']:.2f}")
            rows.append(" & ".join(cells) + r" \\")
        if t != TARGETS[-1]:
            rows.append(r"\midrule")
    h1 = r" & & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{PN[p]}}}" for p in POL) + r" \\"
    cm = "".join(rf"\cmidrule(lr){{{4 + 2 * i}-{5 + 2 * i}}}" for i in range(5))
    h2 = r"Model & Oracle HTER & FRR@FAR$\leq$5\% & " + " & ".join(r"FRR/FAR & TTG" for _ in POL) + r" \\"
    write("s_tab_policies.tex", tabular(r"@{}lrr" + "rr" * 5 + "@{}", h1 + "\n" + cm + "\n" + h2, rows))


# ---- S-D: TTG intervals, speaker level and registered clip level, all policies (A3)
def s_ttg():
    """TTG (pp) with the speaker-cluster 95% interval at every policy; the registered clip-level interval exists for the EER policy only."""
    rows = []
    for t in TARGETS:
        rows.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\emph{{{TNAME[t]}}} ({A['a3_e2']['ttg'][t]['clusters']} clusters)}} \\")
        for m in M5:
            cells = [NICE[m]]
            for p in POL:
                c = A["a3_e2"]["ttg"][t]["cells"][f"{m}|{p}"]
                cells.append(f"{c['ttg_pp']:.2f} {ci(*c['cluster_ci95_pp'], 2)}")
                if p == "eer":
                    cells.append(ci(*c["registered_ci95_pp"], 2) if c["registered_ci95_pp"] else "n/a")
            rows.append(" & ".join(cells) + r" \\")
        if t != TARGETS[-1]:
            rows.append(r"\midrule")
    hdr = (r" & \multicolumn{2}{c}{" + PN["eer"] + r"} & " + " & ".join(PN[p] for p in POL[1:]) + r" \\" "\n"
           r"\cmidrule(lr){2-3}" "\n"
           r"Model & TTG [speaker CI] & clip CI (registered) & " + " & ".join("TTG [speaker CI]" for _ in POL[1:]) + r" \\")
    write("s_tab_ttg_spk.tex", tabular(r"@{}lrrrrrr@{}", hdr, rows))


# ---- S-E: rank-based band and miss versus the SD-based ones (D)
def s_band():
    rows = []
    for t in TARGETS:
        for m in M5:
            c = DET[m]["targets"][t]
            q = c["policies"]["eer"]
            rows.append(rf"{TNAME[t].split(':')[0].split(',')[0]} & {NICE[m]} & {c['band_sd']:.2f} & {pct(c['band_rank']['pooled'])} & "
                        rf"{num(q['miss_sd'], 2)} & {pct(q['miss_rank']['pooled'])} \\")
        if t != TARGETS[-1]:
            rows.append(r"\midrule")
    hdr = r"Target & Model & Band (SD) & Band (\% of clips) & Miss (SD) & Miss (\% of clips) \\"
    write("s_tab_band.tex", tabular(r"@{}llrrrr@{}", hdr, rows))


# ---- S-F: H2 details (A1) and per-speaker view (A4)
def s_h2():
    rows = []
    for m in RANKED:
        r = A["a1_h2"]["per_model"][m]
        pc = H2C["per_model"][m]["p_one_sided"]
        p_clip = r"$\leq$.001" if pc == 0 else f"{pc:.3f}"
        rows.append(rf"{NICE[m]} & {num(r['delta_pp'], 2, True)} & {num(r['macro_speaker_delta_pp'], 2, True)} & "
                    rf"{num(r['loo_min_pp'], 1, True)} to {num(r['loo_max_pp'], 1, True)} & {pct(r['share_replicates_delta_positive'], 1)} & "
                    rf"{p_clip} & {r['p_one_sided']:.4f} \\")
    hdr = (r"Model & Pooled $\frr-\far$ (pp) & Speaker-macro (pp) & Delete-one-speaker range (pp) & Cluster replicates $>0$ (\%) & "
           r"$p$ (clips) & $p$ (speakers) \\")
    write("s_tab_h2.tex", tabular(r"@{}lrrrrrr@{}", hdr, rows))


def s_speakers():
    rows = []
    for m in M5:
        a = A["a4_speakers"][m]
        q = a["speaker_frr_quartiles"]
        rows.append(rf"{NICE[m]} & {pct(a['pooled_frr'])} & {pct(a['macro_frr'])} & {pct(q[0], 0)} / {pct(q[1], 0)} / {pct(q[2], 0)} & "
                    rf"{a['speakers_frr_gt_50']} & {a['speakers_frr_gt_90']} & {pct(a['share_of_rejections_from_top5_speakers'], 0)} & "
                    rf"{pct(a['share_of_genuine_clips_top5'], 0)} \\")
    hdr = (r"Model & Pooled FRR (\%) & Speaker-macro FRR (\%) & Speaker FRR quartiles (\%) & Speakers $>$50\% & Speakers $>$90\% & "
           r"Top-5 speakers: share of rejections (\%) & share of genuine clips (\%) \\")
    write("s_tab_speakers.tex", tabular(r"@{}lrrcrrrr@{}", hdr, rows))


# ---- S-G: H5 details (A2)
def s_h5():
    rows = []
    for m in RANKED:
        r = A["a2_h5"]["per_model"][m]
        h = H5["per_model"][m]
        rows.append(rf"{NICE[m]} & {100 * h['frr_clean']:.1f}$\to${100 * h['frr_encodec']:.1f} & {100 * h['far_clean']:.1f}$\to${100 * h['far_encodec']:.1f} & "
                    rf"{num(r['dfrr_pp'], 1, True)} {ci(*r['dfrr_ci95_pp'], 1, True)} & {r['p_dfrr']:.4f} & "
                    rf"{num(r['dfar_pp'], 1, True)} {ci(*r['dfar_ci95_pp'], 1, True)} & {r['p_dfar']:.4f} & {r['dfrr_bona_clips_changed']} \\")
    hdr = (r"Model & FRR (\%) clean$\to$EnCodec & FAR (\%) clean$\to$EnCodec & $\Delta$FRR (pp) [speaker 95\% CI] & $p$ & "
           r"$\Delta$FAR (pp) [speaker 95\% CI] & $p$ & Bona fide clips whose decision changed \\")
    write("s_tab_h5.tex", tabular(r"@{}lrrlrlrr@{}", hdr, rows))


# ---- S-H: H1 and H4 as registered
def s_h14():
    h1, h4 = H14["h1"], H14["h4"]
    rows = [rf"H1, ITW, 4 models $\times$ 3 policies & {h1['primary_itw']['n']} & {h1['primary_itw']['rho']:+.2f} & {ci(h1['primary_itw']['lo'], h1['primary_itw']['hi'], 2, True)} \\",
            rf"H1, ITW + DF21 & {h1['secondary_itw_df21']['n']} & {h1['secondary_itw_df21']['rho']:+.2f} & {ci(h1['secondary_itw_df21']['lo'], h1['secondary_itw_df21']['hi'], 2, True)} \\",
            rf"H4, ITW, $\log_{{10}}$ parameters & {h4['itw']['n']} & {h4['itw']['rho']:+.2f} & {ci(h4['itw']['lo'], h4['itw']['hi'], 2, True)} \\",
            rf"H4, DF21 verified & {h4['df21_verified']['n']} & {h4['df21_verified']['rho']:+.2f} & {ci(h4['df21_verified']['lo'], h4['df21_verified']['hi'], 2, True)} \\"]
    rows = [r.replace("$+", "$+") for r in rows]
    hdr = r"Exploratory correlation (Spearman) & $n$ & $\rho$ & Bootstrap 95\% CI \\"
    write("s_tab_h14.tex", tabular(r"@{}lrrl@{}", hdr, rows))


# ---- S-I: DF21 documentation (G)
def s_df21():
    rows = []
    for cond, v in G["by_condition"].items():
        rows.append(rf"\texttt{{{cond.replace('_', chr(92) + '_')}}} & {v['eval'][0]:,} & {100 * v['eval'][1]:.2f} & {v['A'][0]:,} & {v['B'][0]:,} & "
                    rf"{100 * G['class_b_share_by_condition'][cond]:.1f} & {v['verified'][0]:,} & {100 * v['verified'][1]:.2f} & {v['clean'][0]:,} \\")
    t = G["total"]
    rows.append(r"\midrule")
    rows.append(rf"all & {t['eval'][0]:,} & {100 * t['eval'][1]:.2f} & {t['A'][0]:,} & {t['B'][0]:,} & {100 * t['B'][0] / t['eval'][0]:.1f} & "
                rf"{t['verified'][0]:,} & {100 * t['verified'][1]:.2f} & {t['clean'][0]:,} \\")
    hdr = r"Condition & Eval trials & Bona fide (\%) & Class A & Class B & B share (\%) & Verified & Bona fide (\%) & Clean \\"
    write("s_tab_df21.tex", tabular(r"@{}lrrrrrrrr@{}", hdr, rows))
    txt = (RES / "g_docs.txt").read_text(encoding="utf-8")
    m = re.search(r"by source \(eval / class B / verified\): (.*)", txt)
    if not m or (RES / "g_docs_by_source.txt").read_text(encoding="utf-8").strip() != m.group(1):
        R.die("DF21 source summary differs from the packaged g_docs.txt")


# ---- S-J: silence (F1, F3)
def s_silence():
    st = pd.read_csv(R.input_file(RES / "vad_stats.csv"), dtype={"utt_id": str})
    fs = 16000.0
    rows = []
    names = {"asvspoof19_dev": "LA19 dev", "asvspoof19_eval": "LA19 eval", "itw": "ITW"}
    for ds in ("asvspoof19_dev", "asvspoof19_eval", "itw"):
        for cls, lab in (("bona", "bona fide" if ds != "itw" else "genuine"), ("spoof", "spoof")):
            d = FS["distributions"][f"{ds}|{cls}"]
            sub = st[(st.dataset == ds) & (st.label == (1 if cls == "bona" else 0))]
            med = {T: float(np.median(sub[f"lead{T}"]) / fs) for T in (25, 35, 45)}
            rows.append(rf"{names[ds]} & {lab} & {d['n']:,} & {d['dur_median_s']:.2f} & {d['lead_median_s']:.2f} [{d['lead_p10_p90_s'][0]:.2f}, "
                        rf"{d['lead_p10_p90_s'][1]:.2f}] & {100 * d['lead_lt_0.1s']:.1f} & {d['trail_median_s']:.2f} & "
                        rf"{med[25]:.2f} / {med[35]:.2f} / {med[45]:.2f} \\")
        eer = FS["silence_only_eer"][ds]
        rows.append(rf"\multicolumn{{8}}{{r}}{{\emph{{leading silence alone as a bona fide detector, EER (\%): $T=25$ {eer['25']:.1f}, "
                    rf"$T=35$ {eer['35']:.1f}, $T=45$ {eer['45']:.1f}}}}} \\")
        if ds != "itw":
            rows.append(r"\midrule")
    hdr = (r"Set & Class & $n$ & Duration (s) & Leading silence (s), median [P10, P90] & Leading $<0.1$\,s (\%) & Trailing (s) & "
           r"Leading median at $T=25/35/45$ (s) \\")
    write("s_tab_silence.tex", tabular(r"@{}llrrlrrr@{}", hdr, rows))


def s_silence_frr():
    bins = {"itw": ["0", "$<$0.2", "0.2--0.3", "0.3--0.4", r"$\geq$0.4"], "19la_eval": ["$<$0.65", "0.65--0.82", "0.82--0.99", "0.99--1.22", r"$\geq$1.22"]}
    rows = []
    for tgt, name in (("itw", "ITW genuine clips: FRR (\\%) by leading silence (s)"), ("19la_eval", "LA19 eval bona fide clips: FRR (\\%) by leading-silence quintile (s)")):
        rows.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\emph{{{name}}}}} \\")
        rows.append(r"Model & Spearman($s$, lead) & " + " & ".join(bins[tgt]) + r" & short minus long (pp) \\")
        for m in M5:
            q = FS["by_quintile"][tgt][m]
            f = [100 * x for x in q["frr_by_quintile"]]
            rows.append(rf"{NICE[m]} & {q['spearman']:+.3f} & " + " & ".join(f"{x:.1f}" for x in f) + rf" & {num(f[0] - f[-1], 1, True)} \\")
        rows.append(r"\midrule")
    rows.append(r"\multicolumn{8}{@{}l}{\emph{ITW spoofs: FAR (\%) by leading silence (s)}} \\")
    for m in M5:
        f = [100 * x for x in FS["by_quintile"][f"itw_spoof_{m}"]]
        rows.append(rf"{NICE[m]} & & " + " & ".join(f"{x:.1f}" for x in f) + r" & \\")
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{8}{@{}l}{\emph{LA19 eval spoofs: FAR (\%) by leading-silence quintile (s): $<$0.02, 0.02--0.04, 0.04--0.07, 0.07--0.5, $\geq$0.5}} \\")
    for m in M5:
        f = [100 * x for x in FS["by_quintile"][f"19la_eval_spoof_{m}"]]
        rows.append(rf"{NICE[m]} & & " + " & ".join(f"{x:.1f}" for x in f) + r" & \\")
    write("s_tab_silence_frr.tex", "\\begin{tabular}{@{}lrrrrrrr@{}}\n\\toprule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def s_trim():
    rows = []
    for tgt, name in (("19la_eval", "LA19 eval"), ("itw", "ITW")):
        rows.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\emph{{{name}: FRR/FAR \% (HTER \%, TTG pp)}}}} \\")
        for m in M5:
            c = FT[m][tgt]

            def f(x):
                return f"{100 * x['frr_fixed']:.1f}/{100 * x['far_fixed']:.1f} ({100 * x['hter_fixed']:.1f}, {100 * x['ttg']:.2f})"
            rows.append(rf"{NICE[m]} & {c['eer_untrim']:.2f}$\to${c['eer_trim']:.2f} & {f(c['reg_untrim'])} & {f(c['reg_trim'])} & {f(c['tautrim_trim'])} & "
                        rf"{100 * c['reg_untrim']['hter_oracle']:.2f}$\to${100 * c['reg_trim']['hter_oracle']:.2f} \\")
        if tgt == "19la_eval":
            rows.append(r"\midrule")
    hdr = (r"Model & EER (\%) untrimmed$\to$trimmed & $\tauE$, untrimmed audio & $\tauE$, trimmed audio & $\tau_{\mathrm{trim}}$, trimmed audio & "
           r"Oracle HTER (\%) u$\to$t \\")
    write("s_tab_trim.tex", tabular(r"@{}lrccc r@{}", hdr, rows))
    drows = []
    taus = {m: R.all_taus(m) for m in M5}
    for m in M5:
        drows.append(rf"{NICE[m]} & {FT[m]['dev_eer_untrim_pct']:.3f}$\to${FT[m]['dev_eer_trim_pct']:.3f} & {num(taus[m]['eer'], 2)}$\to${num(FT[m]['taus_trim']['eer'], 2)} & "
                     rf"{num(taus[m]['frr1'], 2)}$\to${num(FT[m]['taus_trim']['frr1'], 2)} & {num(taus[m]['far1'], 2)}$\to${num(FT[m]['taus_trim']['far1'], 2)} \\")
    write("s_tab_trim_dev.tex", tabular(r"@{}lrrrr@{}", r"Model & Dev EER (\%) & $\tauE$ & $\tauF{1}$ & $\tauA{1}$ \\", drows))


# ---- S-K: the check from genuine channel speech (A5)
def s_cheap():
    rows = []
    for m in M5:
        for k in ("50", "100", "500"):
            r = C5["results"][m][k]
            e = r["est_error_pp_med_p5_p95"]
            hf, ha = r["held_frr_new"], r["held_far_new"]
            sf, sa = r["held_frr_shipped"], r["held_far_shipped"]
            rows.append(rf"{NICE[m] if k == '50' else ''} & {k} & {num(e[0], 1, True)} [{num(e[1], 1, True)}, {num(e[2], 1, True)}] & {100 * r['share_est_within_5pp']:.0f} & "
                        rf"{sf[0]:.1f} / {sa[0]:.1f} & {hf[0]:.1f} [{hf[1]:.1f}, {hf[2]:.1f}] & {ha[0]:.1f} [{ha[1]:.1f}, {ha[2]:.1f}] \\")
        if m != M5[-1]:
            rows.append(r"\midrule")
    hdr = (r"Model & $k$ & FRR-estimate error (pp), median [5, 95\%] & Within 5\,pp (\%) & Held-out FRR / FAR at shipped $\tau$ (\%) & "
           r"Held-out FRR after re-set (\%) & Held-out FAR after re-set (\%) \\")
    write("s_tab_cheap.tex", tabular(r"@{}lrlrrll@{}", hdr, rows))


# ---- S-L: cost windows, oracle costs and stability (E1, E2)
def s_cost():
    rows = []
    for t in ("itw", "df21_verified", "19la_eval"):
        rows.append(rf"\multicolumn{{9}}{{@{{}}l}}{{\emph{{{TNAME[t]}}}}} \\")
        for m in M5:
            w = E["windows"][f"{m}|{t}"]
            cells = [NICE[m]]
            for p in POL:
                v = w[p]
                cells.append("none" if (v is None or len(v) == 0) else f"({v[0]:.2f}, {v[1]:.2f})")
            o = w["oracle"]
            pm = o["max_p_with_gain_10pct"]
            cells += [f"{o['c_screening']:.2f}", f"{o['c_forensic']:.2f}", "---" if (pm is None or (isinstance(pm, float) and np.isnan(pm))) else f"{pm:.2f}"]
            rows.append(" & ".join(cells) + r" \\")
        if t != "19la_eval":
            rows.append(r"\midrule")
    hdr = (r" & \multicolumn{5}{c}{Break-even window of bona fide priors $p$ at} & \multicolumn{3}{c}{Oracle threshold} \\" "\n"
           r"\cmidrule(lr){2-6}\cmidrule(lr){7-9}" "\n"
           r"Model & " + " & ".join(PN[p] for p in POL) + r" & $C$ screening & $C$ forensic & max $p$ with $\geq$10\% gain \\")
    write("s_tab_cost.tex", tabular(r"@{}lrrrrrrrr@{}", hdr, rows))
    srows = []
    for t in TARGETS:
        srows.append(rf"\multicolumn{{4}}{{@{{}}l}}{{\emph{{{TNAME[t]}}}}} \\")
        for p in ("eer", "frr1", "frr5"):
            s = A["a3_e2"]["stability"][t][p]
            cells = [PN[p]]
            for meas in ("hter", "screening", "forensic"):
                q = s[meas]
                cells.append(f"{SHORT[q['point_argmax']]} ({100 * q['share_point_argmax']:.1f}\\%)")
            srows.append(" & ".join(cells) + r" \\")
        if t != TARGETS[-1]:
            srows.append(r"\midrule")
    hdr2 = r"Policy & Largest HTER gap & Largest screening-cost gap & Largest forensic-cost gap \\"
    write("s_tab_stability.tex", tabular(r"@{}llll@{}", hdr2, srows))


def s_costsplit():
    """The registered cost check at tau_EER, split into the oracle cost (discrimination) and the remainder (transfer); committed values."""
    op = json.loads((R.COMMITTED / "operating_points.json").read_text(encoding="utf-8"))
    cells = {(c["model"], c["target"]): c for c in op["cells"] if c["policy"] == "eer"}
    rows = []
    for t in ("itw", "df21_verified", "19la_eval"):
        rows.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\emph{{{TNAME[t]}}}}} \\")
        for m in M5:
            c = cells[(m, t)]["costs"]
            s, fo = c["screening"], c["forensic"]
            rows.append(rf"{NICE[m]} & {s['c_norm_fixed']:.2f} & {s['c_norm_min']:.2f} & {s['ttg_cost']:.2f} & "
                        rf"{fo['c_norm_fixed']:.2f} & {fo['c_norm_min']:.2f} & {fo['ttg_cost']:.2f} \\")
        if t != "19la_eval":
            rows.append(r"\midrule")
    hdr = (r" & \multicolumn{3}{c}{Screening ($p=0.95$)} & \multicolumn{3}{c}{Forensic ($p=1/11$)} \\" "\n"
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}" "\n"
           r"Model & $C(\tauE)$ & oracle & transfer & $C(\tauE)$ & oracle & transfer \\")
    write("s_tab_costsplit.tex", tabular(r"@{}lrrrrrr@{}", hdr, rows))


if __name__ == "__main__":
    R.gate(verbose=False)
    for f in (s_thresholds, s_interval, s_evalthr, s_policies, s_ttg, s_band, s_h2, s_speakers, s_h5, s_h14, s_df21, s_silence, s_silence_frr,
              s_trim, s_cheap, s_cost, s_costsplit):
        f()
    print("wrote supplement tables")
