"""Claim-by-claim number audit of paper/main.tex (ARS revision, final check).

    python audit_claims.py

For every statement of the paper that carries a number, the expected phrase is BUILT FROM THE RESULT FILES here (never typed from memory) and
must appear in the LaTeX source after both are normalised (comments, math delimiters, thin spaces and all whitespace removed). A mismatch
means that the paper and the results disagree (or that the wording changed, in which case the claim below is updated after the numbers have
been re-checked by hand). Numbers quoted from other papers or from project documents are listed separately as `cited` and are not computed.
"""
from __future__ import annotations

import json
import re
import sys

import numpy as np

import revlib as R

PAPER = R.REV / "paper"
RES = R.RESULTS
M5, RK = list(R.MODELS), list(R.RANKED)
B = json.loads((RES / "b_thresholds.json").read_text(encoding="utf-8"))
A = json.loads((RES / "a_cluster.json").read_text(encoding="utf-8"))
E = json.loads((RES / "e_cost.json").read_text(encoding="utf-8"))
C5 = json.loads((RES / "a5_cheapcheck.json").read_text(encoding="utf-8"))
G = json.loads((RES / "g_docs.json").read_text(encoding="utf-8"))
FS = json.loads((RES / "f_vad_stats_analysis.json").read_text(encoding="utf-8"))
FT = json.loads((RES / "f_vad_scores_analysis.json").read_text(encoding="utf-8"))
EX = json.loads((RES / "extra_facts.json").read_text(encoding="utf-8"))
H5 = json.loads((R.COMMITTED / "h5_test.json").read_text(encoding="utf-8"))
D = B["deterministic"]


def tgt(m, t):
    return D[m]["targets"][t]


def pol(m, t, p="eer"):
    return D[m]["targets"][t]["policies"][p]


def rng(vals, nd, scale=1.0):
    lo, hi = min(vals) * scale, max(vals) * scale
    return f"{lo:.{nd}f}-{hi:.{nd}f}"


def f(x, nd=1):
    return f"{x:.{nd}f}"


def sgn(x, nd=1):
    return f"{x:+.{nd}f}"


def norm_tex(s):
    s = re.sub(r"(?m)(?<!\\)%.*$", "", s)
    s = s.replace("\\%", "%").replace("\\(", "").replace("\\)", "").replace("\\,", "").replace("~", "")
    s = s.replace("{,}", ",").replace("$-$", "-").replace("$+$", "+").replace("--", "-")
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = re.sub(r"[{}$\\]", "", s)
    return re.sub(r"\s+", "", s)


def norm_claim(s):
    return re.sub(r"\s+", "", s.replace("--", "-"))


def main():
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    body = norm_tex(tex.split(r"\begin{document}", 1)[1])
    fails, n = [], 0

    def chk(name, text):
        nonlocal n
        n += 1
        if norm_claim(text) not in body:
            fails.append((name, text))

    ttg = lambda m, t, p="eer": pol(m, t, p)["ttg_pp"]
    cl = lambda m, t, p="eer": A["a3_e2"]["ttg"][t]["cells"][f"{m}|{p}"]["cluster_ci95_pp"]

    # ---- abstract
    chk("abs.clusters", f"{A['a1_h2']['clusters']} speaker labels")
    chk("abs.ttg", f"AASIST-L {f(ttg('aasist', '19la_eval'))} and {f(ttg('aasist_l', '19la_eval'))} HTER points")
    iv = D["xlsr_aasist"]["targets"]["itw"]
    chk("abs.xlsr_range", f"ranges from {f(iv['tau_lo']['frr_pct'], 0)}% to {f(iv['tau_hi']['frr_pct'], 0)}%")
    # ---- protocol
    dv = {m: D[m]["dev"] for m in M5}
    chk("proto.dev_n", f"({dv['aasist']['n_bona'] + dv['aasist']['n_spoof']:,} trials, {dv['aasist']['n_bona']:,} bona fide)")
    pr_b = [dv[m]["n_bona_below_tau_eer"] for m in RK]
    pr_s = [dv[m]["n_spoof_at_or_above_tau_eer"] for m in RK]
    chk("proto.pins", f"{min(pr_b)}-{max(pr_b)} bona fide and {min(pr_s)}-{max(pr_s)} spoof trials lie on its wrong side for the primary models, and "
                      f"{dv['xlsr_aasist']['n_bona_below_tau_eer']} and {dv['xlsr_aasist']['n_spoof_at_or_above_tau_eer']} for the reference")
    # ---- targets
    ids, y, s = R.load("aasist", "19la_eval")
    chk("tgt.la19", f"({len(y):,} trials, {int((y == 1).sum()):,} bona fide)")
    ids, y, s = R.load("aasist", "itw")
    a4 = A["a4_speakers"]["aasist"]["clips_per_speaker_min_med_max"]
    chk("tgt.itw", f"{len(y):,} clips ({int((y == 1).sum()):,} bona fide)")
    chk("tgt.itw_labels", f"{A['a1_h2']['clusters']} distinct speaker labels (the dataset paper reports 58), with {a4[0]} to {a4[2]:,} clips each")
    gt = G["total"]
    n_eval, n_b, n_ver = gt["eval"][0], gt["B"][0], gt["verified"][0]
    n_a = gt["A"][0]
    chk("tgt.df21_reject", f"rejects {100 * (n_a + n_b) / n_eval:.0f}% of the {n_eval:,} evaluation files, and for {n_b:,} files")
    bona_eval = round(gt["eval"][1] * n_eval)
    bona_b = round(gt["B"][1] * n_b)
    bona_ver = round(gt["verified"][1] * n_ver)
    chk("tgt.df21_verified", f"verified subset of {n_ver:,} trials ({bona_ver:,} bona fide)")
    chk("tgt.df21_excl", f"remove {100 * bona_b / bona_eval:.1f}% of bona fide and {100 * (n_b - bona_b) / (n_eval - bona_eval):.1f}% of spoof trials")
    hm = G["h5_manifest"]
    sp = list(hm["attacks"].values())
    chk("tgt.h5_manifest", f"covers all {hm['speakers']} speakers and {min(sp)}-{max(sp)} spoofs of each attack")
    chk("tgt.attacks", "eleven are unseen (A07-A15, A17, A18) and two are known (A16 = A04, A19 = A06)")
    # ---- IV-A
    la = "19la_eval"
    chk("a.la_far_frr", f"accept {rng([pol(m, la)['far_pct'] for m in RK], 1)}% of spoofs and reject {rng([pol(m, la)['frr_pct'] for m in RK], 1)}% of genuine clips")
    chk("a.la_ttg", f"(TTG {f(ttg('aasist', la), 2)} and {f(ttg('aasist_l', la), 2)}pp, speaker-level 95% CIs "
                    f"[{f(cl('aasist', la)[0], 2)}, {f(cl('aasist', la)[1], 2)}] and [{f(cl('aasist_l', la)[0], 2)}, {f(cl('aasist_l', la)[1], 2)}]), "
                    f"{100 * pol('aasist', la)['transfer_share']:.0f}% and {100 * pol('aasist_l', la)['transfer_share']:.0f}% of their deployed HTER")
    chk("a.la_eer_hter", f"the EER hides it ({f(tgt('aasist', la)['eer_pct'], 2)}% against {f(pol('aasist', la)['hter_pct'], 2)}%)")
    chk("a.la_rank", f"Res-TSSDNet ({f(pol('tssdnet', la)['hter_pct'], 2)}%) overtakes AASIST ({f(pol('aasist', la)['hter_pct'], 2)}%) and AASIST-L ({f(pol('aasist_l', la)['hter_pct'], 2)}%)")
    it = "itw"
    chk("a.itw_eer", f"their EERs are {rng([tgt(m, it)['eer_pct'] for m in RK], 0)}%, their oracle HTERs {rng([tgt(m, it)['oracle_hter_pct'] for m in RK], 0)}%, "
                     f"and transfer is only {rng([pol(m, it)['transfer_share'] for m in RK], 0, 100)}% of the deployed HTER")
    x = "xlsr_aasist"
    chk("a.xlsr", f"whose EER ({f(tgt(x, it)['eer_pct'])}% on ITW, {f(tgt(x, 'df21_verified')['eer_pct'])}% on DF21) is far below its deployed HTER "
                  f"({f(pol(x, it)['hter_pct'])}% and {f(pol(x, 'df21_verified')['hter_pct'])}%), of which {100 * pol(x, it)['transfer_share']:.0f}% and "
                  f"{100 * pol(x, 'df21_verified')['transfer_share']:.0f}% is transfer")
    chk("a.fewest", f"(HTER {f(pol(x, it)['hter_pct'])}% against {rng([pol(m, it)['hter_pct'] for m in RK], 1)}% on ITW, {f(pol(x, 'df21_verified')['hter_pct'])}% against "
                    f"{rng([pol(m, 'df21_verified')['hter_pct'] for m in RK], 1)}% on DF21)")
    st = A["a3_e2"]["stability"]
    chk("a.stable_df21", f"in at least {min(st['df21_verified']['eer']['hter']['share_point_argmax'], 1) * 100:.1f}% on DF21".replace(".5%", ".5%"))
    chk("a.stable_itw", "largest in every speaker-bootstrap replicate on ITW" if st["itw"]["eer"]["hter"]["share_point_argmax"] == 1.0 else "XXX")
    # exceptions to FRR > FAR at tau_EER
    for t, exc in (("itw", {"rawnet2"}), ("df21_verified", {"aasist"})):
        got = {m for m in M5 if pol(m, t)["frr_pct"] <= pol(m, t)["far_pct"]}
        if got != exc:
            fails.append((f"a.exceptions.{t}", f"expected {exc}, data say {got}"))
        n += 1
    # ---- IV-B
    chk("b.frr1", f"rejects {rng([pol(m, it, 'frr1')['frr_pct'] for m in M5], 1)}% of genuine ITW speech and {rng([pol(m, 'df21_verified', 'frr1')['frr_pct'] for m in M5], 1)}% on DF21")
    chk("b.frr5", f"at 5% it rejects {rng([pol(m, it, 'frr5')['frr_pct'] for m in M5], 1)}% on ITW")
    chk("b.far1", f"a dev FAR of 1% becomes {rng([pol(m, it, 'far1')['far_pct'] for m in M5], 1)}% on ITW")
    chk("b.far5", f"a dev FAR of 5% becomes {rng([pol(m, it, 'far5')['far_pct'] for m in M5], 1)}%")
    far1_gt = {m for m in M5 if pol(m, it, "far1")["far_pct"] > pol(m, it, "far1")["frr_pct"]}
    far5_gt = {m for m in M5 if pol(m, it, "far5")["far_pct"] > pol(m, it, "far5")["frr_pct"]}
    n += 2
    if far1_gt != {"aasist", "aasist_l", "rawnet2"}:
        fails.append(("b.far1_dir", str(far1_gt)))
    if far5_gt != set(M5):
        fails.append(("b.far5_dir", str(far5_gt)))
    chk("b.oracle", f"the primary models still reject {rng([tgt(m, it)['target_oracle_frr_at_far']['0.05']['frr_pct'] for m in RK], 0)}% of genuine clips, "
                    f"and the reference {f(tgt(x, it)['target_oracle_frr_at_far']['0.05']['frr_pct'], 0)}%")
    # ---- IV-C
    h2 = A["a1_h2"]["per_model"]
    chk("c.h2_res", f"Only Res-TSSDNet is rejected ({sgn(h2['tssdnet']['delta_pp'])}pp [{sgn(h2['tssdnet']['ci95_pp'][0])}, {sgn(h2['tssdnet']['ci95_pp'][1])}])")
    chk("c.h2_aasist", f"AASIST ({sgn(h2['aasist']['delta_pp'])} [{sgn(h2['aasist']['ci95_pp'][0])}, {sgn(h2['aasist']['ci95_pp'][1])}], p={h2['aasist']['p_one_sided']:.2f}".replace("p=0.", "p=."))
    chk("c.h2_aasist_l", f"AASIST-L ({sgn(h2['aasist_l']['delta_pp'])} [{sgn(h2['aasist_l']['ci95_pp'][0])}, {sgn(h2['aasist_l']['ci95_pp'][1])}], p={h2['aasist_l']['p_one_sided']:.3f}".replace("p=0.", "p=."))
    chk("c.h2_loo", f"between {sgn(h2['aasist']['loo_min_pp'])} and {sgn(h2['aasist']['loo_max_pp'])}pp")
    chk("c.auc", f"(AUC {f(EX['auc_itw']['rawnet2'], 2)})")
    q = A["a4_speakers"]["aasist"]["speaker_frr_quartiles"]
    gts = [A["a4_speakers"][m]["speakers_frr_gt_50"] for m in M5]
    chk("c.quart", f"(quartiles {100 * q[0]:.0f}, {100 * q[1]:.0f} and {100 * q[2]:.0f}% for AASIST) and exceeds 50% for {min(gts)}-{max(gts)} of {A['a1_h2']['clusters']} labels")
    h5 = A["a2_h5"]["per_model"]
    chk("c.h5_dfrr", f"{sgn(h5['aasist']['dfrr_pp'])}, {sgn(h5['aasist_l']['dfrr_pp'])}, {sgn(h5['rawnet2']['dfrr_pp'])} and {sgn(h5['tssdnet']['dfrr_pp'])}pp")
    chk("c.h5_dfar", f"{sgn(h5['aasist']['dfar_pp'])}, {sgn(h5['aasist_l']['dfar_pp'])}, {sgn(h5['rawnet2']['dfar_pp'])} and {sgn(h5['tssdnet']['dfar_pp'])}pp")
    chk("c.h5_68", f"Res-TSSDNet rejects {100 * H5['per_model']['tssdnet']['frr_encodec']:.1f}% of resynthesised bona fide")
    chk("c.h5_6of500", f"RawNet2's rise is {h5['rawnet2']['dfrr_bona_clips_changed']} of 500")
    n += 1
    if not all(h5[m]["holm_dfrr"] and h5[m]["holm_dfar"] and h5[m]["holm_stat"] for m in RK):
        fails.append(("c.h5_holm", "not all components significant"))
    n += 1
    if sum(1 for m in RK if h2[m]["holm_reject_h0"]) != 1:
        fails.append(("c.h2_count", "cluster H2 is not 1 of 4"))
    # ---- IV-D
    iv_x = D[x]["eer_equivalent_interval"]
    chk("d.interval", f"every threshold in [{f(iv_x['tau_lo'], 2)},+{f(iv_x['tau_hi'], 2)}]".replace("[-", "[-"))
    chk("d.frr_range", f"runs from {f(iv['tau_lo']['frr_pct'])}% to {f(iv['tau_hi']['frr_pct'])}% over that interval ({f(pol(x, it)['frr_pct'])}% at the registered")
    bx = B["dev_bootstrap"][x]["targets"]["itw"]["eer"]["frr_ci95"]
    chk("d.devboot", f"[{100 * bx[0]:.1f}, {100 * bx[1]:.1f}]% for that FRR")
    spans = [tgt(m, it)["tau_hi"]["frr_pct"] - tgt(m, it)["tau_lo"]["frr_pct"] for m in RK]
    chk("d.span", f"moves ITW FRR by {min(spans):.0f} to {max(spans):.0f}pp")
    n += 1
    a_lo, a_hi = tgt("aasist", it)["tau_lo"], tgt("aasist", it)["tau_hi"]
    if not ((a_lo["frr_pct"] < a_lo["far_pct"]) and (a_hi["frr_pct"] > a_hi["far_pct"])):
        fails.append(("d.sign_flip", "AASIST sign does not flip"))
    chk("d.evalthr", f"gives the reference {f(tgt(x, it)['tau_eval']['frr_pct'])}% FRR on ITW")
    br = [tgt(m, it)["band_rank"]["pooled"] for m in RK]
    mr = [pol(m, it)["miss_rank"]["pooled"] for m in RK]
    chk("d.band", f"the reference's 1-pp band holds {100 * tgt(x, it)['band_rank']['pooled']:.1f}% of ITW clips against {rng(br, 0, 100)}% for the other models")
    chk("d.miss", f"decisions of {100 * pol(x, it)['miss_rank']['pooled']:.1f}% of clips relative to the oracle threshold ({rng(mr, 0, 100)}% for the others)")
    # ---- IV-E
    dist = FS["distributions"]
    chk("e.lead", f"a median {f(dist['asvspoof19_dev|bona']['lead_median_s'], 2)}s (dev) and {f(dist['asvspoof19_eval|bona']['lead_median_s'], 2)}s (eval) of silence (energy VAD), "
                  f"against {f(dist['asvspoof19_dev|spoof']['lead_median_s'], 2)} and {f(dist['asvspoof19_eval|spoof']['lead_median_s'], 2)}s for spoofs")
    chk("e.itw_lead", f"ITW genuine clips begin with {f(dist['itw|bona']['lead_median_s'], 2)}s (half under 0.1s) and ITW spoofs with {f(dist['itw|spoof']['lead_median_s'], 2)}s")
    n += 1
    if not 0.49 < dist["itw|bona"]["lead_lt_0.1s"] < 0.51:
        fails.append(("e.half", str(dist["itw|bona"]["lead_lt_0.1s"])))
    chk("e.silence_eer", f"{f(FS['silence_only_eer']['asvspoof19_eval']['35'])}% EER")
    chk("e.silence_eer_itw", f"ITW with {f(FS['silence_only_eer']['itw']['35'])}%")
    chk("e.trim_eer", f"from {', '.join(f(FT[m]['19la_eval']['eer_untrim'], 2) for m in M5[:-1])} and {f(FT[x]['19la_eval']['eer_untrim'], 2)}% to "
                      f"{', '.join(f(FT[m]['19la_eval']['eer_trim']) for m in M5[:-1])} and {f(FT[x]['19la_eval']['eer_trim'])}%")
    qx = FS["by_quintile"]["itw"][x]["frr_by_quintile"]
    chk("e.xlsr_frr_bins", f"{100 * qx[0]:.0f}% to {100 * qx[-1]:.0f}% for the reference")
    n += 1
    dirs = {m: FS["by_quintile"]["itw"][m]["frr_by_quintile"][0] > FS["by_quintile"]["itw"][m]["frr_by_quintile"][-1] for m in M5}
    if sum(dirs.values()) != 4 or dirs["tssdnet"]:
        fails.append(("e.dirs", str(dirs)))
    dfrr = [100 * (FT[m]["itw"]["reg_untrim"]["frr_fixed"] - FT[m]["itw"]["tautrim_trim"]["frr_fixed"]) for m in M5]
    dfar = [100 * (FT[m]["itw"]["tautrim_trim"]["far_fixed"] - FT[m]["itw"]["reg_untrim"]["far_fixed"]) for m in M5]
    deer = [FT[m]["itw"]["eer_trim"] - FT[m]["itw"]["eer_untrim"] for m in M5]
    chk("e.trim_frr", f"FRR falls for all five models (by {min(dfrr):.0f}-{max(dfrr):.0f}pp) but FAR rises (by {min(dfar):.0f}-{max(dfar):.0f}pp), and ITW EER moves by only "
                      f"{sgn(min(deer))} to {sgn(max(deer))}pp")
    hu, ht = (100 * FT[x]["itw"]["reg_untrim"]["hter_fixed"], 100 * FT[x]["itw"]["tautrim_trim"]["hter_fixed"])
    chk("e.xlsr_hter", f"({f(hu)} to {f(ht)}%, TTG {f(100 * FT[x]['itw']['reg_untrim']['ttg'], 2)} to {f(100 * FT[x]['itw']['tautrim_trim']['ttg'], 2)}pp)")
    gain = {m: 100 * (FT[m]["itw"]["reg_untrim"]["hter_fixed"] - FT[m]["itw"]["tautrim_trim"]["hter_fixed"]) for m in M5}
    chk("e.small_gain", f"(by {f(gain['aasist_l'])} and {f(gain['tssdnet'])}pp)")
    n += 1
    if not (gain["aasist"] < 0 and gain["rawnet2"] < 0):
        fails.append(("e.worse", str(gain)))
    # ---- IV-F
    res = C5["results"]
    errs = [res[m][k]["est_error_pp_med_p5_p95"][0] for m in M5 for k in ("50", "100", "500")]
    chk("f.bias", f"(error {sgn(min(errs))} to {sgn(max(errs))}pp)")
    spread = [max(abs(res[m][k]["est_error_pp_med_p5_p95"][1]), abs(res[m][k]["est_error_pp_med_p5_p95"][2])) for m in M5 for k in ("50", "100", "500")]
    lo_s = [min(abs(res[m][k]["est_error_pp_med_p5_p95"][1]), abs(res[m][k]["est_error_pp_med_p5_p95"][2])) for m in M5 for k in ("50", "100", "500")]
    chk("f.spread", f"spreads by 18-26pp".replace("18-26", f"{min(lo_s):.0f}-{max(spread):.0f}"))
    w5 = [100 * res[m][k]["share_est_within_5pp"] for m in M5 for k in ("50", "100", "500")]
    chk("f.within5", f"only {min(w5):.0f}-{max(w5):.0f}% of estimates fall within 5pp")
    hn = [res[m][k]["held_frr_new"][0] for m in M5 for k in ("50", "100", "500")]
    fp = [res[m][k]["held_far_new"][0] for m in RK for k in ("50", "100", "500")]
    fx = [res[x][k]["held_far_new"][0] for k in ("50", "100", "500")]
    chk("f.reset", f"(median {min(hn):.1f}-{max(hn):.1f}%) but accepts {min(fp):.0f}-{max(fp):.1f}% of held-out spoofs for the primary models and {min(fx):.0f}-{max(fx):.0f}% for the reference")
    w = E["windows"]

    def win(m, t="itw", p="eer"):
        v = w[f"{m}|{t}"][p]
        return "none" if not v else f"({v[0]:.2f}, {v[1]:.2f})"
    chk("f.windows", f"{win('aasist')} for AASIST, {win('aasist_l')} for AASIST-L, {win('tssdnet')} for Res-TSSDNet, {win(x)} for the reference, and none for RawNet2")
    n += 1
    if win("rawnet2") != "none":
        fails.append(("f.rawnet2_window", win("rawnet2")))
    need = [100 * (1 - pol(m, it)["far_pct"] / 100) / 19 for m in M5]
    got = [pol(m, it)["frr_pct"] for m in M5]
    chk("f.screening_need", f"needs an FRR below {min(need):.1f}-{max(need):.1f}% on ITW against the {min(got):.1f}-{max(got):.1f}% obtained".replace("FRR below", "FRR below"))
    orc = [w[f"{m}|itw"]["oracle"]["c_screening"] for m in RK]
    chk("f.oracle_c", f"costs {min(orc):.2f}-{max(orc):.2f} for the primary models ({w[f'{x}|itw']['oracle']['c_screening']:.2f} for the reference)")
    sh = [st[t][p][k]["share_point_argmax"] for t in ("itw", "df21_verified") for p in ("eer", "frr1", "frr5") for k in ("hter", "screening", "forensic")]
    n += 2
    if sum(1 for v in sh if v >= 0.81) != 17 or len(sh) != 18:
        fails.append(("f.stab_count", f"{sum(1 for v in sh if v >= 0.81)} of {len(sh)}"))
    scr = st["itw"]["eer"]["screening"]["freq"]
    chk("f.stab_exc", f"(AASIST-L {100 * scr['aasist_l']:.0f}%, Res-TSSDNet {100 * scr['tssdnet']:.0f}%, reference {100 * scr['xlsr_aasist']:.0f}%)")
    # ---- discussion and conclusion
    chk("g.rawnet2_far", f"RawNet2 on ITW (FAR {f(pol('rawnet2', it)['far_pct'], 0)}%)")
    tr = EX["triage"]
    chk("g.triage", f"it would send {100 * tr[x]['itw']['genuine']:.0f}% of ITW genuine clips (and {100 * tr[x]['itw']['spoof']:.0f}% of spoofs) to review for the reference and "
                    f"{min(100 * tr[m]['itw']['genuine'] for m in RK):.0f}-{max(100 * tr[m]['itw']['genuine'] for m in RK):.0f}% of genuine clips for the primary models")
    chk("g.concl", f"(FRR {f(iv['tau_lo']['frr_pct'], 0)} to {f(iv['tau_hi']['frr_pct'], 0)}%)")
    print(f"{n} claims checked, {len(fails)} failed")
    for name, text in fails:
        print(f"  FAIL {name}: {text}")
    return len(fails)


if __name__ == "__main__":
    R.gate(verbose=False)
    sys.exit(1 if main() else 0)
