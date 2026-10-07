"""Claim-by-claim number audit of the T-BIOM manuscript (paper/tbiom/main.tex), W5.

Round 1: each number is BUILT from the ICME revision's audited result files (revlib and DFV_Conference/revision/results,
the method of that revision's audit_claims.py) and must appear in the LaTeX source after normalisation (comments, math
delimiters, thin spaces and all whitespace removed). The round-1 generated tables must be byte-identical to the
revision's. Round 2: built from results/round2 once the final analysis is collected; until then its claims are listed as
pending. Numbers quoted from other papers or from project documents are in CITED and are not computed.

Run from DFV_Conference/revision/analysis, so that revlib imports and its reproduction gate runs first:
    python C:/dev/deploybench/paper/tbiom/audit_claims.py [--final]
--final: pending round-2 claims, any \\todo and more than 10 pages are failures.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
if not (Path.cwd() / "revlib.py").exists():
    sys.exit("REFUSED: run from DFV_Conference/revision/analysis (revlib.py not found in the working directory)")
import revlib as R  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = R.RESULTS
M5, RK = list(R.MODELS), list(R.RANKED)
B = json.loads((RES / "b_thresholds.json").read_text(encoding="utf-8"))
A = json.loads((RES / "a_cluster.json").read_text(encoding="utf-8"))
E = json.loads((RES / "e_cost.json").read_text(encoding="utf-8"))
C5 = json.loads((RES / "a5_cheapcheck.json").read_text(encoding="utf-8"))
G = json.loads((RES / "g_docs.json").read_text(encoding="utf-8"))
FS = json.loads((RES / "f_vad_stats_analysis.json").read_text(encoding="utf-8"))
FT = json.loads((RES / "f_vad_scores_analysis.json").read_text(encoding="utf-8"))
H2 = json.loads((R.COMMITTED / "h2_test.json").read_text(encoding="utf-8"))
H5 = json.loads((R.COMMITTED / "h5_test.json").read_text(encoding="utf-8"))
D = B["deterministic"]
R2A = R.CLONE / "results" / "round2" / "analysis"

# quoted, not computed: (claim text, source)
CITED = [("0.21% source EER that rejects 78.7%", "Zhou and Wang [zhou26_eer_hides]"),
         ("from 3.6% to 15.5%", "Mueller et al. [muller21_silence]"),
         ("HTER reaches 50%", "Korshunov and Marcel [korshunov16_crossdb]"),
         ("per-trial agreement 3.7x10^-5", "round-1 RawNet2 DF21 gate (phase_1_reproduction)"),
         ("11 of 13 attacks unseen", "LA19 attack list (A16 = A04, A19 = A06)"),
         ("registered on 19 July 2026 ... frozen on 28 September 2026", "PREREGISTRATION.md history, tag roster-freeze"),
         ("B = 1,000 / B = 10,000", "PREREGISTRATION.md, DEVIATIONS.md")]


def tgt(m, t):
    return D[m]["targets"][t]


def pol(m, t, p="eer"):
    return D[m]["targets"][t]["policies"][p]


def rng(vals, nd, scale=1.0):
    return f"{min(vals) * scale:.{nd}f}-{max(vals) * scale:.{nd}f}"


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


def main(final: bool):
    tex = (HERE / "main.tex").read_text(encoding="utf-8")
    body = norm_tex(tex.split(r"\begin{document}", 1)[1])
    fails, pending, n = [], [], 0

    def chk(name, text):
        nonlocal n
        n += 1
        if norm_claim(text) not in body:
            fails.append((name, text))

    def truth(name, ok, detail=""):
        nonlocal n
        n += 1
        if not ok:
            fails.append((name, detail or "false"))

    # ---- generated round-1 tables: byte-identical to the audited revision's
    for g in sorted((HERE / "gen").glob("*.tex")):
        if g.name.startswith("r2_"):
            continue
        src = R.REV / "paper" / "gen" / g.name
        truth(f"gen.{g.stem}", src.exists() and src.read_bytes() == g.read_bytes(), f"{g.name} differs from {src}")

    # ---- study design
    dv = D["aasist"]["dev"]
    chk("design.dev_n", f"({dv['n_bona'] + dv['n_spoof']:,} trials, {dv['n_bona']:,} bona fide)")
    _, y, _ = R.load("aasist", "19la_eval")
    chk("design.la19", f"({len(y):,} trials; 11 of 13 attacks unseen)")
    _, y, _ = R.load("aasist", "itw")
    chk("design.itw", f"({len(y):,} clips of public figures with {A['a1_h2']['clusters']} distinct speaker labels")
    gt = G["total"]
    n_eval, n_a, n_b, n_ver = gt["eval"][0], gt["A"][0], gt["B"][0], gt["verified"][0]
    chk("design.df21", f"({n_ver:,} trials; libFLAC 1.4 rejects {100 * (n_a + n_b) / n_eval:.0f}% of the public evaluation "
                       f"files and for {n_b:,} of them")
    hm = G["h5_manifest"]
    chk("design.h5", f"1,000 LA19-eval clips (500 per class, all {hm['speakers']} speakers)")

    # ---- round 1: unseen attacks
    la, it, df = "19la_eval", "itw", "df21_verified"
    ttg = lambda m, t, p="eer": pol(m, t, p)["ttg_pp"]  # noqa: E731
    cl = lambda m, t, p="eer": A["a3_e2"]["ttg"][t]["cells"][f"{m}|{p}"]["cluster_ci95_pp"]  # noqa: E731
    chk("r1.la_far_frr", f"accept {rng([pol(m, la)['far_pct'] for m in RK], 1)}% of spoofs and reject "
                         f"{rng([pol(m, la)['frr_pct'] for m in RK], 1)}% of genuine clips")
    chk("r1.la_ttg", f"lose {f(ttg('aasist', la), 2)} and {f(ttg('aasist_l', la), 2)} HTER points to transfer (speaker-level "
                     f"95% CIs [{f(cl('aasist', la)[0], 2)}, {f(cl('aasist', la)[1], 2)}] and [{f(cl('aasist_l', la)[0], 2)}, "
                     f"{f(cl('aasist_l', la)[1], 2)}]), {100 * pol('aasist', la)['transfer_share']:.0f}% and "
                     f"{100 * pol('aasist_l', la)['transfer_share']:.0f}% of their deployed HTER")
    chk("r1.la_eer_hter", f"the EER hides ({f(tgt('aasist', la)['eer_pct'], 2)}% against "
                          f"{f(pol('aasist', la)['hter_pct'], 2)}% for AASIST)")
    chk("r1.la_rank", f"Res-TSSDNet ({f(pol('tssdnet', la)['hter_pct'], 2)}%) overtakes both")
    truth("r1.la_rank_order", all(pol("tssdnet", la)["hter_pct"] < pol(m, la)["hter_pct"] and
                                  tgt("tssdnet", la)["eer_pct"] > tgt(m, la)["eer_pct"] for m in ("aasist", "aasist_l")))

    # ---- round 1: shifted targets
    for t in (it, df):
        exc = {m for m in M5 if pol(m, t)["frr_pct"] <= pol(m, t)["far_pct"]}
        truth(f"r1.four_of_five.{t}", len(exc) == 1, f"FRR <= FAR at tau_EER for {exc}")
    chk("r1.itw_eer", f"(EER {rng([tgt(m, it)['eer_pct'] for m in RK], 0)}%; transfer only "
                      f"{rng([pol(m, it)['transfer_share'] for m in RK], 0, 100)}% of the deployed HTER)")
    x = "xlsr_aasist"
    chk("r1.xlsr_share", f"({100 * pol(x, it)['transfer_share']:.0f}% and {100 * pol(x, df)['transfer_share']:.0f}% of its "
                         f"HTER on ITW and DF21)")
    truth("r1.xlsr_largest_gap", all(ttg(x, t) > ttg(m, t) for t in (it, df) for m in RK),
          "the reference's TTG is not the largest on ITW and DF21")
    chk("r1.fewest", f"(HTER {f(pol(x, it)['hter_pct'])}% on ITW against {rng([pol(m, it)['hter_pct'] for m in RK], 1)}%)")
    truth("r1.fewest_order", pol(x, it)["hter_pct"] < min(pol(m, it)["hter_pct"] for m in RK))

    # ---- round 1: registered tests
    h2 = A["a1_h2"]["per_model"]
    truth("r1.h2_clip", H2["n_models_frr_gt_far"] == 3 and H2["h2_supported"], f"clip-level H2: {H2['n_models_frr_gt_far']}")
    chk("r1.h2_clusters", f"resampling ITW's {A['a1_h2']['clusters']} speaker labels")
    chk("r1.h2_res", f"only Res-TSSDNet's FRR - FAR excludes zero ({sgn(h2['tssdnet']['delta_pp'])}pp "
                     f"[{sgn(h2['tssdnet']['ci95_pp'][0])}, {sgn(h2['tssdnet']['ci95_pp'][1])}])")
    truth("r1.h2_count", sum(1 for m in RK if h2[m]["holm_reject_h0"]) == 1, "cluster H2 is not 1 of 4")
    chk("r1.h2_loo", f"moves AASIST's difference between {sgn(h2['aasist']['loo_min_pp'])} and "
                     f"{sgn(h2['aasist']['loo_max_pp'])}pp")
    h5 = A["a2_h5"]["per_model"]
    truth("r1.h5_dirs", all(h5[m]["dfrr_pp"] > 0 and h5[m]["dfar_pp"] < 0 for m in RK), "EnCodec FRR up / FAR down")
    truth("r1.h5_holm", all(h5[m]["holm_dfrr"] and h5[m]["holm_dfar"] and h5[m]["holm_stat"] for m in RK))
    chk("r1.h5_68", f"Res-TSSDNet rejects {100 * H5['per_model']['tssdnet']['frr_encodec']:.1f}% of resynthesised bona fide")

    # ---- round 1: policy, determinacy
    chk("r1.frr1", f"rejects {rng([pol(m, it, 'frr1')['frr_pct'] for m in M5], 1)}% of genuine ITW speech")
    chk("r1.far5", f"a dev FAR of 5% becomes {rng([pol(m, it, 'far5')['far_pct'] for m in M5], 1)}% FAR with FAR above FRR "
                   f"for all five models")
    truth("r1.far5_dir", all(pol(m, it, "far5")["far_pct"] > pol(m, it, "far5")["frr_pct"] for m in M5))
    ivx, iv = D[x]["eer_equivalent_interval"], D[x]["targets"]["itw"]
    chk("r1.interval", f"every threshold in [{sgn(ivx['tau_lo'], 2)}, {sgn(ivx['tau_hi'], 2)}] ties")
    chk("r1.frr_range", f"its ITW FRR runs from {f(iv['tau_lo']['frr_pct'])}% to {f(iv['tau_hi']['frr_pct'])}% over that "
                        f"interval")

    # ---- round 1: silence
    dist = FS["distributions"]
    leads = {k: (dist[f"asvspoof19_{k}|bona"]["lead_median_s"], dist[f"asvspoof19_{k}|spoof"]["lead_median_s"])
             for k in ("dev", "eval")}
    hits = [k for k, (b, s) in leads.items()
            if norm_claim(f"begin with a median {f(b, 2)}s of silence against {f(s, 2)}s for spoofs") in body]
    truth("r1.lead", bool(hits), f"LA19 lead medians dev {leads['dev']} / eval {leads['eval']} not in the text")
    chk("r1.itw_lead", f"ITW genuine clips begin with {f(dist['itw|bona']['lead_median_s'], 2)}s")
    fold = [FT[m]["19la_eval"]["eer_trim"] / FT[m]["19la_eval"]["eer_untrim"] for m in M5]
    chk("r1.trim_fold", f"of all five models {min(fold):.0f}-{max(fold):.0f} fold")
    dfrr = [100 * (FT[m]["itw"]["reg_untrim"]["frr_fixed"] - FT[m]["itw"]["tautrim_trim"]["frr_fixed"]) for m in M5]
    dfar = [100 * (FT[m]["itw"]["tautrim_trim"]["far_fixed"] - FT[m]["itw"]["reg_untrim"]["far_fixed"]) for m in M5]
    deer = [FT[m]["itw"]["eer_trim"] - FT[m]["itw"]["eer_untrim"] for m in M5]
    chk("r1.trim", f"lowers ITW FRR ({min(dfrr):.0f}-{max(dfrr):.0f}pp) but raises FAR ({min(dfar):.0f}-{max(dfar):.0f}pp)")
    chk("r1.trim_eer", f"moves ITW EER by only {sgn(min(deer))} to {sgn(max(deer))}pp")

    # ---- round 1: checks for deployers
    res = C5["results"]
    fp = [res[m][k]["held_far_new"][0] for m in RK for k in ("50", "100", "500")]
    chk("r1.reset", f"accepts {min(fp):.0f}-{max(fp):.1f}% of held-out spoofs for the primary models")
    w = E["windows"]
    truth("r1.windows", not w["rawnet2|itw"]["eer"] and all(w[f"{m}|itw"]["eer"] for m in M5 if m != "rawnet2"),
          "break-even windows: RawNet2 must have none, the others one")

    # ---- round 2 (built from the collected final analysis; pending until then)
    reg = R2A / "r2_registered.json"
    if reg.exists():
        h4 = json.loads(reg.read_text(encoding="utf-8"))["h4"]
        cond = h4["conditions"]
        d4 = lambda c, k: [cond[c]["per_model"][m][k] for m in R.RANKED]  # noqa: E731
        chk("r2.h4_opus6k", f"Opus at 6 kbps raises FRR by {rng(d4('opus6k', 'dfrr_pp'), 0)} pp")
        chk("r2.h4_amr", f"AMR-NB at 4.75 kbps by {rng(d4('amr4.75', 'dfrr_pp'), 0)} pp across the four primary models")
        truth("r2.h4_far_falls", all(v < 0 for c in ("opus6k", "amr4.75") for v in d4(c, "dfar_pp")))
        truth("r2.h4_4of4", all(cond[c]["n_rejected"] == 4 for c in ("opus6k", "amr4.75", "opus24k", "mic_ir")))
        truth("r2.h4_opus24k_small", max(abs(v) for v in d4("opus24k", "dfrr_pp")) <= 1.0)
        chk("r2.h4_ir", f"raises FRR by {rng(d4('mic_ir', 'dfrr_pp'), 0)} pp (four of four)")
        truth("r2.h4_mp3", cond["mp3_32"]["n_rejected"] == 1)
        aa = [cond[c]["per_model"][m]["dfrr_pp"] for c in ("snr10", "snr5") for m in ("aasist", "aasist_l")]
        chk("r2.h4_noise_frr", f"AASIST and AASIST-L reject {rng(aa, 0)} pp more genuine speech at 10 and 5 dB SNR")
        ts = [cond[c]["per_model"]["tssdnet"]["dfar_pp"] for c in ("snr10", "snr5")]
        chk("r2.h4_noise_far", f"Res-TSSDNet's FAR rises by {rng(ts, 0)} pp")
        truth("r2.h4_noise_2of4", all(cond[c]["n_rejected"] == 2 for c in ("snr20", "snr10", "snr5")))
        chk("r2.h4_total", f"R2-H4 holds for {['no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight'][h4['n_conditions_supported']]} of the eight conditions")
    else:
        pending.append("R2-H4 paragraph (Sec. VI-D): results/round2/analysis/r2_registered.json not collected yet")
    for t in re.findall(r"\\todo\{([^}]{0,70})", tex.split(r"\begin{document}", 1)[1]):
        pending.append("text: " + " ".join(t.split()))

    if final:
        todo = len(re.findall(r"\\todo\{", tex.split(r"\begin{document}", 1)[1]))
        truth("final.todo", todo == 0, f"{todo} \\todo left")
        log = HERE / "main.log"
        pages = re.search(r"Output written on main\.pdf \((\d+) pages", log.read_text(errors="replace")) if log.exists() else None
        truth("final.pages", pages is not None and int(pages.group(1)) <= 10, f"pages: {pages.group(1) if pages else '?'}")
        for p in pending:
            fails.append(("final.pending", p))
    print(f"{n} claims checked, {len(fails)} failed; {len(pending)} round-2 items pending; {len(CITED)} cited (not computed)")
    for name, text in fails:
        print(f"  FAIL {name}: {text}")
    for p in pending:
        print(f"  pending: {p}")
    return len(fails)


if __name__ == "__main__":
    R.gate(verbose=False)
    sys.exit(1 if main("--final" in sys.argv[1:]) else 0)
