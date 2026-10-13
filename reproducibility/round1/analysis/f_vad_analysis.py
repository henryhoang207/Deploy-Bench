"""DEV-8 F1 and F3: silence statistics against the per-clip decisions, and the VAD-trimmed re-scoring results.

    python f_vad_analysis.py stats      # F1: needs results/vad_stats.csv (from sol1, f_vad_stats.py)
    python f_vad_analysis.py scores     # F3: needs results/vad/scores_<model>_<dataset>_vad35.csv (from sol1, f_vad_score.py)

Descriptive only, no test. The reading rule for F3 was fixed in DEV-8 before any run.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import revlib as R

SR = 16000
BINS = np.array([0.005, 0.2, 0.3, 0.4])      # ITW: 40% of genuine clips have zero leading silence, so fixed bins, not quantiles
OUT = []


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def stats():
    st = pd.read_csv(R.input_file(R.RESULTS / "vad_stats.csv"), dtype={"utt_id": str})
    say(f"vad_stats.csv: {len(st)} clips (LA19 dev {int((st.dataset == 'asvspoof19_dev').sum())}, LA19 eval "
        f"{int((st.dataset == 'asvspoof19_eval').sum())}, ITW {int((st.dataset == 'itw').sum())})")
    names = {"asvspoof19_dev": "LA19 dev", "asvspoof19_eval": "LA19 eval", "itw": "ITW"}
    res = {"distributions": {}, "silence_only_eer": {}, "by_quintile": {}}
    say("\n== F1a  Leading / trailing silence (seconds, energy VAD, T = 35 dB re the loudest frame): median [P10, P90] and share < 0.1 s")
    say(f"{'set':10s} {'class':7s} {'n':>6s} {'duration':>16s} {'lead':>24s} {'lead<0.1s':>9s} {'trail':>24s}")
    for ds in ("asvspoof19_dev", "asvspoof19_eval", "itw"):
        for lab, cname in ((1, "bona"), (0, "spoof")):
            d = st[(st.dataset == ds) & (st.label == lab)]
            lead, trail, dur = d.lead35 / SR, d.trail35 / SR, d.n_samples / SR

            def q(x):
                return f"{np.median(x):5.2f} [{np.percentile(x, 10):5.2f},{np.percentile(x, 90):5.2f}]"

            res["distributions"][f"{ds}|{cname}"] = {
                "n": int(len(d)), "dur_median_s": float(np.median(dur)), "lead_median_s": float(np.median(lead)),
                "lead_p10_p90_s": [float(np.percentile(lead, 10)), float(np.percentile(lead, 90))],
                "trail_median_s": float(np.median(trail)), "lead_lt_0.1s": float(np.mean(lead < 0.1))}
            say(f"{names[ds]:10s} {cname:7s} {len(d):6d} {q(dur):>16s} {q(lead):>24s} {100 * np.mean(lead < 0.1):8.1f}% {q(trail):>24s}")
    say("\n== F1b  Sensitivity of the leading-silence median (s) to the VAD threshold T (dB below the loudest frame)")
    for ds in ("asvspoof19_dev", "asvspoof19_eval", "itw"):
        row = []
        for lab, cname in ((1, "bona"), (0, "spoof")):
            d = st[(st.dataset == ds) & (st.label == lab)]
            row.append(f"{cname}: " + "  ".join(f"T{T}={np.median(d[f'lead{T}']) / SR:5.2f}" for T in (25, 35, 45)))
        say(f"{names[ds]:10s} " + " | ".join(row))
    say("\n== F1c  Leading silence alone as a bona-fide detector (EER %, longer silence = bona fide); Mueller et al. 2021 report 15.1% on LA19 eval")
    for ds in ("asvspoof19_dev", "asvspoof19_eval", "itw"):
        d = st[st.dataset == ds]
        row = {T: 100 * R.eer(d[f"lead{T}"].to_numpy(float), d.label.to_numpy(int))[0] for T in (25, 35, 45)}
        res["silence_only_eer"][ds] = row
        say(f"{names[ds]:10s} " + "  ".join(f"T{T}: {v:5.1f}%" for T, v in row.items()))
    say("\n== F1d  Does the per-clip decision depend on leading silence? Registered tau_eer; bona fide clips of ITW and LA19 eval")
    for ds, tgt in (("itw", "itw"), ("asvspoof19_eval", "19la_eval")):
        sub = st[(st.dataset == ds) & (st.label == 1)].set_index("utt_id")
        edges = BINS if ds == "itw" else np.percentile(sub.lead35 / SR, [20, 40, 60, 80])
        nb = np.bincount(np.digitize(sub.lead35 / SR, edges), minlength=5)
        say(f"-- {names[ds]} bona fide: bin edges of leading silence (s) = {np.round(edges, 3).tolist()}; clips per bin {nb.tolist()}")
        say(f"{'model':12s} {'Spearman(score, lead)':>22s} | FRR % at tau_eer by leading-silence bin (shortest -> longest) | short minus long")
        res["by_quintile"][tgt] = {}
        for m in R.MODELS:
            ids, y, s = R.load(m, tgt)
            tau = R.frozen_taus(m)[0]["eer"]
            ser = pd.Series(s, index=ids)[y == 1]
            lead = (sub.reindex(ser.index).lead35 / SR).to_numpy()
            rho = float(spearmanr(ser.to_numpy(), lead).statistic)
            qi = np.digitize(lead, edges)
            frr_q = [float(np.mean(ser.to_numpy()[qi == k] < tau)) for k in range(5)]
            res["by_quintile"][tgt][m] = {"spearman": rho, "frr_by_quintile": frr_q}
            say(f"{R.NICE[m]:12s} {rho:+22.3f} | " + "  ".join(f"{100 * f:5.1f}" for f in frr_q) + f" | {100 * (frr_q[0] - frr_q[-1]):+5.1f} pp")
    say("\n== F1e  Same for spoofs: FAR % at tau_eer by leading-silence bin of the spoof clips (shortest -> longest)")
    for ds, tgt in (("itw", "itw"), ("asvspoof19_eval", "19la_eval")):
        sub = st[(st.dataset == ds) & (st.label == 0)].set_index("utt_id")
        edges = BINS if ds == "itw" else np.percentile(sub.lead35 / SR, [20, 40, 60, 80])
        nb = np.bincount(np.digitize(sub.lead35 / SR, edges), minlength=5)
        say(f"-- {names[ds]} spoof: bin edges (s) = {np.round(edges, 3).tolist()}; clips per bin {nb.tolist()}")
        for m in R.MODELS:
            ids, y, s = R.load(m, tgt)
            tau = R.frozen_taus(m)[0]["eer"]
            ser = pd.Series(s, index=ids)[y == 0]
            lead = (sub.reindex(ser.index).lead35 / SR).to_numpy()
            qi = np.digitize(lead, edges)
            far_q = [float(np.mean(ser.to_numpy()[qi == k] >= tau)) for k in range(5)]
            res["by_quintile"][f"{tgt}_spoof_{m}"] = far_q
            say(f"{R.NICE[m]:12s} " + "  ".join(f"{100 * f:5.1f}" for f in far_q))
    R.write_json("f_vad_stats_analysis.json", res)
    (R.RESULTS / "f_vad_stats_analysis.txt").write_text("\n".join(OUT) + "\n", encoding="utf-8")


def scores():
    st = pd.read_csv(R.input_file(R.RESULTS / "vad_stats.csv"), dtype={"utt_id": str}).set_index(["dataset", "utt_id"])
    vdir = R.RESULTS / "vad"
    res = {}
    say("DEV-8 F3: VAD-trimmed (T = 35 dB, both ends) fp32 scores; tau_trim = the five estimators on the TRIMMED dev scores of each model")
    for ds in ("asvspoof19_dev", "asvspoof19_eval", "itw"):
        d = st.loc[ds]
        kept = d.n_samples - d.lead35 - d.trail35
        say(f"   {ds:16s} clips with something trimmed: {100 * np.mean((d.lead35 + d.trail35) > 0):5.1f}%, median kept "
            f"{np.median(kept) / SR:5.2f} s of {np.median(d.n_samples) / SR:5.2f} s; guard (< 0.25 s kept, left untrimmed): "
            f"{int(np.sum((kept < 4000) & ((d.lead35 + d.trail35) > 0)))}")
    names = {"19la_eval": "LA19 eval", "itw": "ITW"}
    ds_of = {"19la_eval": "asvspoof19_eval", "itw": "itw"}

    def read(m, ds):
        p = vdir / f"scores_{m}_{ds}_vad35.csv"
        df = pd.read_csv(R.input_file(p), usecols=["utt_id", "label", "score", "condition", "precision"], dtype={"utt_id": str},
                         float_precision="round_trip")
        if set(df.condition) != {"vad35"} or set(df.precision) != {"fp32"}:
            R.die(f"{p}: unexpected condition/precision")
        return df["utt_id"].to_numpy(), df["label"].to_numpy(np.int64), df["score"].to_numpy(np.float64)

    taus_trim = {}
    say("\n== dev: untrimmed -> trimmed EER (%) and thresholds (computed from the trimmed DEV scores only)")
    for m in R.MODELS:
        ids, y, s = read(m, "asvspoof19_dev")
        if len(y) != 24844:
            R.die(f"{m}: trimmed dev has {len(y)} rows")
        c = R.calibrate_tau(s, y)
        c["far1"] = R.tau_at_far(s, y, 0.01)
        c["far5"] = R.tau_at_far(s, y, 0.05)
        taus_trim[m] = {k: float(v) for k, v in c.items()}
        _, yu, su = R.load(m, "dev")
        tu = R.all_taus(m)
        res[m] = {"dev_eer_untrim_pct": 100 * R.eer(su, yu)[0], "dev_eer_trim_pct": 100 * R.eer(s, y)[0], "taus_trim": taus_trim[m]}
        say(f"{R.NICE[m]:12s} dev EER {res[m]['dev_eer_untrim_pct']:6.3f} -> {res[m]['dev_eer_trim_pct']:6.3f} | tau_eer {tu['eer']:+8.3f} -> "
            f"{taus_trim[m]['eer']:+8.3f} | tau_frr1 {tu['frr1']:+8.3f} -> {taus_trim[m]['frr1']:+8.3f}")
    R.write_json("f_taus_trim.json", taus_trim)           # written from dev scores only, before any trimmed target is opened
    for tgt in ("19la_eval", "itw"):
        say(f"\n== {names[tgt]}: EER %; FRR/FAR % (HTER %, TTG pp) at the registered tau_eer on untrimmed and trimmed audio, and at tau_trim on trimmed audio")
        say(f"{'model':12s} {'EER u->t':>13s} | {'registered, untrimmed':>26s} | {'registered, trimmed':>26s} | {'tau_trim, trimmed':>26s} | {'oracle HTER u->t':>16s}")
        for m in R.MODELS:
            idu, yu, su = R.load(m, tgt)
            idt, yt, stt = read(m, ds_of[tgt])
            if len(yt) != len(yu):
                R.die(f"{m}/{tgt}: trimmed n={len(yt)} != {len(yu)}")
            tau = R.frozen_taus(m)[0]["eer"]
            eu, et = 100 * R.eer(su, yu)[0], 100 * R.eer(stt, yt)[0]
            cu, ct, cq = R.ttg(su, yu, tau), R.ttg(stt, yt, tau), R.ttg(stt, yt, taus_trim[m]["eer"])
            res[m][tgt] = {"eer_untrim": eu, "eer_trim": et, "reg_untrim": cu, "reg_trim": ct, "tautrim_trim": cq}

            def f(c):
                return f"{100 * c['frr_fixed']:5.1f}/{100 * c['far_fixed']:5.1f} ({100 * c['hter_fixed']:4.1f},{100 * c['ttg']:5.2f})"

            say(f"{R.NICE[m]:12s} {eu:5.2f} -> {et:5.2f} | {f(cu):>26s} | {f(ct):>26s} | {f(cq):>26s} | {100 * cu['hter_oracle']:6.2f} -> {100 * ct['hter_oracle']:6.2f}")
    R.write_json("f_vad_scores_analysis.json", res)
    (R.RESULTS / "f_vad_scores_analysis.txt").write_text("\n".join(OUT) + "\n", encoding="utf-8")


if __name__ == "__main__":
    {"stats": stats, "scores": scores}[sys.argv[1]]()
