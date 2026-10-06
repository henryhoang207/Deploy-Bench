"""Figures for the revised paper and its supplement (DEV-8). Every plotted number comes from the committed fp32 scores through
revlib (gate-checked) or from revision/results/*.json; nothing is typed in by hand.

    python figures_rev.py      # -> revision/paper/figures/*.pdf

main text:  fig_frr_far.pdf (two-column: (a) ITW trade-off curves, five policies, dev-equivalent interval; (b) H5)
supplement: figS_sensitivity.pdf, figS_speakers.pdf, figS_cost.pdf, figS_silence.pdf, figS_overview.pdf
Palette: Okabe-Ito, with the two oranges of the first version replaced (Res-TSSDNet is now reddish purple) and every model also
distinguished by line style / marker, so the figures survive greyscale printing and colour-vision deficiency.
"""
from __future__ import annotations

import json

import matplotlib
import matplotlib.ticker  # noqa: F401

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import revlib as R  # noqa: E402

COL_W, PAGE_W = 3.45, 7.1
OUT = R.REV / "paper" / "figures"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "lines.linewidth": 1.0, "axes.linewidth": 0.6, "grid.linewidth": 0.4,
})
NAMES = {"aasist": "AASIST", "aasist_l": "AASIST-L", "rawnet2": "RawNet2", "tssdnet": "Res-TSSDNet", "xlsr_aasist": "XLS-R+AASIST (ref.)"}
COL = {"aasist": "#0072B2", "aasist_l": "#56B4E9", "rawnet2": "#009E73", "tssdnet": "#CC79A7", "xlsr_aasist": "#D55E00"}
LS = {"aasist": "-", "aasist_l": "--", "rawnet2": "-.", "tssdnet": ":", "xlsr_aasist": "-"}
LW = {"aasist": 1.0, "aasist_l": 1.0, "rawnet2": 1.0, "tssdnet": 1.3, "xlsr_aasist": 1.4}
POLMARK = {"far5": "D", "far1": "d", "eer": "o", "frr1": "s", "frr5": "^"}
POLNAME = {"far5": "FAR 5%", "far1": "FAR 1%", "eer": "EER", "frr1": "FRR 1%", "frr5": "FRR 5%"}
PANELS = (("19la_eval", "(a) LA19 eval: seen corpus, unseen attacks"), ("itw", "(b) In-the-Wild"), ("df21_verified", "(c) DF21, verified surface"))


def fig_sensitivity():
    """Supplement figure S1: HTER against the threshold in SD units (the original paper's Fig. 1), legend and labels inside the canvas."""
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, 2.15), sharey=True)
    fig.subplots_adjust(wspace=0.07, left=0.06, right=0.995, top=0.80, bottom=0.2)
    for ax, (tgt, title) in zip(axes, PANELS):
        for m in R.MODELS:
            ids, y, s = R.load(m, tgt)
            tau = R.frozen_taus(m)[0]["eer"]
            taus, frr, far = R.det_curve(s, y)
            h = 0.5 * (frr + far)
            sd = float(np.std(s))
            t_or = float(taus[int(np.argmin(h))])
            x = (taus - t_or) / sd
            keep = np.abs(x) <= 3
            ax.plot(x[keep], 100 * h[keep], LS[m], color=COL[m], lw=LW[m], label=NAMES[m])
            ax.plot((tau - t_or) / sd, 100 * R.hter_at(tau, s, y), "v", color=COL[m], ms=4.5, mec="k", mew=0.4, zorder=5)
        ax.axvline(0, color="0.3", lw=0.5)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-1.5, 55)
        ax.set_title(title, pad=3)
        ax.set_xlabel(r"$(\tau-\tau_{\mathrm{oracle}})\,/\,\mathrm{SD}$ of target scores", labelpad=1)
    axes[0].set_ylabel("HTER (%)", labelpad=1)
    h, lab = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], ls="none", marker="v", color="0.5", mec="k", mew=0.4, ms=4.5))
    lab.append(r"frozen $\tau^{*}_{\mathrm{EER}}$")
    fig.legend(h, lab, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=1.1)
    fig.savefig(OUT / "figS_sensitivity.pdf")
    plt.close(fig)


def fig_frr_far():
    """Main-text Fig. 1 (one column): ITW, the full FRR-FAR trade-off of each model, the five frozen thresholds on it and the
    EER-equivalent dev interval (B3) as a thick segment. Everything inside the canvas."""
    b3 = json.loads((R.RESULTS / "b_thresholds.json").read_text())["deterministic"]
    fig, a = plt.subplots(figsize=(COL_W, 2.62))
    fig.subplots_adjust(left=0.135, right=0.985, top=0.965, bottom=0.30)
    a.plot([0, 100], [0, 100], color="0.6", lw=0.6, ls=":")
    xs = np.array([0, 100.0])
    a.plot(xs, 100 * (1 - xs / 100) / 19, color="0.25", lw=0.7, ls=(0, (4, 2)))          # screening break-even FRR = (1 - FAR) / 19
    a.text(99, 6.2, r"screening $C{<}1$ below", fontsize=5.4, ha="right", va="bottom", color="0.25")
    for m in R.MODELS:
        ids, y, s = R.load(m, "itw")
        taus = R.all_taus(m)
        t, frr, far = R.det_curve(s, y)
        a.plot(100 * far, 100 * frr, "-", color=COL[m], lw=0.7, alpha=0.85, zorder=2, label=NAMES[m])
        iv = b3[m]["eer_equivalent_interval"]
        sel = (t >= iv["tau_lo"] - 1e-12) & (t <= iv["tau_hi"] + 1e-12)
        a.plot(100 * far[sel], 100 * frr[sel], "-", color=COL[m], lw=3.2, alpha=0.45, solid_capstyle="butt", zorder=3)
        for p in ("far5", "far1", "eer", "frr1", "frr5"):
            a.plot(100 * R.far_at(taus[p], s, y), 100 * R.frr_at(taus[p], s, y), POLMARK[p], color=COL[m],
                   ms=3.6 if p != "eer" else 4.4, mec="k", mew=0.4, zorder=5)
    a.set_xlim(-2, 100)
    a.set_ylim(-2, 100)
    a.set_xlabel("FAR (%)", labelpad=1)
    a.set_ylabel("FRR (%)", labelpad=1)
    a.text(5, 97, "FRR > FAR", fontsize=5.6, va="top", color="0.25")
    a.text(97, 52, "FAR > FRR", fontsize=5.6, ha="right", color="0.25")
    hm = [plt.Line2D([], [], ls="-", marker="s", color=COL[m], ms=3.2, lw=1.0) for m in R.MODELS]
    lm = [NAMES[m] for m in R.MODELS]
    a.legend(hm, lm, loc="upper right", frameon=False, fontsize=5.8, handlelength=1.6, handletextpad=0.35, labelspacing=0.25,
             bbox_to_anchor=(1.0, 0.99))
    ht = [plt.Line2D([], [], ls="none", marker=POLMARK[p], mfc="white", mec="k", mew=0.5, ms=3.8) for p in ("far5", "far1", "eer", "frr1", "frr5")]
    ht += [plt.Line2D([], [], ls="-", color="0.4", lw=3.2, alpha=0.45, solid_capstyle="butt")]
    lt = [POLNAME[p] for p in ("far5", "far1", "eer", "frr1", "frr5")] + ["dev-equivalent interval of EER threshold"]
    fig.legend(ht, lt, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=5.8,
               handletextpad=0.3, columnspacing=0.9, handlelength=1.1, labelspacing=0.25)
    fig.savefig(OUT / "fig_frr_far.pdf")
    plt.close(fig)


def figS_h5():
    """Supplement: the H5 arrows (clean -> EnCodec 6 kbps at the frozen tau_EER)."""
    h5 = json.loads((R.COMMITTED / "h5_test.json").read_text())
    fig, b = plt.subplots(figsize=(COL_W, 2.25))
    fig.subplots_adjust(left=0.14, right=0.985, top=0.96, bottom=0.30)
    for m in R.RANKED:
        r = h5["per_model"][m]
        x0, y0, x1, y1 = 100 * r["far_clean"], 100 * r["frr_clean"], 100 * r["far_encodec"], 100 * r["frr_encodec"]
        b.annotate("", xy=(x1, y1), xytext=(x0, y0),
                   arrowprops={"arrowstyle": "-|>", "color": COL[m], "lw": 1.0, "mutation_scale": 7, "shrinkA": 0, "shrinkB": 0})
        b.plot(x0, y0, "o", color=COL[m], ms=3.6, mec="k", mew=0.4)
        b.plot(x1, y1, "o", color=COL[m], ms=3.6, mec="k", mew=0.4, mfc="white")
    b.set_xlim(0, 16)
    b.set_ylim(-2, 72)
    b.set_xlabel("FAR (%)", labelpad=1)
    b.set_ylabel("FRR (%)", labelpad=1)
    hm = [plt.Line2D([], [], ls="-", marker="s", color=COL[m], ms=3.4, lw=1.0) for m in R.RANKED]
    hm += [plt.Line2D([], [], ls="none", marker="o", color="0.55", mec="k", mew=0.4, ms=3.4),
           plt.Line2D([], [], ls="none", marker="o", mfc="white", mec="k", mew=0.4, ms=3.4)]
    lm = [NAMES[m] for m in R.RANKED] + ["clean", "EnCodec 6 kbps"]
    fig.legend(hm, lm, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=6.0,
               handletextpad=0.3, columnspacing=0.9, handlelength=1.4, labelspacing=0.25)
    fig.savefig(OUT / "figS_h5.pdf")
    plt.close(fig)


def figS_overview():
    cells = {(c["model"], c["target"]): c for c in json.loads((R.COMMITTED / "operating_points.json").read_text())["cells"] if c["policy"] == "eer"}
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    ax.plot([0, 75], [0, 75], color="0.5", lw=0.6, ls=":")
    mk = {"19la_eval": "o", "itw": "^", "df21_verified": "s"}
    for m in R.MODELS:
        for t, k in mk.items():
            c = cells[(m, t)]
            ax.plot(100 * c["far"], 100 * c["frr"], k, color=COL[m], ms=4.6, mec="k", mew=0.35)
    ax.set_xlim(0, 75)
    ax.set_ylim(0, 75)
    ax.set_xlabel("FAR (%)")
    ax.set_ylabel("FRR (%)")
    hm = [plt.Line2D([], [], ls="none", marker="s", color=COL[m], ms=4) for m in R.MODELS]
    ht = [plt.Line2D([], [], ls="none", marker=k, mfc="white", mec="k", mew=0.5, ms=4) for k in mk.values()]
    l1 = ax.legend(hm, [NAMES[m].replace(" (ref.)", "") for m in R.MODELS], loc="upper right", frameon=False, fontsize=5.8, handletextpad=0.2)
    ax.add_artist(l1)
    ax.legend(ht, ["LA19 eval", "In-the-Wild", "DF21 verified"], loc="lower right", frameon=False, fontsize=5.8, handletextpad=0.2)
    ax.set_title(r"FRR against FAR at the frozen $\tau^{*}_{\mathrm{EER}}$", pad=3)
    fig.savefig(OUT / "figS_overview.pdf", bbox_inches="tight")
    plt.close(fig)


def figS_speakers():
    df = pd.read_csv(R.RESULTS / "per_speaker_itw.csv")
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    rng = np.random.default_rng(0)
    for i, m in enumerate(R.MODELS):
        d = df[df.model == m]
        x = i + rng.uniform(-0.28, 0.28, len(d))
        ax.scatter(x, 100 * d.frr_at_tau_eer, s=2 + 0.012 * d.n_bona, color=COL[m], alpha=0.65, edgecolors="none")
        q = np.percentile(100 * d.frr_at_tau_eer, [25, 50, 75])
        ax.plot([i - 0.38, i + 0.38], [q[1], q[1]], color="k", lw=1.0)
        ax.plot([i - 0.38, i + 0.38], [q[0], q[0]], color="k", lw=0.5, ls=":")
        ax.plot([i - 0.38, i + 0.38], [q[2], q[2]], color="k", lw=0.5, ls=":")
        ids, y, s = R.load(m, "itw")
        ax.plot(i, 100 * R.frr_at(R.frozen_taus(m)[0]["eer"], s, y), "D", mfc="white", mec="k", ms=4, zorder=5)
    ax.set_xticks(range(5), [NAMES[m].replace(" (ref.)", "").replace("+", "+\n") for m in R.MODELS])
    ax.set_ylabel(r"per-speaker FRR at $\tau^{*}_{\mathrm{EER}}$ (%)")
    ax.set_ylim(-3, 103)
    ax.set_title("ITW: one dot per speaker label (area $\\propto$ genuine clips); line = median, dotted = quartiles, $\\diamond$ = pooled", fontsize=6.2, pad=3)
    fig.savefig(OUT / "figS_speakers.pdf", bbox_inches="tight")
    plt.close(fig)


def figS_cost():
    e = json.loads((R.RESULTS / "e_cost.json").read_text())
    P = np.array(e["p_grid"])
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.0), sharey=True)
    for ax, (tgt, title) in zip(axes, (("itw", "In-the-Wild"), ("df21_verified", "DF21, verified surface"))):
        for m in R.MODELS:
            cur = e["curves"][f"{m}|{tgt}"]
            ax.plot(P, cur["eer"], LS[m], color=COL[m], lw=LW[m], label=NAMES[m])
            ax.plot(P, cur["oracle"], ls=(0, (1, 1)), color=COL[m], lw=0.7)
        ax.axhline(1, color="k", lw=0.6)
        for p, lab in ((1 / 11, "forensic"), (0.95, "screening")):
            ax.axvline(p, color="0.5", lw=0.6, ls="--")
            ax.text(p, 0.012, lab, rotation=90, fontsize=5.6, ha="right", va="bottom", color="0.3")
        ax.set_xscale("logit")
        ax.set_yscale("log")
        ax.set_xticks([0.01, 0.1, 0.5, 0.9, 0.99])
        ax.set_xticklabels(["0.01", "0.1", "0.5", "0.9", "0.99"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlim(0.01, 0.99)
        ax.set_ylim(0.01, 60)
        ax.set_xlabel("effective prior of bona fide  $p$  (logit scale)")
        ax.set_title(title, pad=3)
    axes[0].set_ylabel("normalised cost $C(p)$")
    h, lab = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], ls=(0, (1, 1)), color="0.3", lw=0.7))
    lab.append("target oracle (dotted)")
    fig.legend(h, lab, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.13), handlelength=2.2, columnspacing=1.1, fontsize=6.4)
    fig.savefig(OUT / "figS_cost.pdf", bbox_inches="tight")
    plt.close(fig)


def figS_silence():
    st = pd.read_csv(R.RESULTS / "vad_stats.csv", dtype={"utt_id": str})
    fig, (a, b) = plt.subplots(1, 2, figsize=(PAGE_W, 2.0))
    for (ds, lab, lbl), c, ls in ((("asvspoof19_dev", 1, "LA19 dev bona fide"), "#0072B2", "-"), (("asvspoof19_dev", 0, "LA19 dev spoof"), "#0072B2", "--"),
                                  (("itw", 1, "ITW genuine"), "#D55E00", "-"), (("itw", 0, "ITW spoof"), "#D55E00", "--")):
        d = np.sort(st[(st.dataset == ds) & (st.label == lab)].lead35.to_numpy() / R_SR)
        a.plot(d, np.arange(1, len(d) + 1) / len(d), ls, color=c, lw=1.1, label=lbl)
    a.set_xlim(0, 2.0)
    a.set_xlabel("leading silence (s, energy VAD, $T=35$ dB)")
    a.set_ylabel("cumulative share of clips")
    a.legend(frameon=False, fontsize=6, loc="lower right")
    a.set_title("(a) Leading silence by set and class", pad=3)
    sub = st[(st.dataset == "itw") & (st.label == 1)].set_index("utt_id")
    edges = np.array([0.005, 0.2, 0.3, 0.4])
    for m in R.MODELS:
        ids, y, s = R.load(m, "itw")
        tau = R.frozen_taus(m)[0]["eer"]
        ser = pd.Series(s, index=ids)[y == 1]
        lead = sub.reindex(ser.index).lead35.to_numpy() / R_SR
        qi = np.digitize(lead, edges)
        frr = [100 * np.mean(ser.to_numpy()[qi == k] < tau) for k in range(5)]
        b.plot(range(5), frr, LS[m], color=COL[m], lw=LW[m], marker="o", ms=2.8, label=NAMES[m].replace(" (ref.)", ""))
    b.set_xticks(range(5), ["0", "<0.2", "0.2-0.3", "0.3-0.4", ">=0.4"])
    b.set_xlabel("leading silence of the genuine clip (s)")
    b.set_ylabel(r"FRR at $\tau^{*}_{\mathrm{EER}}$ (%)")
    b.set_ylim(0, 96)
    b.legend(frameon=False, fontsize=5.6, ncol=3, loc="upper center", columnspacing=0.8, handlelength=1.8)
    b.set_title("(b) ITW genuine clips: FRR by leading silence", pad=3)
    fig.tight_layout()
    fig.savefig(OUT / "figS_silence.pdf", bbox_inches="tight")
    plt.close(fig)


R_SR = 16000.0

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    R.gate(verbose=False)
    for f in (fig_sensitivity, fig_frr_far, figS_h5, figS_overview, figS_speakers, figS_cost, figS_silence):
        f()
        print("wrote", f.__name__, flush=True)
