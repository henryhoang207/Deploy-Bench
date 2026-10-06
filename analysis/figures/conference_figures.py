"""Figures for the conference paper (D49), from committed fp32 scores and committed results.

    python analysis/figures/conference_figures.py   # -> paper/conference/figures/*.pdf + data json

fig_sensitivity.pdf  (two-column) HTER(tau) per target, tau in target-score SD units from the
                     target-oracle threshold; triangles = frozen tau*_eer. The curves come from
                     paper_figures.collect() (the same computation as the TIFS figure).
fig_frr_far.pdf      (one column) (a) FRR vs FAR at tau*_eer, 5 models x 3 targets, from
                     results/analysis/operating_points.json; (b) H5: clean -> EnCodec 6 kbps on the
                     1,000-clip parity manifest, 4 ranked models, from results/analysis/h5_test.json.
Descriptive only. Every plotted point is written to conference_figures_data.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_figures  # noqa: E402  (imports style, which sets the Agg backend)
from style import COL_W, PAGE_W, plt  # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42,          # TrueType, not Type 3 (IEEE PDF eXpress)
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "lines.linewidth": 1.0,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4,
})

OUT = R / "paper" / "conference" / "figures"
MODELS = ("aasist", "aasist_l", "rawnet2", "tssdnet", "xlsr_aasist")
NAMES = {"aasist": "AASIST", "aasist_l": "AASIST-L", "rawnet2": "RawNet2",
         "tssdnet": "Res-TSSDNet", "xlsr_aasist": "XLS-R+AASIST (ref.)"}
# Okabe-Ito, colour-blind safe; the AASIST pair shares a hue
COL = {"aasist": "#0072B2", "aasist_l": "#56B4E9", "rawnet2": "#009E73",
       "tssdnet": "#E69F00", "xlsr_aasist": "#D55E00"}
LS = {"aasist": "-", "aasist_l": "--", "rawnet2": "-", "tssdnet": "-", "xlsr_aasist": "-"}
PANELS = (("19LA eval", "19la_eval", "(a) LA19 eval: seen corpus, unseen attacks"),
          ("ITW", "itw", "(b) In-the-Wild"),
          ("DF21 verified", "df21_verified", "(c) DF21, verified surface"))
TGT_MARK = {"19la_eval": "o", "itw": "^", "df21_verified": "s"}
TGT_NAME = {"19la_eval": "LA19 eval", "itw": "In-the-Wild", "df21_verified": "DF21 verified"}


def fig_sensitivity(data, record):
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, 1.72), sharey=True)
    fig.subplots_adjust(wspace=0.07)
    for ax, (key, short, title) in zip(axes, PANELS):
        for m in MODELS:
            d = data[(m, key)]
            x, h = d["_curve"]
            keep = np.abs(x) <= 3
            ax.plot(x[keep], h[keep] * 100, LS[m], color=COL[m], lw=1.3 if m == "xlsr_aasist" else 1.0,
                    label=NAMES[m])
            ax.plot(d["tau_star_offset_sd"], d["hter_fixed"] * 100, "v", color=COL[m], ms=4.5,
                    mec="k", mew=0.4, zorder=5)
            record.setdefault("sensitivity", {})[f"{m}|{short}"] = {
                "tau_star_offset_sd": d["tau_star_offset_sd"], "hter_fixed": d["hter_fixed"],
                "hter_oracle": d["hter_oracle"], "safe_band_sd": d["safe_band_sd"]}
        ax.axvline(0, color="0.3", lw=0.5)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-1.5, 55)
        ax.set_title(title, pad=3)
        ax.set_xlabel(r"$(\tau-\tau_{\mathrm{oracle}})\,/\,\mathrm{SD}$ of target scores", labelpad=1)
    axes[0].set_ylabel("HTER (%)", labelpad=1)
    h, lab = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], ls="none", marker="v", color="0.5", mec="k", mew=0.4, ms=4.5))
    lab.append(r"frozen $\tau^{*}_{\mathrm{EER}}$")
    fig.legend(h, lab, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.12),
               handlelength=1.8, columnspacing=1.2)
    fig.savefig(OUT / "fig_sensitivity.pdf")
    plt.close(fig)


def fig_frr_far(op, h5, record):
    cells = {(c["model"], c["target"]): c for c in op["cells"] if c["policy"] == "eer"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL_W, 2.0), gridspec_kw={"width_ratios": [1.0, 1.0]})
    fig.subplots_adjust(wspace=0.36)
    a.plot([0, 75], [0, 75], color="0.5", lw=0.6, ls=":")
    a.text(9, 73, "FRR > FAR:\ngenuine speech rejected", fontsize=5.8, va="top", color="0.25")
    a.text(73, 4, "FAR > FRR:\nfakes accepted", fontsize=5.8, ha="right", color="0.25")
    for m in MODELS:
        for t, mk in TGT_MARK.items():
            c = cells[(m, t)]
            a.plot(100 * c["far"], 100 * c["frr"], mk, color=COL[m], ms=4.2, mec="k", mew=0.35)
            record.setdefault("frr_far_tau_eer", {})[f"{m}|{t}"] = {"frr": c["frr"], "far": c["far"]}
    a.set_xlim(0, 75)
    a.set_ylim(0, 75)
    a.set_xlabel("FAR (%)", labelpad=1)
    a.set_ylabel("FRR (%)", labelpad=1)
    a.set_title(r"(a) Transfer, at $\tau^{*}_{\mathrm{EER}}$", pad=3)
    for m in MODELS[:4]:
        r = h5["per_model"][m]
        x0, y0, x1, y1 = (100 * r["far_clean"], 100 * r["frr_clean"],
                          100 * r["far_encodec"], 100 * r["frr_encodec"])
        b.annotate("", xy=(x1, y1), xytext=(x0, y0),
                   arrowprops={"arrowstyle": "-|>", "color": COL[m], "lw": 1.0, "mutation_scale": 7,
                               "shrinkA": 0, "shrinkB": 0})
        b.plot(x0, y0, "o", color=COL[m], ms=3.2, mec="k", mew=0.35)
        b.plot(x1, y1, "o", color=COL[m], ms=3.2, mec="k", mew=0.35, mfc="white")
        record.setdefault("h5_clean_to_encodec", {})[m] = {k: r[k] for k in
                                                           ("frr_clean", "frr_encodec", "far_clean", "far_encodec")}
    b.set_xlim(0, 16)
    b.set_ylim(-2, 72)
    b.set_xlabel("FAR (%)", labelpad=1)
    b.set_ylabel("FRR (%)", labelpad=1)
    b.set_title(r"(b) Clean $\rightarrow$ EnCodec 6 kbps", pad=3)
    hm = [plt.Line2D([], [], ls="none", marker="s", color=COL[m], ms=4) for m in MODELS]
    lm = [NAMES[m].replace(" (ref.)", "") for m in MODELS]
    ht = [plt.Line2D([], [], ls="none", marker=mk, mfc="white", mec="k", mew=0.5, ms=4)
          for mk in TGT_MARK.values()]
    ht += [plt.Line2D([], [], ls="none", marker="o", color="0.55", mec="k", mew=0.35, ms=3.2),
           plt.Line2D([], [], ls="none", marker="o", mfc="white", mec="k", mew=0.35, ms=3.2)]
    lt = list(TGT_NAME.values()) + ["clean", "EnCodec"]
    leg = fig.legend(hm, lm, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.13),
                     fontsize=5.6, handletextpad=0.15, columnspacing=0.7, handlelength=0.9)
    fig.add_artist(leg)
    fig.legend(ht, lt, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.21),
               fontsize=5.6, handletextpad=0.15, columnspacing=0.7, handlelength=0.9)
    fig.savefig(OUT / "fig_frr_far.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    op = json.loads((R / "results" / "analysis" / "operating_points.json").read_text(encoding="utf-8"))
    h5 = json.loads((R / "results" / "analysis" / "h5_test.json").read_text(encoding="utf-8"))
    data = paper_figures.collect()
    committed = json.loads((R / "paper" / "figures" / "figures_data.json").read_text(encoding="utf-8"))
    for (m, t), d in data.items():   # the curves must be the ones the TIFS figure was drawn from
        for k, v in committed[f"{m}|{t}"].items():
            if abs(d[k] - v) > 1e-12:
                raise SystemExit(f"REFUSED: {m}|{t} {k}: {d[k]} != committed {v}")
    record = {"sources": ["results/analysis/operating_points.json", "results/analysis/h5_test.json",
                          "paper_figures.collect() == paper/figures/figures_data.json"]}
    fig_sensitivity(data, record)
    fig_frr_far(op, h5, record)
    (OUT / "conference_figures_data.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("*"))))


if __name__ == "__main__":
    main()
