"""Results-section figures, from the committed fp32 score files and the frozen 19LA-dev taus.

    python analysis/figures/paper_figures.py            # -> paper/figures/*.pdf + figures_data.json

fig_ttg.pdf          per model and target: oracle min-HTER vs HTER at the frozen tau*_eer
                     (the gap is TTG, D19).
fig_frr_far.pdf      FRR and FAR at tau*_eer, models x targets (the decomposition behind TTG).
fig_sensitivity.pdf  HTER(tau) on each target, tau in target-score SD units from the target
                     oracle; marks the frozen tau*_eer; the 1-pp safe band
                     (deploybench.metrics.threshold_sensitivity, band = 0.01).
Descriptive only: no test is computed here. Numbers plotted are written to figures_data.json.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from deploybench.metrics import det_curve, far_at, frr_at, hter_at, min_hter  # noqa: E402
from style import COL_W, PAGE_W, plt  # noqa: E402

MODELS = ("aasist", "aasist_l", "rawnet2", "tssdnet", "xlsr_aasist")
NAMES = {"aasist": "AASIST", "aasist_l": "AASIST-L", "rawnet2": "RawNet2",
         "tssdnet": "Res-TSSDNet", "xlsr_aasist": "XLS-R+AASIST$^{ref}$"}
P1 = R / "results" / "phase_1_reproduction"
MM = R / "results" / "phase_2_conversion_app" / "main_matrix"
TARGETS = {"19LA eval": lambda m: P1 / f"scores_repro_{m}.csv",
           "DF21 verified": lambda m: MM / m / "scores_fp32_pytorch_df21_verified.csv",
           "ITW": lambda m: MM / m / "scores_fp32_pytorch_itw.csv"}
BAND = 0.01
OUT = R / "paper" / "figures"


def load(p: Path):
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    if {r["precision"] for r in rows} != {"fp32"}:
        raise SystemExit(f"{p}: not all fp32")
    return (np.array([float(r["score"]) for r in rows]),
            np.array([int(r["label"]) for r in rows]))


def collect():
    data = {}
    for m in MODELS:
        tau = json.loads((P1 / f"tau_{m}.json").read_text())["taus"]["eer"]
        for t, f in TARGETS.items():
            p = f(m)
            if not p.is_file():
                raise SystemExit(f"missing {p}")
            s, y = load(p)
            h_or, tau_or = min_hter(s, y)
            taus, frr, far = det_curve(s, y)
            h = 0.5 * (frr + far)
            sd = float(np.std(s))
            ok = taus[h <= h.min() + BAND]
            data[(m, t)] = {
                "n": int(len(y)), "tau_star": tau, "tau_oracle": tau_or, "score_sd": sd,
                "hter_fixed": hter_at(tau, s, y), "hter_oracle": h_or,
                "frr_fixed": frr_at(tau, s, y), "far_fixed": far_at(tau, s, y),
                "safe_band_sd": float((ok.max() - ok.min()) / sd) if len(ok) else 0.0,
                "tau_star_offset_sd": float((tau - tau_or) / sd),
                "_curve": ((taus - tau_or) / sd, h)}
    return data


def fig_ttg(data):
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, 1.9), sharey=True)
    for ax, t in zip(axes, TARGETS):
        for i, m in enumerate(MODELS):
            d = data[(m, t)]
            ax.plot([d["hter_oracle"] * 100, d["hter_fixed"] * 100], [i, i], "-", color="0.6", lw=1.5)
            ax.plot(d["hter_oracle"] * 100, i, "o", mfc="white", mec="k", ms=4)
            ax.plot(d["hter_fixed"] * 100, i, "o", color="C3", ms=4)
        ax.set_title(t)
        ax.set_xlabel("HTER (%)")
        ax.set_yticks(range(len(MODELS)), [NAMES[m] for m in MODELS])
        ax.invert_yaxis()
    axes[0].plot([], [], "o", mfc="white", mec="k", label="target oracle")
    axes[0].plot([], [], "o", color="C3", label=r"frozen $\tau^*_{\mathrm{EER}}$")
    axes[0].legend(loc="lower right", frameon=False)
    fig.savefig(OUT / "fig_ttg.pdf")
    plt.close(fig)


def fig_frr_far(data):
    tg = list(TARGETS)
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 1.9), sharey=True)
    fig.subplots_adjust(wspace=0.08)
    for ax, key, title in ((axes[0], "frr_fixed", r"FRR at $\tau^*_{\mathrm{EER}}$ (bona fide rejected)"),
                           (axes[1], "far_fixed", r"FAR at $\tau^*_{\mathrm{EER}}$ (spoof accepted)")):
        v = np.array([[data[(m, t)][key] * 100 for t in tg] for m in MODELS])
        ax.imshow(v, cmap="Reds", vmin=0, vmax=100, aspect="auto")
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                ax.text(j, i, f"{v[i, j]:.1f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v[i, j] > 55 else "black")
        ax.set_xticks(range(len(tg)), tg)
        ax.set_yticks(range(len(MODELS)), [NAMES[m] for m in MODELS])
        ax.tick_params(axis="y", labelleft=ax is axes[0])
        ax.set_title(title)
        ax.grid(False)
    fig.savefig(OUT / "fig_frr_far.pdf")
    plt.close(fig)


def fig_sensitivity(data):
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, 1.9), sharey=True)
    for ax, t in zip(axes, TARGETS):
        for i, m in enumerate(MODELS):
            d = data[(m, t)]
            x, h = d["_curve"]
            keep = np.abs(x) <= 3
            ax.plot(x[keep], h[keep] * 100, color=f"C{i}", lw=1, label=NAMES[m])
            ax.plot(d["tau_star_offset_sd"], d["hter_fixed"] * 100, "v", color=f"C{i}", ms=4)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlim(-3, 3)
        ax.set_title(t)
        ax.set_xlabel(r"$\tau-\tau_{\mathrm{oracle}}$ (target score SD)")
    axes[0].set_ylabel("HTER (%)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(MODELS), frameon=False,
               bbox_to_anchor=(0.5, -0.3))
    fig.savefig(OUT / "fig_sensitivity.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = collect()
    fig_ttg(data)
    fig_frr_far(data)
    fig_sensitivity(data)
    js = {f"{m}|{t}": {k: v for k, v in d.items() if not k.startswith("_")}
          for (m, t), d in data.items()}
    (OUT / "figures_data.json").write_text(json.dumps(js, indent=2) + "\n")
    for k, d in js.items():
        print(f"{k:28s} TTG {100*(d['hter_fixed']-d['hter_oracle']):6.2f} pp  "
              f"FRR {100*d['frr_fixed']:5.1f} FAR {100*d['far_fixed']:5.1f}  "
              f"tau* offset {d['tau_star_offset_sd']:+.2f} SD  safe band {d['safe_band_sd']:.2f} SD")


if __name__ == "__main__":
    main()
