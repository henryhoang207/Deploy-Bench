"""Figures from analysis output CSVs. Each function = one paper figure;
regenerable by one command:  python analysis/figures/make_figures.py --in analysis/out/"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import COL_W, PAGE_W  # noqa: E402


def fig_ttg_headline(ttg: pd.DataFrame, out: Path):
    """Fig 1 candidate: oracle EER vs fixed-tau HTER per model (the gap IS the finding)."""
    d = ttg[(ttg.policy == "eer") & (ttg.precision == "fp32")]
    agg = d.groupby(["model", "condition"]).agg(
        oracle=("oracle_eer", "mean"), fixed=("hter_fixed", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    for _, r in agg.iterrows():
        ax.plot([0, 1], [r.oracle, r.fixed], "o-", ms=3, alpha=0.6)
        ax.annotate(f"{r.model[:8]}/{r.condition[:6]}", (1, r.fixed),
                    fontsize=5, alpha=0.7)
    ax.set_xticks([0, 1], ["oracle EER", "HTER @ transferred tau"])
    ax.set_ylabel("error rate")
    fig.savefig(out / "fig_ttg_headline.pdf")
    plt.close(fig)


def fig_frr_far_heatmap(ttg: pd.DataFrame, out: Path):
    d = ttg[(ttg.policy == "eer") & (ttg.precision == "fp32")]
    piv_frr = d.pivot_table(index="model", columns="condition", values="frr_fixed")
    piv_far = d.pivot_table(index="model", columns="condition", values="far_fixed")
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.2))
    for ax, piv, title in [(axes[0], piv_frr, "FRR (bona fide rejected)"),
                           (axes[1], piv_far, "FAR (spoof accepted)")]:
        im = ax.imshow(piv.values, cmap="Reds", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(piv.columns)), piv.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(piv.index)), piv.index)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(out / "fig_frr_far_heatmap.pdf")
    plt.close(fig)


def fig_qts(qts: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    for i, (model, g) in enumerate(qts.groupby("model")):
        ax.scatter([i] * len(g), g.qts, s=8, alpha=0.6)
    ax.set_xticks(range(qts.model.nunique()), sorted(qts.model.unique()),
                  rotation=45, ha="right")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("QTS (HTER penalty of stale tau)")
    fig.savefig(out / "fig_qts.pdf")
    plt.close(fig)


def fig_pareto(device_csv: Path, ttg: pd.DataFrame, out: Path):
    """Needs device metrics (phase 3): csv with model,phone,precision,latency_p50_ms,energy_mj,mem_mb."""
    if not device_csv.exists():
        return
    dev = pd.read_csv(device_csv)
    acc = ttg[(ttg.policy == "eer") & (ttg.condition == "itw")] \
        .groupby(["model", "precision"]).hter_fixed.mean().reset_index()
    m = dev.merge(acc, on=["model", "precision"])
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    for phone, g in m.groupby("phone"):
        ax.scatter(g.latency_p50_ms, g.hter_fixed, s=g.mem_mb / 4,
                   alpha=0.6, label=phone)
    ax.set_xlabel("latency p50 (ms)"); ax.set_ylabel("HTER@tau* (ITW)")
    ax.set_xscale("log"); ax.legend()
    fig.savefig(out / "fig_pareto.pdf")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    a = ap.parse_args()
    ind = Path(a.indir)
    ttg = pd.read_csv(ind / "ttg_table.csv")
    qts = pd.read_csv(ind / "qts_table.csv")
    fig_ttg_headline(ttg, ind)
    fig_frr_far_heatmap(ttg, ind)
    if len(qts):
        fig_qts(qts, ind)
    fig_pareto(ind / "device_metrics.csv", ttg, ind)
    print(f"figures -> {ind}")
