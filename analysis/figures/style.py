"""Publication figure style (IEEEtran column widths)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL_W = 3.5      # single column inches
PAGE_W = 7.16    # double column
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 200, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.3, "lines.linewidth": 1.2,
})
