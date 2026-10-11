"""Shared library for the round-2 analyses (PREREGISTRATION_ROUND2.md; D54-D57).

Conventions are the project's: label 1 = bona fide, higher score = more bona fide, SPOOF if score < tau,
FRR = P(bona < tau), FAR = P(spoof >= tau). Thresholds come only from deploybench.tau (via r2_taus.py) and are read
here, never re-tuned. The cluster-bootstrap machinery is ported verbatim from the DEV-8 revision library
(DFV_Conference/revision/analysis/revlib.py: cluster_counts, cluster_aggregates, cluster_rates, Oracle) and is
checked against DEV-8's committed results by tests/test_round2_analysis.py.

Score files (written by score.py, gzipped when committed):
  round 2:  results/round2/scores/<model>/s<seed>/<surface>.csv[.gz]     (seed 0 = a released checkpoint)
  round 1:  the committed fp32 files (phase_1_reproduction, main_matrix, mlaad5), plus their round-2 stress scores
            under results/round2/scores/<model>/s0/stress_<condition>.csv[.gz]
Surfaces: la19_dev, la19_eval, itw, df21_verified, mlaad5, as5_dev, as5_eval, stress_<condition>.
Metadata (speaker / system clusters) is read from $R2_META if set, else from the data tree under $DATA_ROOT.
"""
from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from deploybench.metrics import det_curve, eer, far_at, frr_at, hter_at, holm_bonferroni  # noqa: E402

P1 = REPO / "results" / "phase_1_reproduction"
MM = REPO / "results" / "phase_2_conversion_app" / "main_matrix"
ML = REPO / "results" / "phase_2_conversion_app" / "mlaad5"
R2 = REPO / "results" / "round2"
R2_SCORES = R2 / "scores"
R2_TAUS = R2 / "taus"

ROUND1 = ("aasist", "aasist_l", "rawnet2", "tssdnet", "xlsr_aasist")
PRIMARY4 = ROUND1[:4]
DETECTORS_R2 = ("as5_aasist", "pascu_lr", "xlsr_mamba")
RUN_SEEDS = (1, 2, 3, 4, 5)
PIPELINE_SEED = 1234
POLICIES5 = ("eer", "frr1", "frr5", "far1", "far5")
STRESS_NEW = ("opus6k", "opus24k", "amr4.75", "mp3_32", "snr20", "snr10", "snr5", "mic_ir")
SURFACE_N = {"la19_dev": 24844, "la19_eval": 71237, "itw": 31779, "df21_verified": 455508, "mlaad5": 494853,
             "as5_dev": 140950, "as5_eval": 680774}
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "~/data/deepfake")).expanduser()


def die(msg):
    raise SystemExit(f"REFUSED: {msg}")


# --------------------------------------------------------------------------- scores
def _open(p: Path):
    if p.exists():
        return p
    gz = p.with_name(p.name + ".gz")
    if gz.exists():
        return gz
    die(f"score file missing: {p}[.gz]")


def read_scores(p: Path):
    """(utt_ids, labels, scores) of one fp32 score file. Refuses non-fp32 rows, labels other than {0, 1}, NaN."""
    p = _open(Path(p))
    df = pd.read_csv(p, usecols=["utt_id", "label", "score", "precision"], dtype={"utt_id": str},
                     float_precision="round_trip", compression="infer")
    if set(df["precision"]) != {"fp32"}:
        die(f"{p}: not all fp32")
    if set(df["label"].unique()) != {0, 1}:
        die(f"{p}: labels are not exactly {{0, 1}}")
    s = df["score"].to_numpy(np.float64)
    if not np.all(np.isfinite(s)):
        die(f"{p}: non-finite scores")
    if df["utt_id"].duplicated().any():
        die(f"{p}: duplicate utt_ids")
    return df["utt_id"].to_numpy(), df["label"].to_numpy(np.int64), s


def r1_path(model: str, surface: str) -> Path:
    if surface.startswith("stress_"):
        return R2_SCORES / model / "s0" / f"{surface}.csv"
    paths = {"la19_dev": P1 / f"scores_dev_{model}.csv",
             "la19_eval": P1 / f"scores_repro_{model}.csv",
             "itw": MM / model / "scores_fp32_pytorch_itw.csv",
             "df21_verified": MM / model / "scores_fp32_pytorch_df21_verified.csv",
             "mlaad5": ML / model / "scores_fp32_pytorch_mlaad.csv"}
    if surface not in paths:                 # e.g. as5_eval: a target for as5_aasist only (D24, D54)
        die(f"{model}: round 1 has no {surface} scores")
    return paths[surface]


def score_path(model: str, surface: str, seed: int = 0) -> Path:
    if model in ROUND1:
        return r1_path(model, surface)
    return R2_SCORES / model / f"s{seed}" / f"{surface}.csv"


_cache = {}


def load(model: str, surface: str, seed: int = 0, check_n: bool = True):
    k = (model, surface, seed)
    if k not in _cache:
        ids, y, s = read_scores(score_path(model, surface, seed))
        n_want = SURFACE_N.get(surface)
        if surface.startswith("stress_"):
            n_want = 1000
        if check_n and n_want is not None and len(y) != n_want:
            die(f"{model}/{surface}/s{seed}: n={len(y)}, the registered surface has {n_want}")
        _cache[k] = (ids, y, s)
    return _cache[k]


def taus(model: str, seed: int = 0) -> dict:
    """The five frozen policies. Round 1: tau_<m>.json (eer, frr1, frr5) + the DEV-8 FAR budgets from the same dev
    scores through deploybench.tau. Round 2: results/round2/taus/<model>__s<seed>.json (r2_taus.py)."""
    from deploybench.tau import tau_at_far
    if model in ROUND1:
        j = json.loads((P1 / f"tau_{model}.json").read_text(encoding="utf-8"))
        t = dict(j["taus"])
        _, y, s = load(model, "la19_dev")
        t["far1"] = tau_at_far(s, y, 0.01)
        t["far5"] = tau_at_far(s, y, 0.05)
        return t
    f = R2_TAUS / f"{model}__s{seed}.json"
    if not f.exists():
        die(f"{f} missing: run r2_taus.py on the source-dev scores first")
    return json.loads(f.read_text(encoding="utf-8"))["taus"]


# --------------------------------------------------------------------------- metadata -> clusters
def meta_file(kind: str) -> Path:
    names = {"itw": "meta.csv", "la19_eval": "ASVspoof2019.LA.cm.eval.trl.txt",
             "la19_dev": "ASVspoof2019.LA.cm.dev.trl.txt", "df21": "trial_metadata.txt",
             "mlaad5": "protocol.csv", "as5_dev": "ASVspoof5.dev.track_1.tsv", "as5_eval": "ASVspoof5.eval.track_1.tsv"}
    data = {"itw": DATA_ROOT / "release_in_the_wild" / "meta.csv",
            "la19_eval": DATA_ROOT / "LA" / "ASVspoof2019_LA_cm_protocols" / names["la19_eval"],
            "la19_dev": DATA_ROOT / "LA" / "ASVspoof2019_LA_cm_protocols" / names["la19_dev"],
            "df21": DATA_ROOT / "keys" / "DF" / "CM" / "trial_metadata.txt",
            "mlaad5": DATA_ROOT / "mlaad5" / "protocol.csv",
            "as5_dev": DATA_ROOT / "asvspoof5" / names["as5_dev"],
            "as5_eval": DATA_ROOT / "asvspoof5" / names["as5_eval"]}
    m = os.environ.get("R2_META")
    if m and (Path(m) / names[kind]).exists():
        return Path(m) / names[kind]
    if data[kind].exists():
        return data[kind]
    die(f"metadata for {kind} not found ($R2_META or {data[kind]})")


def _speaker_map(kind: str) -> dict:
    if kind == "itw":
        df = pd.read_csv(meta_file("itw"), dtype=str)
        return dict(zip(df["file"].str.replace(r"\.wav$", "", regex=True), df["speaker"]))
    if kind in ("la19_eval", "la19_dev"):
        out = {}
        for line in open(meta_file(kind), encoding="utf-8"):
            f = line.split()
            if len(f) >= 5:
                out[f[1]] = f[0]
        return out
    if kind == "df21":
        out = {}
        for line in open(meta_file("df21"), encoding="utf-8"):
            f = line.split()
            if len(f) >= 2:
                out[f[1]] = f[0]
        return out
    if kind == "mlaad5":
        # bona fide: language + M-AILABS speaker (the component after male/female; French has no by_book/ level),
        # or language + book for the multi-speaker mix/ folders; spoof: the TTS system.
        df = pd.read_csv(meta_file("mlaad5"), dtype=str)
        out = {}
        for u, p, lab, sysname in zip(df["utt_id"], df["path"], df["label"], df["system"]):
            if lab == "1":
                parts = Path(p).parts
                lang = parts[1]
                k = next((i for i, x in enumerate(parts) if x in ("male", "female", "mix")), None)
                if k is None or k + 1 >= len(parts):
                    die(f"mlaad5: bona fide path without a male/female/mix component: {p}")
                out[u] = (f"bona:{lang}/mix/{parts[k + 1]}" if parts[k] == "mix" else f"bona:{lang}/{parts[k + 1]}")
            else:
                out[u] = f"spoof:{sysname}"
        return out
    if kind in ("as5_dev", "as5_eval"):
        out = {}
        for line in open(meta_file(kind), encoding="utf-8"):
            f = line.split()
            out[f[1]] = f[0]
        return out
    raise ValueError(kind)


_smap = {}


def clusters(surface: str, ids):
    """Cluster codes (0..G-1, sorted names) and names for a surface's utt_ids."""
    kind = {"la19_eval": "la19_eval", "itw": "itw", "df21_verified": "df21", "mlaad5": "mlaad5",
            "as5_dev": "as5_dev", "as5_eval": "as5_eval", "la19_dev": "la19_dev"}.get(surface)
    if surface.startswith("stress_"):
        kind = "la19_eval"
    if kind is None:
        raise ValueError(surface)
    if kind not in _smap:
        _smap[kind] = _speaker_map(kind)
    m = _smap[kind]
    names = [m.get(u) for u in ids]
    if any(n is None for n in names):
        die(f"{surface}: {sum(n is None for n in names)} utt_ids have no metadata row")
    uniq, codes = np.unique(np.asarray(names, dtype=str), return_inverse=True)
    return codes.astype(np.int64), list(uniq)


# --------------------------------------------------------------------------- cluster bootstrap (ported from revlib)
def cluster_counts(G, B, seed):
    """(B, G) multinomial draws: how often each of the G clusters is drawn in each replicate."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(G, np.full(G, 1.0 / G), size=B).astype(np.float64)


def cluster_aggregates(s, y, g, G, tau):
    b = y == 1
    nb = np.bincount(g[b], minlength=G).astype(np.float64)
    rej = np.bincount(g[b & (s < tau)], minlength=G).astype(np.float64)
    ns = np.bincount(g[~b], minlength=G).astype(np.float64)
    acc = np.bincount(g[(~b) & (s >= tau)], minlength=G).astype(np.float64)
    return nb, rej, ns, acc


def cluster_rates(s, y, g, G, tau, C):
    """FRR and FAR in every replicate (rows of C) at a fixed tau."""
    nb, rej, ns, acc = cluster_aggregates(s, y, g, G, tau)
    return (C @ rej) / (C @ nb), (C @ acc) / (C @ ns)


class Oracle:
    """Replicate-wise oracle HTER: min over thresholds (the unique scores plus the two trivial systems, as
    metrics.det_curve) of HTER, capped at 0.5. With every cluster drawn once it equals metrics.min_hter."""

    def __init__(self, s, y, g):
        order = np.argsort(s, kind="stable")
        ss = s[order]
        self.g = g[order]
        yy = y[order]
        self.isb = (yy == 1).astype(np.float64)
        self.isp = 1.0 - self.isb
        chg = np.concatenate([[True], ss[1:] != ss[:-1]])
        self.first = np.nonzero(chg)[0]

    def run(self, C):
        B = C.shape[0]
        h_or = np.empty(B)
        first = self.first
        for r in range(B):
            w = C[r][self.g]
            wb, wp = w * self.isb, w * self.isp
            cb = np.cumsum(wb)[first - 1]
            cp = np.cumsum(wp)[first - 1]
            cb[0] = 0.0
            cp[0] = 0.0
            Nb, Np = wb.sum(), wp.sum()
            frr = cb / Nb
            far = 1.0 - cp / Np
            h_or[r] = min((0.5 * (frr + far)).min(), 0.5)
        return h_or


def paired_cluster_test(y, s_clean, s_cond, g, G, tau, C):
    """dFRR, dFAR and dFRR - dFAR per replicate (rows of C) for paired clean / condition scores of the same clips
    (DEV-8 A2 / R2-H4). Returns (point dfrr, point dfar, replicate dfrr, replicate dfar)."""
    b, p = y == 1, y == 0
    nb = np.bincount(g[b], minlength=G).astype(float)
    ns = np.bincount(g[p], minlength=G).astype(float)
    rc = np.bincount(g[b & (s_clean < tau)], minlength=G).astype(float)
    rx = np.bincount(g[b & (s_cond < tau)], minlength=G).astype(float)
    ac = np.bincount(g[p & (s_clean >= tau)], minlength=G).astype(float)
    ax = np.bincount(g[p & (s_cond >= tau)], minlength=G).astype(float)
    dfrr = (C @ (rx - rc)) / (C @ nb)
    dfar = (C @ (ax - ac)) / (C @ ns)
    return (float((rx.sum() - rc.sum()) / nb.sum()), float((ax.sum() - ac.sum()) / ns.sum()), dfrr, dfar)


def pval_le0(stat):
    """One-sided bootstrap p for H0: statistic <= 0, p = (1 + #{stat <= 0}) / (B + 1)."""
    return float((1 + np.sum(stat <= 0)) / (len(stat) + 1))


def pct(a, lo=2.5, hi=97.5):
    return [float(np.percentile(a, lo)), float(np.percentile(a, hi))]


def holm(pvals: dict, alpha=0.05) -> dict:
    return holm_bonferroni(pvals, alpha)


# --------------------------------------------------------------------------- point quantities
def tie_interval(s_dev, y_dev, tau_eer):
    """DEV-8 B3: the connected run of dev thresholds around tau_eer whose dev HTER is within half a bona fide trial
    (0.5 / n_bona) of HTER(tau_eer). Returns (tau_lo, tau_hi)."""
    grid, frr, far = det_curve(s_dev, y_dev)
    h = 0.5 * (frr + far)
    tol = 0.5 / int((y_dev == 1).sum())
    ok = h <= hter_at(tau_eer, s_dev, y_dev) + tol
    j = int(np.searchsorted(grid, tau_eer))
    cand = [p for p in (j - 1, j) if 0 <= p < len(grid) and ok[p]]
    if not cand:
        return float(tau_eer), float(tau_eer)
    lo = hi = cand[0]
    while lo - 1 >= 0 and ok[lo - 1]:
        lo -= 1
    while hi + 1 < len(grid) and ok[hi + 1]:
        hi += 1
    return float(grid[lo]), float(grid[hi])


def point(s, y, tau):
    """FRR, FAR, HTER at tau; oracle HTER; TTG; transfer share."""
    fr, fa = frr_at(tau, s, y), far_at(tau, s, y)
    h = 0.5 * (fr + fa)
    g = np.zeros(len(y), dtype=np.int64)
    h_or = float(Oracle(s, y, g).run(np.ones((1, 1)))[0])
    return {"frr": fr, "far": fa, "hter": h, "hter_oracle": h_or, "ttg": h - h_or,
            "transfer_share": (h - h_or) / h if h > 0 else float("nan"), "eer": eer(s, y)[0]}


def write(name: str, obj: dict, text: list):
    out = R2 / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(obj, indent=1), encoding="utf-8")
    (out / f"{name}.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
