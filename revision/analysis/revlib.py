"""Shared library for the DEV-8 sensitivity analyses (declared in deploybench DEVIATIONS.md DEV-8, 2026-10-04).

Reads the committed fp32 score files, the frozen taus and the committed JSON of the deploybench clone READ-ONLY, plus
the dataset metadata in revision/meta (see revision/meta/README.md). The harness estimators are IMPORTED, never re-implemented, wherever the
registered analysis used them (det_curve, eer, frr_at, far_at, ttg, tau_at_frr, tau_at_far, holm_bonferroni).

Conventions are the project's: label 1 = bona fide, higher score = more bona fide, SPOOF if score < tau,
FRR = P(bona < tau), FAR = P(spoof >= tau). Nothing here changes a score, a threshold file or a registered result.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLONE = Path(os.environ.get("DEPLOYBENCH", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(CLONE))
from deploybench.manifests import read_ids  # noqa: E402
from deploybench.metrics import (det_curve, eer, far_at, frr_at, holm_bonferroni,  # noqa: E402
                                 hter_at, min_hter, ttg)
from deploybench.tau import calibrate_tau, tau_at_far, tau_at_frr  # noqa: E402

HERE = Path(__file__).parent
REV = HERE.parent
META = Path(os.environ.get("DFV_META", REV / "meta"))
RESULTS = REV / "results"
P1 = CLONE / "results" / "phase_1_reproduction"
MM = CLONE / "results" / "phase_2_conversion_app" / "main_matrix"
H5DIR = CLONE / "results" / "phase_3_stress" / "h5"
COMMITTED = CLONE / "results" / "analysis"

MODELS = ("aasist", "aasist_l", "rawnet2", "tssdnet", "xlsr_aasist")
RANKED = MODELS[:4]
NICE = {"aasist": "AASIST", "aasist_l": "AASIST-L", "rawnet2": "RawNet2", "tssdnet": "Res-TSSDNet",
        "xlsr_aasist": "XLS-R+AASIST"}
TARGETS = ("19la_eval", "itw", "df21_verified")
TARGET_N = {"19la_eval": 71237, "itw": 31779, "df21_verified": 455508}
POLICIES3 = ("eer", "frr1", "frr5")                       # registered
POLICIES5 = ("eer", "frr1", "frr5", "far1", "far5")       # + the DEV-8 FAR-budget policies
COSTS = {"screening": {"p_bona": 0.95, "c_frr": 1.0, "c_far": 1.0},
         "forensic": {"p_bona": 0.5, "c_frr": 1.0, "c_far": 10.0}}
TOL = 1e-12


def die(msg):
    raise SystemExit(f"REFUSED: {msg}")


def cost_norm(k):
    c = COSTS[k]
    return min(c["p_bona"] * c["c_frr"], (1.0 - c["p_bona"]) * c["c_far"])


# --------------------------------------------------------------------------- scores
def _read_scores(p: Path):
    df = pd.read_csv(p, usecols=["utt_id", "label", "score", "precision"],
                     dtype={"utt_id": str}, float_precision="round_trip")
    if set(df["precision"]) != {"fp32"}:
        die(f"{p}: not all fp32")
    if set(df["label"].unique()) != {0, 1}:
        die(f"{p}: labels are not exactly {{0, 1}}")
    return df["utt_id"].to_numpy(), df["label"].to_numpy(np.int64), df["score"].to_numpy(np.float64)


def score_path(model, target):
    return {"dev": P1 / f"scores_dev_{model}.csv",
            "19la_eval": P1 / f"scores_repro_{model}.csv",
            "itw": MM / model / "scores_fp32_pytorch_itw.csv",
            "df21_verified": MM / model / "scores_fp32_pytorch_df21_verified.csv"}[target]


_cache = {}


def load(model, target):
    """(utt_ids, labels, scores) of one committed fp32 score file; cached."""
    k = (model, target)
    if k not in _cache:
        ids, y, s = _read_scores(score_path(model, target))
        n_want = {"dev": 24844, **TARGET_N}[target]
        if len(y) != n_want:
            die(f"{score_path(model, target)}: n={len(y)}, frozen surface has {n_want}")
        _cache[k] = (ids, y, s)
    return _cache[k]


def frozen_taus(model):
    j = json.loads((P1 / f"tau_{model}.json").read_text(encoding="utf-8"))
    return j["taus"], j


def all_taus(model):
    """The three registered taus (frozen files) and the DEV-8 FAR-budget taus (dev-only, same harness rule)."""
    t, _ = frozen_taus(model)
    _, y, s = load(model, "dev")
    t = dict(t)
    t["far1"] = tau_at_far(s, y, 0.01)
    t["far5"] = tau_at_far(s, y, 0.05)
    return t


# --------------------------------------------------------------------------- metadata -> clusters
def itw_meta():
    df = pd.read_csv(META / "meta.csv", dtype=str)
    df["utt_id"] = df["file"].str.replace(r"\.wav$", "", regex=True)
    return df.set_index("utt_id")


def la19_protocol(part):
    rows = {}
    for line in open(META / f"ASVspoof2019.LA.cm.{part}.trl.txt", encoding="utf-8"):
        f = line.split()
        if len(f) >= 5:
            rows[f[1]] = (f[0], f[3], 1 if f[4] == "bonafide" else 0)    # speaker, attack, label
    return rows


DF21_COLS = ("speaker", "trial", "compr", "source", "attack", "label", "trim", "subset", "vocoder", "task", "team",
             "gender-pair", "language")


def df21_meta():
    df = pd.read_csv(META / "trial_metadata.txt", sep=r"\s+", header=None, names=DF21_COLS, dtype=str)
    return df.set_index("trial")


def clusters(target, ids):
    """Cluster codes (0..G-1) and names for a target's utt_ids: ITW speaker, LA19-eval speaker, DF21 speaker."""
    if target == "itw":
        m = itw_meta()["speaker"]
        names = m.reindex(ids)
    elif target == "19la_eval":
        p = la19_protocol("eval")
        names = pd.Series([p[u][0] for u in ids], index=ids)
    elif target == "df21_verified":
        names = df21_meta()["speaker"].reindex(ids)
    else:
        raise ValueError(target)
    if names.isna().any():
        die(f"{target}: {int(names.isna().sum())} utt_ids have no metadata row")
    # canonical (sorted-name) codes, so the same speaker has the same code for every model's file
    uniq, codes = np.unique(names.to_numpy().astype(str), return_inverse=True)
    return codes.astype(np.int64), list(uniq)


# --------------------------------------------------------------------------- point estimates
def point(model, target, tau):
    ids, y, s = load(model, target)
    d = ttg(s, y, tau)
    return {"frr": d["frr_fixed"], "far": d["far_fixed"], "hter": d["hter_fixed"], "hter_oracle": d["hter_oracle"],
            "ttg": d["ttg"], "tau_oracle": d["tau_oracle"]}


def cost_point(model, target, tau):
    """Normalised costs at tau and at the oracle (min over thresholds and the two trivial systems), as
    analysis/operating_points.py computes them."""
    ids, y, s = load(model, target)
    taus, frr, far = det_curve(s, y)
    out = {}
    for k, c in COSTS.items():
        z = cost_norm(k)
        fixed = (c["p_bona"] * c["c_frr"] * frr_at(tau, s, y) + (1 - c["p_bona"]) * c["c_far"] * far_at(tau, s, y)) / z
        curve = (c["p_bona"] * c["c_frr"] * frr + (1.0 - c["p_bona"]) * c["c_far"] * far) / z
        out[k] = {"c_fixed": float(fixed), "c_min": float(curve.min()), "gap": float(fixed - curve.min())}
    return out


# --------------------------------------------------------------------------- cluster bootstrap machinery
def cluster_counts(G, B, seed):
    """(B, G) multinomial draws: how often each of the G clusters is drawn in each replicate (G draws, with
    replacement, equal probability). Shared across models so replicates are paired."""
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
    """Replicate-wise oracle quantities for one (model, target): min over thresholds of HTER and of the two
    normalised costs (thresholds = the unique scores plus the two trivial systems, as det_curve). With C = one draw of
    every cluster exactly once the results equal metrics.min_hter / the committed oracle costs (gate)."""

    def __init__(self, s, y, g):
        order = np.argsort(s, kind="stable")
        ss = s[order]
        self.g = g[order]
        yy = y[order]
        self.isb = (yy == 1).astype(np.float64)
        self.isp = 1.0 - self.isb
        chg = np.concatenate([[True], ss[1:] != ss[:-1]])
        self.first = np.nonzero(chg)[0]
        self.zc = {k: cost_norm(k) for k in COSTS}

    def run(self, C, cost_names=("screening", "forensic")):
        B = C.shape[0]
        h_or = np.empty(B)
        c_or = {k: np.empty(B) for k in cost_names}
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
            for k in cost_names:
                c = COSTS[k]
                z = self.zc[k]
                cost = (c["p_bona"] * c["c_frr"] * frr + (1 - c["p_bona"]) * c["c_far"] * far) / z
                triv = min((1 - c["p_bona"]) * c["c_far"], c["p_bona"] * c["c_frr"]) / z      # accept-all / reject-all
                c_or[k][r] = min(cost.min(), triv)
        return h_or, c_or


def fixed_costs(frr, far, k):
    c = COSTS[k]
    return (c["p_bona"] * c["c_frr"] * frr + (1 - c["p_bona"]) * c["c_far"] * far) / cost_norm(k)


def pct(a, lo=2.5, hi=97.5):
    return [float(np.percentile(a, lo)), float(np.percentile(a, hi))]


# --------------------------------------------------------------------------- H5 inputs
def h5_inputs(model):
    ids = read_ids(CLONE / "configs" / "parity_manifest_tssdnet.txt")
    out = {}
    for cond in ("clean", "encodec6k"):
        rows = pd.read_csv(H5DIR / f"scores_{model}_{cond}.csv", usecols=["utt_id", "label", "score"],
                           dtype={"utt_id": str}, float_precision="round_trip").set_index("utt_id")
        if set(rows.index) != set(ids):
            die(f"{model}/{cond}: H5 score ids differ from the parity manifest")
        out[cond] = rows.reindex(ids)
    if not np.array_equal(out["clean"]["label"].to_numpy(), out["encodec6k"]["label"].to_numpy()):
        die(f"{model}: labels differ between H5 conditions")
    y = out["clean"]["label"].to_numpy(np.int64)
    return ids, y, out["clean"]["score"].to_numpy(np.float64), out["encodec6k"]["score"].to_numpy(np.float64)


# --------------------------------------------------------------------------- the gate
def check(name, got, want, tol=TOL):
    if abs(got - want) > tol:
        die(f"{name}: computed {got!r} != committed {want!r}")


def gate(verbose=True):
    """Refuse (exit) unless every point estimate this library recomputes equals the committed JSON to 1e-12:
    dev taus and the dev file hash, FRR/FAR/HTER/oracle/TTG for 5 models x 3 targets x 3 policies, the H2 deltas, the H5
    rates, the oracle costs, and the cluster-oracle machinery with unit counts."""
    op = json.loads((COMMITTED / "operating_points.json").read_text(encoding="utf-8"))
    cells = {(c["model"], c["target"], c["policy"]): c for c in op["cells"]}
    h2 = json.loads((COMMITTED / "h2_test.json").read_text(encoding="utf-8"))["per_model"]
    h5 = json.loads((COMMITTED / "h5_test.json").read_text(encoding="utf-8"))["per_model"]
    n = 0
    for m in MODELS:
        taus, j = frozen_taus(m)
        ids, y, s = load(m, "dev")
        sha = hashlib.sha256((P1 / f"scores_dev_{m}.csv").read_bytes()).hexdigest()
        if sha != j["dev_scores_sha256"]:
            die(f"{m}: dev score file sha256 differs from tau_{m}.json")
        re = calibrate_tau(s, y)
        for pol in POLICIES3:
            check(f"{m}/dev tau_{pol}", re[pol], taus[pol])
        for tgt in TARGETS:
            ids_t, yt, st = load(m, tgt)
            g, names = clusters(tgt, ids_t)
            C1 = np.ones((1, len(names)))
            h_or, c_or = Oracle(st, yt, g).run(C1)
            for pol in POLICIES3:
                c = cells[(m, tgt, pol)]
                d = point(m, tgt, taus[pol])
                for a, b in (("frr", "frr"), ("far", "far"), ("hter", "hter"), ("hter_oracle", "hter_oracle"),
                             ("ttg", "ttg")):
                    check(f"{m}/{tgt}/{pol}/{a}", d[a], c[b])
                cp = cost_point(m, tgt, taus[pol])
                for k in COSTS:
                    check(f"{m}/{tgt}/{pol}/{k} c_fixed", cp[k]["c_fixed"], c["costs"][k]["c_norm_fixed"])
                    check(f"{m}/{tgt}/{pol}/{k} c_min", cp[k]["c_min"], c["costs"][k]["c_norm_min"])
                n += 1
            check(f"{m}/{tgt}/eer", eer(st, yt)[0], cells[(m, tgt, "eer")]["eer"])
            # cluster machinery with every cluster drawn once == the point estimate
            check(f"{m}/{tgt}/cluster-oracle hter", h_or[0], cells[(m, tgt, "eer")]["hter_oracle"])
            for k in COSTS:
                check(f"{m}/{tgt}/cluster-oracle {k}", c_or[k][0], cells[(m, tgt, "eer")]["costs"][k]["c_norm_min"])
            for pol in POLICIES3:
                fr, fa = cluster_rates(st, yt, g, len(names), taus[pol], C1)
                check(f"{m}/{tgt}/{pol}/cluster frr", fr[0], cells[(m, tgt, pol)]["frr"])
                check(f"{m}/{tgt}/{pol}/cluster far", fa[0], cells[(m, tgt, pol)]["far"])
        if m in RANKED:
            ids_t, yt, st = load(m, "itw")
            r = h2[m]
            check(f"{m}/H2 delta", frr_at(taus["eer"], st, yt) - far_at(taus["eer"], st, yt), r["delta"])
            _, y5, sc, se = h5_inputs(m)
            ib, isp = y5 == 1, y5 == 0
            t = taus["eer"]
            q = h5[m]
            check(f"{m}/H5 frr_clean", float(np.mean(sc[ib] < t)), q["frr_clean"])
            check(f"{m}/H5 frr_encodec", float(np.mean(se[ib] < t)), q["frr_encodec"])
            check(f"{m}/H5 far_clean", float(np.mean(sc[isp] >= t)), q["far_clean"])
            check(f"{m}/H5 far_encodec", float(np.mean(se[isp] >= t)), q["far_encodec"])
        if verbose:
            print(f"  gate OK: {m}", flush=True)
    if verbose:
        print(f"GATE PASSED: {n} model x target x policy cells, taus, dev hashes, H2, H5, cost and oracle machinery "
              f"equal the committed numbers to {TOL:g}", flush=True)


def write_json(name, obj):
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=1, default=float) + "\n", encoding="utf-8")
    return p


if __name__ == "__main__":
    gate()
