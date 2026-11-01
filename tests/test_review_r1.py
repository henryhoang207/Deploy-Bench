"""Tests for the analyses prompted by the round-1 review (analysis/review_r1/; DEV-10). They run on committed score
files, metadata and result files only, without vendor code, checkpoints or GPUs. They import none of the run_*.py
scripts that execute at import and write results/review_r1, and never call run_h2cv_fast.main or run_seed.

- cvlib.Surface.oracle equals the registered oracle (r2lib.Oracle) on random clustered data with tied scores and on one
  real surface (AASIST, LA19 eval).
- Surface.cv_oracle with one cluster per fold equals Surface.loso_oracle at the original sample, on random data and on
  one real surface; with fewer folds it equals a clip-by-clip transcription of its definition (extra check).
- The windowed R2-H2 statistic (run_h2cv_fast.windowed_oracles) equals the full-grid Surface.oracle / Surface.cv_oracle
  on small synthetic cases and on one real surface, and refuses a window whose search ends on its edge, for the
  in-sample oracle and for a fold's training clusters.
- r2lib.tie_interval on the reference detector's LA19 dev scores reproduces the round-1 (DEV-8 B3) interval, and the TTG
  at its two ends on ITW equals the round-1 value derived from b_thresholds.json; results/review_r1 prints the same.
- rvlib.REPO is the repository root; rvlib's SortedTarget and oracle agree with the metrics library (extra check).

Exact ties. When two distinct thresholds tie exactly in the HTER of the training clusters, which one a cross-validated
search picks depends on the last bit of 0.5 * (FRR + FAR), and the two implementations round it differently. The
synthetic comparisons therefore use data and seeds without such ties (strictly positive random cluster weights for the
windowed search); the registered integer bootstrap counts are used on the real surfaces.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis" / "round2"))
sys.path.insert(0, str(ROOT / "analysis" / "review_r1"))

import r2lib as R  # noqa: E402
from cvlib import Surface, random_folds  # noqa: E402
from deploybench.metrics import far_at, frr_at  # noqa: E402

TOL = 1e-12
ROUND1_B = ROOT / "reproducibility" / "round1" / "results" / "b_thresholds.json"
REVIEW_OUT = ROOT / "results" / "review_r1"


# --------------------------------------------------------------------------- data
def _clustered(seed, n=360, G=10, decimals=1):
    """Random clustered scores: every cluster has both classes and its own shift; rounding makes scores tie."""
    rng = np.random.default_rng(seed)
    g = np.concatenate([np.arange(G), np.arange(G), rng.integers(0, G, n - 2 * G)])
    y = np.concatenate([np.ones(G, dtype=int), np.zeros(G, dtype=int), rng.integers(0, 2, n - 2 * G)])
    s = np.where(y == 1, 1.0, -1.0) + rng.normal(0.0, 0.8, G)[g] + rng.normal(0.0, 1.2, n)
    return np.round(s, decimals), y, g


class _Real:
    def __init__(self, model, surface, seed=0):
        ids, self.y, self.s = R.load(model, surface, seed)
        self.g, names = R.clusters(surface, ids)
        self.G = len(names)
        self.sf = Surface(self.s, self.y, self.g, self.G)


@pytest.fixture(scope="module")
def aasist_la19():
    return _Real("aasist", "la19_eval")


@pytest.fixture(scope="module")
def aasist_l_run1_la19():
    return _Real("aasist_l_r2", "la19_eval", 1)


def _gap(a, b):
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


# --------------------------------------------------------------------------- 1. Surface.oracle == r2lib.Oracle
@pytest.mark.parametrize("seed", range(6))
def test_surface_oracle_equals_r2lib_oracle_on_clustered_data(seed):
    s, y, g = _clustered(seed)
    G = 10
    assert len(np.unique(s)) < len(s)                                  # tied scores are present
    sf = Surface(s, y, g, G)
    assert sf.exact
    C = R.cluster_counts(G, 60, seed)
    assert (C == 0).any()                                               # some replicates leave clusters out
    assert _gap(sf.oracle(C), R.Oracle(s, y, g).run(C)) < TOL
    ones = np.ones((1, G))                                              # the original sample
    assert _gap(sf.oracle(ones), R.Oracle(s, y, g).run(ones)) < TOL


def test_surface_oracle_equals_r2lib_oracle_on_a_real_surface(aasist_la19):
    d = aasist_la19
    assert d.sf.exact                                                   # the full grid, not a quantile grid
    C = R.cluster_counts(d.G, 50, 0)
    assert _gap(d.sf.oracle(C), R.Oracle(d.s, d.y, d.g).run(C)) < TOL
    ones = np.ones((1, d.G))
    assert _gap(d.sf.oracle(ones), R.Oracle(d.s, d.y, d.g).run(ones)) < TOL


# --------------------------------------------------------------------------- 2. cv_oracle with K = G == loso_oracle
@pytest.mark.parametrize("seed", range(6))
def test_cv_oracle_with_one_cluster_per_fold_equals_loso_on_clustered_data(seed):
    s, y, g = _clustered(100 + seed)
    G = 10
    sf = Surface(s, y, g, G)
    cv = sf.cv_oracle(np.ones((1, G)), np.arange(G), G)
    loso = sf.loso_oracle()
    for a, b in zip(cv, loso):                                          # (HTER, FRR, FAR)
        assert abs(float(a[0]) - b) < TOL


def test_cv_oracle_with_one_cluster_per_fold_equals_loso_on_a_real_surface(aasist_la19):
    d = aasist_la19
    cv = d.sf.cv_oracle(np.ones((1, d.G)), np.arange(d.G), d.G)
    loso = d.sf.loso_oracle()
    for a, b in zip(cv, loso):
        assert abs(float(a[0]) - b) < TOL


def _brute_force_cv(s, y, g, counts, folds, K):
    """Surface.cv_oracle's definition, clip by clip: draw each cluster counts[c] times, choose each fold's threshold
    on the other folds' clips (first HTER minimiser over the unique scores plus one point above the maximum), and
    pool the decisions on the held-out clips. FRR and FAR come from deploybench.metrics."""
    grid = np.concatenate([np.unique(s), [s.max() + 1.0]])
    idx = np.repeat(np.arange(len(s)), counts[g].astype(int))
    sd, yd, fd = s[idx], y[idx], folds[g[idx]]
    rej = acc = 0.0
    for k in range(K):
        tr = fd != k
        h = np.array([0.5 * (frr_at(t, sd[tr], yd[tr]) + far_at(t, sd[tr], yd[tr])) for t in grid])
        tau = grid[int(np.argmin(h))]
        te = ~tr
        rej += np.sum((yd[te] == 1) & (sd[te] < tau))
        acc += np.sum((yd[te] == 0) & (sd[te] >= tau))
    frr, far = rej / np.sum(yd == 1), acc / np.sum(yd == 0)
    return 0.5 * (frr + far), frr, far


@pytest.mark.parametrize("seed", range(3))
def test_cv_oracle_equals_its_clip_level_definition(seed):
    """Extra check beyond the plan: K < G folds and bootstrap weights, against a transcription that does not use the
    cluster-count matrices, so that cv_oracle and loso_oracle cannot share a convention error unnoticed."""
    s, y, g = _clustered(200 + seed, n=300, G=10)
    G, K = 10, 3
    sf = Surface(s, y, g, G)
    folds = random_folds(G, K, np.random.default_rng(seed))
    C = R.cluster_counts(G, 4, seed)
    C = np.vstack([np.ones((1, G)), C])                                 # the original sample and 4 replicates
    h, fr, fa = sf.cv_oracle(C, folds, K)
    for r in range(C.shape[0]):
        ref = _brute_force_cv(s, y, g, C[r], folds, K)
        assert abs(h[r] - ref[0]) < TOL and abs(fr[r] - ref[1]) < TOL and abs(fa[r] - ref[2]) < TOL


# --------------------------------------------------------------------------- 3. windowed R2-H2 statistic
def _h2():
    import run_h2cv_fast                                                # safe to import: its work is under main()
    return run_h2cv_fast


def _case(seed):
    """A small synthetic case: 12 clusters, 500 clips with ties, 300 strictly positive random cluster weights (two
    chunks of the windowed search), and a random assignment of the clusters to the script's K folds."""
    s, y, g = _clustered(300 + seed, n=500, G=12, decimals=2)
    sf = Surface(s, y, g, 12)
    C = np.random.default_rng(seed).exponential(1.0, size=(300, 12))
    folds = random_folds(12, _h2().K, np.random.default_rng(seed))
    return sf, C, folds


def _argmins(sf, C, folds, K, cols=None):
    """First HTER minimiser (a grid column, restricted to `cols` if given) per row of C, in-sample and on each fold's
    training clusters; returns (in-sample, [fold 0, ..., fold K-1]). Computed from Surface.curves, not from the
    windowed search."""
    out = []
    for W in [C] + [C * (folds != k).astype(float) for k in range(K)]:
        frr, far = sf.curves(W)
        h = 0.5 * (frr + far)
        out.append(np.argmin(h if cols is None else h[:, cols], axis=1))
    return out[0], out[1:]


def _contained(sf, C, folds, K, cols):
    """True if every full-grid minimiser lies strictly inside the window, so that the windowed search must agree."""
    inn, trn = _argmins(sf, C, folds, K)
    m = np.concatenate([inn] + trn)
    return bool(cols[0] < m.min() and m.max() < cols[-1])


def _edge_hits(sf, C, folds, K, cols):
    """(#rows whose in-sample search, #(row, fold) training searches) end on the window's first or last column."""
    inn, trn = _argmins(sf, C, folds, K, cols)
    edge = lambda j: int(np.sum((j == 0) | (j == len(cols) - 1)))   # noqa: E731
    return edge(inn), sum(edge(j) for j in trn)


@pytest.mark.parametrize("seed", range(3))
def test_windowed_statistic_equals_the_full_grid_one_on_clustered_data(seed):
    H = _h2()
    sf, C, folds = _case(seed)
    for win in (0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3):                # the narrowest window that keeps every minimiser
        cols = H.window(sf, win)[0]
        if _contained(sf, C, folds, H.K, cols):
            break
    else:
        pytest.fail("no candidate window contains every minimiser")
    assert len(cols) < len(sf.t)                                        # a real window, not the whole grid
    or_in, h_cv = H.windowed_oracles(sf, C, folds, cols)
    assert _gap(or_in, sf.oracle(C)) < TOL
    assert _gap(h_cv, sf.cv_oracle(C, folds, H.K)[0]) < TOL


def test_windowed_statistic_equals_the_full_grid_one_on_a_real_surface(aasist_l_run1_la19):
    """The configuration of run_seed (R2-H2, seed-1 run, 3-pp window, fold rng = the run seed), at B = 50."""
    H = _h2()
    d = aasist_l_run1_la19
    C = R.cluster_counts(d.G, 50, H.SEED)
    folds = random_folds(d.G, H.K, np.random.default_rng(1))
    cols = H.window(d.sf, H.WIN)[0]
    assert len(cols) < len(d.sf.t) and _contained(d.sf, C, folds, H.K, cols)
    or_in, h_cv = H.windowed_oracles(d.sf, C, folds, cols)
    assert _gap(or_in, d.sf.oracle(C)) < TOL
    assert _gap(h_cv, d.sf.cv_oracle(C, folds, H.K)[0]) < TOL


def _narrow_case(wanted):
    """The first (seed, narrow window) of the synthetic cases whose search ends on the window's edge as `wanted` says."""
    H = _h2()
    for seed in range(10):
        sf, C, folds = _case(seed)
        for win in (0.005, 0.01, 0.02, 0.03):
            cols = H.window(sf, win)[0]
            n_in, n_train = _edge_hits(sf, C, folds, H.K, cols)
            if len(cols) >= 10 and wanted(n_in, n_train):
                return sf, C, folds, cols
    pytest.fail("no synthetic case with the wanted edge hits")


def test_windowed_search_refuses_when_an_in_sample_minimiser_lands_on_the_edge():
    H = _h2()
    sf, C, folds, cols = _narrow_case(lambda n_in, n_train: n_in > 0)
    with pytest.raises(SystemExit) as e:
        H.windowed_oracles(sf, C, folds, cols)
    msg = str(e.value.code)
    assert msg.startswith("REFUSED") and "in-sample" in msg and "edge" in msg


def test_windowed_search_refuses_when_only_a_fold_training_minimiser_lands_on_the_edge():
    H = _h2()
    sf, C, folds, cols = _narrow_case(lambda n_in, n_train: n_in == 0 and n_train > 0)
    with pytest.raises(SystemExit) as e:
        H.windowed_oracles(sf, C, folds, cols)
    msg = str(e.value.code)
    assert msg.startswith("REFUSED") and "fold-" in msg and "training" in msg and "edge" in msg


# --------------------------------------------------------------------------- 4. tie_interval and its TTG endpoints
def test_tie_interval_ttg_endpoints_match_round1_for_the_reference_detector():
    """The reference (xlsr_aasist) on ITW. Round 1 stores the DEV-8 B3 interval (eer_equivalent_interval), and the
    target FRR / FAR at its two ends and the oracle HTER (targets.itw); its TTG there is HTER - oracle."""
    b3 = json.loads(ROUND1_B.read_text(encoding="utf-8"))["deterministic"]["xlsr_aasist"]
    iv, itw = b3["eer_equivalent_interval"], b3["targets"]["itw"]
    _, yd, sd = R.load("xlsr_aasist", "la19_dev")
    tau = R.taus("xlsr_aasist")["eer"]
    assert tau == iv["tau_eer"]
    ends = R.tie_interval(sd, yd, tau)
    assert ends == (iv["tau_lo"], iv["tau_hi"])
    assert ends[0] < tau < ends[1]
    _, y, s = R.load("xlsr_aasist", "itw")
    oracle = itw["oracle_hter_pct"]
    ttg_round1 = {}
    for name, t in zip(("tau_lo", "tau_hi"), ends):
        ttg_round1[name] = 0.5 * (itw[name]["frr_pct"] + itw[name]["far_pct"]) - oracle      # pp, from the file
        p = R.point(s, y, t)                                                                  # pp, recomputed
        assert 100 * p["frr"] == pytest.approx(itw[name]["frr_pct"], rel=1e-12)
        assert 100 * p["far"] == pytest.approx(itw[name]["far_pct"], rel=1e-12)
        assert 100 * p["hter_oracle"] == pytest.approx(oracle, rel=1e-12)
        assert 100 * p["ttg"] == pytest.approx(ttg_round1[name], rel=1e-9)
    assert 0 < ttg_round1["tau_lo"] < ttg_round1["tau_hi"]
    # the exploratory results print the same two values (rounded as rv_checks.txt prints them) and store them unrounded
    cell = json.loads((REVIEW_OUT / "dev.json").read_text(encoding="utf-8"))["cells"]["xlsr_aasist|s0|itw"]
    assert cell["tie"] == [ends[0], ends[1]]
    assert 100 * cell["ttg_lo"] == pytest.approx(ttg_round1["tau_lo"], rel=1e-9)
    assert 100 * cell["ttg_hi"] == pytest.approx(ttg_round1["tau_hi"], rel=1e-9)
    rows = [ln for ln in (REVIEW_OUT / "rv_checks.txt").read_text(encoding="utf-8").splitlines()
            if re.match(r"XLS-R ref \| ITW \| [\d.]+ \| [\d.]+ to [\d.]+ \|", ln)]
    assert len(rows) == 1
    assert f"| {ttg_round1['tau_lo']:.2f} to {ttg_round1['tau_hi']:.2f} |" in rows[0]


# --------------------------------------------------------------------------- 5. repository root
def test_rvlib_repo_is_the_repository_root():
    import rvlib
    assert rvlib.REPO.samefile(ROOT)
    assert (rvlib.REPO / "deploybench" / "metrics.py").is_file()
    assert (rvlib.REPO / "analysis" / "review_r1" / "rvlib.py").is_file()
    assert rvlib.OUT == rvlib.REPO / "results" / "review_r1"
    assert rvlib.R.REPO.samefile(rvlib.REPO)                            # the round-2 library agrees on the root


@pytest.mark.parametrize("seed", range(3))
def test_rvlib_target_helpers_match_the_metrics_library(seed):
    """Extra check beyond the plan: SortedTarget (FRR / FAR / HTER at many thresholds, the TTG machinery of the dev,
    ladder and small-label analyses) and rvlib.oracle agree with deploybench.metrics and r2lib.Oracle on tied scores."""
    import rvlib
    s, y, _ = _clustered(500 + seed, n=400, G=8)
    st = rvlib.SortedTarget(s, y)
    u = np.unique(s)
    for t in np.concatenate([u[::5], u[::5] + 0.05, [s.min() - 1.0, s.max() + 1.0]]):
        assert abs(st.frr(t) - frr_at(t, s, y)) < TOL and abs(st.far(t) - far_at(t, s, y)) < TOL
        assert abs(st.hter(t) - rvlib.hter(s, y, t)) < TOL
    h, tau = rvlib.oracle(s, y)
    assert abs(h - R.Oracle(s, y, np.zeros(len(y), dtype=int)).run(np.ones((1, 1)))[0]) < TOL
    assert abs(rvlib.hter(s, y, tau) - h) < TOL                         # tau attains it
    assert all(rvlib.hter(s, y, t) >= h - TOL for t in u)               # and no score threshold does better
