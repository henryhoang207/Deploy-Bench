"""Cost-weighted robustness layer (analysis/operating_points.py) on planted data."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))
import operating_points as op  # noqa: E402
from deploybench.metrics import det_curve, ttg  # noqa: E402


def _scores(seed=3, n=2000, shift=1.5):
    rng = np.random.default_rng(seed)
    s = np.concatenate([rng.normal(shift, 1.0, n), rng.normal(-shift, 1.0, n)])
    return s, np.array([1] * n + [0] * n)


def test_normalisers_are_the_registered_trivial_costs():
    # screening: min(0.95 * 1, 0.05 * 1); forensic: min(0.5 * 1, 0.5 * 10)
    assert abs(op.cost_norm("screening") - 0.05) < 1e-12    # (1 - 0.95) is not exactly 0.05
    assert abs(op.cost_norm("forensic") - 0.5) < 1e-12


def test_trivial_systems_cost_exactly_one_or_more():
    s, y = _scores()
    curve = det_curve(s, y)
    for tau in (s.min() - 1.0, s.max() + 1.0):        # accept all / reject all
        c = op.cell(s, y, tau, curve)
        for k in op.COSTS:
            assert c["costs"][k]["c_norm_fixed"] >= 1.0 - 1e-12
            assert c["costs"][k]["c_norm_min"] <= 1.0 + 1e-12   # minimum includes both
            assert c["costs"][k]["ttg_cost"] >= 0.0


def test_perfect_detector_has_zero_minimum_cost():
    y = np.array([1] * 50 + [0] * 50)
    s = np.where(y == 1, 1.0, -1.0)
    c = op.cell(s, y, 0.0, det_curve(s, y))
    for k in op.COSTS:
        assert c["costs"][k]["c_norm_min"] == 0.0
        assert c["costs"][k]["c_norm_fixed"] == 0.0


def test_hter_ttg_matches_metrics_module():
    s, y = _scores()
    for tau in (-0.5, 0.0, 0.7):
        c = op.cell(s, y, tau, det_curve(s, y))
        r = ttg(s, y, tau)
        assert abs(c["ttg"] - r["ttg"]) < 1e-12
        assert abs(c["hter_oracle"] - r["hter_oracle"]) < 1e-12
        assert abs(c["frr"] - r["frr_fixed"]) < 1e-12 and abs(c["far"] - r["far_fixed"]) < 1e-12


def test_screening_punishes_false_rejection_and_forensic_false_acceptance():
    s, y = _scores()
    curve = det_curve(s, y)
    strict = op.cell(s, y, 1.0, curve)     # rejects more genuine speech
    lax = op.cell(s, y, -1.0, curve)       # accepts more spoofs
    assert strict["costs"]["screening"]["c_norm_fixed"] > lax["costs"]["screening"]["c_norm_fixed"]
    assert lax["costs"]["forensic"]["c_norm_fixed"] > strict["costs"]["forensic"]["c_norm_fixed"]
