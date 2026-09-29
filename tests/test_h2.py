"""H2 test machinery on planted data (analysis/h2_test.py)."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))
import h2_test  # noqa: E402
from deploybench.metrics import far_at, frr_at  # noqa: E402


def _planted(frr, far, n=4000, seed=1):
    rng = np.random.default_rng(seed)
    bona = np.where(rng.random(n) < frr, -1.0, 1.0)       # -1 = rejected at tau 0
    spoof = np.where(rng.random(n) < far, 1.0, -1.0)      # +1 = accepted at tau 0
    return np.concatenate([bona, spoof]), np.array([1] * n + [0] * n)


def test_bootstrap_centres_on_the_point_estimate_and_matches_metrics():
    s, y = _planted(0.40, 0.20)
    d = h2_test.frr_far_bootstrap(s, y, 0.0)
    point = frr_at(0.0, s, y) - far_at(0.0, s, y)
    assert abs(d.mean() - point) < 0.005
    assert len(d) == h2_test.B


def test_one_sided_p_direction():
    s, y = _planted(0.40, 0.20)
    assert np.mean(h2_test.frr_far_bootstrap(s, y, 0.0) <= 0) < 0.01     # FRR > FAR
    s, y = _planted(0.20, 0.40)
    assert np.mean(h2_test.frr_far_bootstrap(s, y, 0.0) <= 0) > 0.99     # FAR > FRR


def test_bootstrap_is_deterministic():
    s, y = _planted(0.3, 0.3)
    assert np.array_equal(h2_test.frr_far_bootstrap(s, y, 0.0),
                          h2_test.frr_far_bootstrap(s, y, 0.0))
