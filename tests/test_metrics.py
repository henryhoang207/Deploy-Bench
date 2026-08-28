"""Validation of every metric against two-Gaussian analytic ground truth.
If these pass, the math is right; everything downstream trusts this module."""
from pathlib import Path

import numpy as np
import pytest
from deploybench import metrics as M
from deploybench import tau as T
from deploybench import simulate as S

N = 20000
TOL = 0.01  # sampling tolerance at N=20k


@pytest.fixture(scope="module")
def gauss():
    scores, labels = S.gaussian_scores(N, N, mu_b=1.0, mu_s=-1.0, sigma=1.0, seed=42)
    return scores, labels


def test_eer_matches_analytic(gauss):
    scores, labels = gauss
    e, t = M.eer(scores, labels)
    assert abs(e - S.analytic_eer(1.0, -1.0, 1.0)) < TOL      # Phi(-1)=0.1587
    assert abs(t - 0.0) < 0.05                                 # symmetric -> tau=0


def test_frr_far_hter_at_fixed_tau(gauss):
    scores, labels = gauss
    tau = 0.5
    assert abs(M.frr_at(tau, scores, labels) - S.analytic_frr(tau, 1.0, 1.0)) < TOL
    assert abs(M.far_at(tau, scores, labels) - S.analytic_far(tau, -1.0, 1.0)) < TOL
    assert abs(M.hter_at(tau, scores, labels) - S.analytic_hter(tau, 1.0, -1.0, 1.0)) < TOL


def test_tau_calibration_hits_targets(gauss):
    scores, labels = gauss
    taus = T.calibrate_tau(scores, labels)
    assert M.frr_at(taus["frr1"], scores, labels) <= 0.01 + 1e-9
    assert M.frr_at(taus["frr5"], scores, labels) <= 0.05 + 1e-9
    # frr1 must be a stricter (lower) threshold than frr5
    assert taus["frr1"] <= taus["frr5"]


def test_ttg_zero_in_domain_positive_under_shift(gauss):
    scores, labels = gauss
    _, tau_star = M.eer(scores, labels)
    r_in = M.ttg(scores, labels, tau_star)
    assert r_in["ttg"] < TOL                     # ~0 in-domain
    # shift both classes down by 1.5 (channel shift): tau* now rejects bona fide
    shifted = scores - 1.5
    r_sh = M.ttg(shifted, labels, tau_star)
    assert r_sh["ttg"] > 0.05                    # large positive gap
    assert r_sh["frr_fixed"] > r_sh["far_fixed"]  # FRR-dominant, like ITW


def test_qts_detects_score_scale_change(gauss):
    scores, labels = gauss
    _, tau_fp32 = M.eer(scores, labels)
    quant = 0.5 * scores - 0.8                   # quantization rescales+shifts
    _, tau_q = M.eer(quant, labels)
    r = M.qts(quant, labels, tau_fp32, tau_q)
    assert r["qts"] > 0.01                       # stale threshold costs HTER
    assert abs(M.hter_at(tau_q, quant, labels) - M.eer(quant, labels)[0]) < TOL


def test_risk_coverage_good_confidence_reduces_risk(gauss):
    scores, labels = gauss
    _, tau = M.eer(scores, labels)
    conf = M.margin_confidence(scores, tau)
    rc = M.risk_coverage(scores, labels, conf, tau)
    assert rc["risks"][0] < rc["risks"][-1]      # high-confidence subset cleaner
    rand = M.risk_coverage(scores, labels, np.random.default_rng(0).random(len(scores)), tau)
    assert rc["aurc"] < rand["aurc"]             # margin beats random abstention


def test_ece_platt_calibrated_is_small(gauss):
    scores, labels = gauss
    a, b = M.platt_fit(scores, labels)
    probs = M.platt_apply(scores, a, b)
    assert M.ece(probs, labels) < 0.02


def test_bootstrap_ci_covers_point(gauss):
    scores, labels = gauss
    idx = np.concatenate([np.arange(0, 2000), np.arange(N, N + 2000)])  # balanced
    r = M.bootstrap_ci(scores[idx], labels[idx],
                       lambda s, l: M.eer(s, l)[0], n_boot=200)
    assert r["lo"] <= r["point"] <= r["hi"]
    assert r["hi"] - r["lo"] < 0.05


def test_paired_bootstrap_detects_better_system(gauss):
    scores, labels = gauss
    noisy = scores + np.random.default_rng(1).normal(0, 1.0, len(scores))
    r = M.paired_bootstrap(noisy, scores, labels,
                           lambda s, l: M.eer(s, l)[0], n_boot=200)
    assert r["delta"] > 0                        # noisy system has higher EER
    assert r["p_a_ge_b"] > 0.95                  # and consistently so


def test_weighted_cost_policies_disagree(gauss):
    scores, labels = gauss
    tau = 0.5
    screening = M.weighted_cost(tau, scores, labels, p_bona=0.95, c_frr=1, c_far=1)
    forensic = M.weighted_cost(tau, scores, labels, p_bona=0.5, c_frr=1, c_far=10)
    assert screening != forensic                 # cost model matters (methodology C4)


def test_no_numpy2_only_symbols():
    """metrics.py must import and run on numpy 1.26 as well as 2.x. `trapz`
    was removed in 2.0 and `trapezoid` does not exist in 1.x, so neither name
    may appear at a bare call site (regression guard, 2026-07-25)."""
    src = Path(M.__file__).read_text(encoding="utf-8")
    assert "np.trapezoid(" not in src and "np.trapz(" not in src
    assert M._trapezoid is not None
    assert abs(M._trapezoid(np.array([1.0, 1.0]), np.array([0.0, 1.0])) - 1.0) < 1e-12


def test_holm_bonferroni():
    out = M.holm_bonferroni({"h2": 0.001, "h3": 0.02, "h5": 0.2}, alpha=0.05)
    assert out["h2"] and not out["h5"]
