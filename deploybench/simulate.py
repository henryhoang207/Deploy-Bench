"""Synthetic score generators with analytically known metrics.

Used by tests/test_metrics.py to validate every metric against closed-form
ground truth (two-Gaussian model). Bona fide ~ N(mu_b, s), spoof ~ N(mu_s, s).

Closed forms (decide SPOOF when score < tau):
    FRR(tau) = Phi((tau - mu_b)/s)        # bona fide scoring below tau
    FAR(tau) = 1 - Phi((tau - mu_s)/s)    # spoof scoring at/above tau
    EER      = Phi(-(mu_b - mu_s)/(2*s))  # equal-variance case
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm


def gaussian_scores(n_bona: int, n_spoof: int, mu_b: float = 1.0,
                    mu_s: float = -1.0, sigma: float = 1.0,
                    seed: int = 0):
    """Return (scores, labels) with labels 1=bona fide, 0=spoof."""
    rng = np.random.default_rng(seed)
    sb = rng.normal(mu_b, sigma, n_bona)
    ss = rng.normal(mu_s, sigma, n_spoof)
    scores = np.concatenate([sb, ss])
    labels = np.concatenate([np.ones(n_bona, dtype=int), np.zeros(n_spoof, dtype=int)])
    return scores, labels


def analytic_frr(tau: float, mu_b: float, sigma: float) -> float:
    return float(norm.cdf((tau - mu_b) / sigma))


def analytic_far(tau: float, mu_s: float, sigma: float) -> float:
    return float(1.0 - norm.cdf((tau - mu_s) / sigma))


def analytic_eer(mu_b: float, mu_s: float, sigma: float) -> float:
    return float(norm.cdf(-(mu_b - mu_s) / (2.0 * sigma)))


def analytic_hter(tau: float, mu_b: float, mu_s: float, sigma: float) -> float:
    return 0.5 * (analytic_frr(tau, mu_b, sigma) + analytic_far(tau, mu_s, sigma))
