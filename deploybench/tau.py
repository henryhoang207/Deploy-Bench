"""Threshold calibration — the ONLY place operating thresholds are computed.

Thresholds are calibrated on SOURCE DEV data of the model's own training
cohort (methodology C3/P5) and then FROZEN. Nothing downstream may re-tune.
"""
from __future__ import annotations
import numpy as np
from .metrics import eer, det_curve


def tau_at_frr(dev_scores, dev_labels, target_frr: float) -> float:
    """Threshold with at most target_frr bona-fide rejection on dev.
    Protocol's 'FPR-x%' points (bona fide flagged as spoof) map here."""
    taus, frr, _ = det_curve(dev_scores, dev_labels)
    ok = taus[frr <= target_frr]
    return float(ok.max()) if len(ok) else float(taus[0])


def tau_at_far(dev_scores, dev_labels, target_far: float) -> float:
    taus, _, far = det_curve(dev_scores, dev_labels)
    ok = taus[far <= target_far]
    return float(ok.min()) if len(ok) else float(taus[-1])


def calibrate_tau(dev_scores, dev_labels) -> dict:
    """The three frozen operating points used everywhere in the paper."""
    _, tau_eer = eer(dev_scores, dev_labels)
    return {"eer": tau_eer,
            "frr1": tau_at_frr(dev_scores, dev_labels, 0.01),
            "frr5": tau_at_frr(dev_scores, dev_labels, 0.05)}
