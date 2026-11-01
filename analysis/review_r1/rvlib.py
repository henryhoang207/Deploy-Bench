"""Shared helpers for the analyses prompted by the round-1 review (DEVIATIONS.md, DEV-10). Exploratory: none of them is
a registered analysis.

Reads the committed score files and frozen thresholds through the round-2 library (analysis/round2/r2lib.py), so every
convention matches the paper: label 1 = bona fide, higher score = more bona fide, spoof if score < tau. Each analysis
writes results/review_r1/<name>.json as {"meta": ..., "cells": ...}; "meta" records its resampling sizes and RNG seeds.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis" / "round2"))
sys.path.insert(0, str(REPO))

import r2lib as R  # noqa: E402
from deploybench.metrics import det_curve, eer, far_at, frr_at  # noqa: E402

OUT = REPO / "results" / "review_r1"
ROUND1 = R.ROUND1
DETECTORS_R2 = R.DETECTORS_R2
TARGETS = ("la19_eval", "itw", "df21_verified")
NAMES = {"aasist": "AASIST", "aasist_l": "AASIST-L", "rawnet2": "RawNet2", "tssdnet": "Res-TSSDNet",
         "xlsr_aasist": "XLS-R ref", "as5_aasist": "AS5 AASIST", "pascu_lr": "Pascu LR", "xlsr_mamba": "XLSR-Mamba"}


def source_dev(model: str) -> str:
    return "as5_dev" if model == "as5_aasist" else "la19_dev"


def targets_for(model: str):
    return TARGETS + (("as5_eval",) if model == "as5_aasist" else ())


def hter(s, y, tau):
    return 0.5 * (frr_at(tau, s, y) + far_at(tau, s, y))


def oracle(s, y):
    """In-sample oracle HTER (as metrics.min_hter, capped at 0.5 as r2lib.Oracle) and its threshold."""
    tg, fr, fa = det_curve(s, y)
    h = 0.5 * (fr + fa)
    i = int(np.argmin(h))
    return float(min(h[i], 0.5)), float(tg[i])


class SortedTarget:
    """Target scores sorted once, for fast FRR/FAR at many thresholds and weighted oracle searches."""

    def __init__(self, s, y, g=None):
        self.s, self.y = np.asarray(s, float), np.asarray(y, int)
        self.g = None if g is None else np.asarray(g, int)
        self.bona = np.sort(self.s[self.y == 1])
        self.spoof = np.sort(self.s[self.y == 0])

    def frr(self, tau):
        return np.searchsorted(self.bona, tau, side="left") / len(self.bona)

    def far(self, tau):
        return 1.0 - np.searchsorted(self.spoof, tau, side="left") / len(self.spoof)

    def hter(self, tau):
        return 0.5 * (self.frr(tau) + self.far(tau))


def write(name: str, meta: dict, cells: dict, summary: dict | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    obj = {"meta": meta, "cells": cells, **({"summary": summary} if summary is not None else {})}
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1), encoding="utf-8")


def read(name: str) -> dict:
    path = OUT / f"{name}.json"
    if not path.is_file():
        R.die(f"{path.relative_to(REPO)} is missing; run analysis/review_r1/run_all.py")
    return json.loads(path.read_text(encoding="utf-8"))
