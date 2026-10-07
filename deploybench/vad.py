"""Energy-VAD trim of leading and trailing non-speech: the DEV-8 F1/F2 rule, moved into the harness for round 2
(D55: the silence-trimmed AASIST-L runs trim train, dev and every target with exactly this rule).

Rule, unchanged from DEVIATIONS.md DEV-8 (implemented there in DFV_Conference/revision/analysis/f_vad_stats.py and
f_vad_score.py): 16 kHz float32 audio as returned by score.load_wav; 25 ms frames (400 samples), 10 ms hop (160),
full frames only; E_t = 10 log10(mean(x^2) + 1e-12) dB; a frame is ACTIVE if E_t >= max_t E_t - T dB with T = 35;
leading silence = start of the first active frame; trailing silence = samples after the end of the last active frame
(frame end = start + 400). Both ends are cut with zero margin. A clip shorter than one frame is not trimmed, and a clip
whose trimmed length would be under 0.25 s (4,000 samples) is left untrimmed (the DEV-8 guard).

The energy is computed in float64 from the float32 samples, so the bounds do not depend on the caller's dtype as long
as the samples are the float32 values load_wav returns.
"""
from __future__ import annotations

import numpy as np

FRAME, HOP = 400, 160
T_DB = 35
MIN_LEN = 4000          # 0.25 s at 16 kHz


def frame_energy_db(x: np.ndarray):
    """Per-frame energy in dB, or None if the clip is shorter than one frame."""
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    if n < FRAME:
        return None
    c = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    nf = 1 + (n - FRAME) // HOP
    i = np.arange(nf) * HOP
    e = (c[i + FRAME] - c[i]) / FRAME
    return 10.0 * np.log10(e + 1e-12)


def bounds(x: np.ndarray, t_db: float = T_DB):
    """(lead, trail) in samples."""
    e = frame_energy_db(x)
    if e is None:
        return 0, 0
    act = np.nonzero(e >= e.max() - t_db)[0]
    return int(act[0] * HOP), int(len(x) - (act[-1] * HOP + FRAME))


def trim(x: np.ndarray, t_db: float = T_DB):
    """(trimmed clip, status) with status in {"trimmed", "guard", "none"}. "guard" = left untrimmed because the
    result would be shorter than MIN_LEN; "none" = nothing to cut (or shorter than one frame)."""
    lead, trail = bounds(x, t_db)
    if lead + trail == 0:
        return x, "none"
    end = len(x) - trail
    if end - lead < MIN_LEN:
        return x, "guard"
    return x[lead:end], "trimmed"
