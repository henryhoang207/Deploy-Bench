"""Core metrics for deployment-realistic evaluation.

Conventions (fixed across the entire project — see README):
    labels: 1 = bona fide, 0 = spoof
    score : higher = more bona-fide-like
    decision rule: SPOOF if score < tau, BONA FIDE if score >= tau
    FRR: fraction of bona fide decided spoof   (false rejection)
    FAR: fraction of spoof decided bona fide   (false acceptance)

"FPR-x%" operating points in the protocol map to FRR targets (positive class
= spoof flag; a false positive is a bona fide utterance flagged as spoof).
"""
from __future__ import annotations
import numpy as np

# ----------------------------------------------------------------- basics

def _split(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have identical shape")
    bona = scores[labels == 1]
    spoof = scores[labels == 0]
    if len(bona) == 0 or len(spoof) == 0:
        raise ValueError("need both classes present")
    return bona, spoof


def frr_at(tau: float, scores, labels) -> float:
    bona, _ = _split(scores, labels)
    return float(np.mean(bona < tau))


def far_at(tau: float, scores, labels) -> float:
    _, spoof = _split(scores, labels)
    return float(np.mean(spoof >= tau))


def hter_at(tau: float, scores, labels) -> float:
    return 0.5 * (frr_at(tau, scores, labels) + far_at(tau, scores, labels))


def weighted_cost(tau: float, scores, labels, p_bona: float = 0.5,
                  c_frr: float = 1.0, c_far: float = 1.0) -> float:
    """DCF-style expected cost under an explicit deployment model.
    screening: p_bona=0.95, c_frr high; forensic: p_bona=0.5, c_far high."""
    return float(p_bona * c_frr * frr_at(tau, scores, labels)
                 + (1.0 - p_bona) * c_far * far_at(tau, scores, labels))

# ------------------------------------------------------- curves / EER

def det_curve(scores, labels):
    """FRR and FAR as functions of threshold, over all candidate thresholds.
    Returns (taus, frr, far) with taus ascending."""
    bona, spoof = _split(scores, labels)
    taus = np.unique(np.concatenate([bona, spoof]))
    # pad so tau below min (nothing rejected) and above max (all rejected) exist
    lo = taus[0] - 1e-8
    hi = taus[-1] + 1e-8
    taus = np.concatenate([[lo], taus, [hi]])
    # FRR(tau) = P(bona < tau): fraction of bona strictly below tau
    frr = np.searchsorted(np.sort(bona), taus, side="left") / len(bona)
    # FAR(tau) = P(spoof >= tau)
    far = 1.0 - np.searchsorted(np.sort(spoof), taus, side="left") / len(spoof)
    return taus, frr, far


def eer(scores, labels):
    """Equal error rate and its threshold. Standard crossing interpolation."""
    taus, frr, far = det_curve(scores, labels)
    diff = frr - far
    idx = int(np.searchsorted(diff > 0, True))  # first tau where FRR >= FAR
    if idx == 0:
        return float(frr[0]), float(taus[0])
    # linear interpolation between idx-1 and idx
    d0, d1 = diff[idx - 1], diff[idx]
    w = 0.0 if d1 == d0 else -d0 / (d1 - d0)
    e = float((1 - w) * (frr[idx - 1] + far[idx - 1]) / 2
              + w * (frr[idx] + far[idx]) / 2)
    t = float((1 - w) * taus[idx - 1] + w * taus[idx])
    return e, t


def min_hter(scores, labels):
    """Oracle: threshold minimising HTER on THIS data, and that HTER."""
    taus, frr, far = det_curve(scores, labels)
    h = 0.5 * (frr + far)
    i = int(np.argmin(h))
    return float(h[i]), float(taus[i])


def threshold_sensitivity(scores, labels, tau_star: float, band: float = 0.01):
    """Width of the tau-interval around the oracle where HTER stays within
    `band` of its minimum, plus HTER at tau_star. Small width = knife-edge."""
    taus, frr, far = det_curve(scores, labels)
    h = 0.5 * (frr + far)
    hmin = h.min()
    ok = taus[h <= hmin + band]
    width = float(ok.max() - ok.min()) if len(ok) else 0.0
    return {"hter_at_tau_star": hter_at(tau_star, scores, labels),
            "min_hter": float(hmin), "safe_band_width": width}

# ------------------------------------------------- headline paper metrics

def ttg(target_scores, target_labels, tau_star: float) -> dict:
    """Threshold Transfer Gap: HTER at the transferred threshold minus the
    target-oracle HTER. Also reports the FRR/FAR decomposition at tau_star."""
    h_fixed = hter_at(tau_star, target_scores, target_labels)
    h_oracle, tau_oracle = min_hter(target_scores, target_labels)
    return {"hter_fixed": h_fixed, "hter_oracle": h_oracle,
            "ttg": h_fixed - h_oracle, "tau_oracle": tau_oracle,
            "frr_fixed": frr_at(tau_star, target_scores, target_labels),
            "far_fixed": far_at(tau_star, target_scores, target_labels)}


def qts(eval_scores_q, eval_labels, tau_fp32: float, tau_q: float) -> dict:
    """Quantization-induced Threshold Shift, on quantized-model scores:
    HTER using the fp32-calibrated threshold minus HTER using the threshold
    recalibrated (offline, source dev) for the quantized artifact."""
    h_old = hter_at(tau_fp32, eval_scores_q, eval_labels)
    h_new = hter_at(tau_q, eval_scores_q, eval_labels)
    return {"hter_tau_fp32": h_old, "hter_tau_quant": h_new,
            "qts": h_old - h_new, "tau_shift": tau_q - tau_fp32}

# ------------------------------------------------- selective prediction

def risk_coverage(scores, labels, confidence, tau: float, n_points: int = 51):
    """Risk-coverage curve: at coverage c, keep the c fraction most confident
    utterances, risk = mean decision error among kept. Returns dict with
    coverages, risks, and AURC (area under the curve, trapezoid)."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    conf = np.asarray(confidence, float)
    decisions = (scores >= tau).astype(int)          # 1 = bona fide
    errors = (decisions != labels).astype(float)
    order = np.argsort(-conf)                        # most confident first
    err_sorted = errors[order]
    cum_err = np.cumsum(err_sorted)
    n = len(errors)
    coverages = np.linspace(1.0 / n, 1.0, n_points)
    ks = np.maximum(1, np.ceil(coverages * n).astype(int))
    risks = cum_err[ks - 1] / ks
    aurc = float(np.trapezoid(risks, coverages))
    return {"coverages": coverages, "risks": risks, "aurc": aurc}


def margin_confidence(scores, tau: float):
    """Cheapest abstention signal: distance from the operating threshold."""
    return np.abs(np.asarray(scores, float) - tau)

# ------------------------------------------------------------ calibration

def ece(probs, labels, n_bins: int = 15) -> float:
    """Expected calibration error, equal-width bins on P(bona fide)."""
    probs = np.asarray(probs, float)
    labels = np.asarray(labels, int)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        m = (probs > edges[i]) & (probs <= edges[i + 1]) if i else \
            (probs >= edges[i]) & (probs <= edges[i + 1])
        if m.sum() == 0:
            continue
        total += (m.sum() / len(probs)) * abs(labels[m].mean() - probs[m].mean())
    return float(total)


def platt_fit(dev_scores, dev_labels, n_iter: int = 200, lr: float = 0.1):
    """Logistic (Platt) map score -> P(bona fide), fitted on SOURCE DEV ONLY.
    Simple gradient descent; returns (a, b) for sigma(a*score+b)."""
    s = np.asarray(dev_scores, float)
    y = np.asarray(dev_labels, float)
    a, b = 1.0, 0.0
    for _ in range(n_iter):
        p = 1.0 / (1.0 + np.exp(-(a * s + b)))
        ga = np.mean((p - y) * s)
        gb = np.mean(p - y)
        a -= lr * ga
        b -= lr * gb
    return a, b


def platt_apply(scores, a: float, b: float):
    return 1.0 / (1.0 + np.exp(-(a * np.asarray(scores, float) + b)))

# ------------------------------------------------------------- statistics

def bootstrap_ci(scores, labels, metric_fn, n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI, resampling utterances within each class
    (preserves class balance). metric_fn(scores, labels) -> float."""
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    ib = np.where(labels == 1)[0]
    isp = np.where(labels == 0)[0]
    vals = np.empty(n_boot)
    for k in range(n_boot):
        idx = np.concatenate([rng.choice(ib, len(ib), replace=True),
                              rng.choice(isp, len(isp), replace=True)])
        vals[k] = metric_fn(scores[idx], labels[idx])
    point = metric_fn(scores, labels)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {"point": float(point), "lo": float(lo), "hi": float(hi)}


def paired_bootstrap(scores_a, scores_b, labels, metric_fn,
                     n_boot: int = 1000, seed: int = 0):
    """P(metric_a >= metric_b) under paired utterance resampling; for
    'model A has lower HTER than B' use small p as evidence. Same utterances,
    two systems."""
    rng = np.random.default_rng(seed)
    sa = np.asarray(scores_a, float)
    sb = np.asarray(scores_b, float)
    labels = np.asarray(labels, int)
    ib = np.where(labels == 1)[0]
    isp = np.where(labels == 0)[0]
    wins = 0
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(ib, len(ib), replace=True),
                              rng.choice(isp, len(isp), replace=True)])
        if metric_fn(sa[idx], labels[idx]) >= metric_fn(sb[idx], labels[idx]):
            wins += 1
    delta = metric_fn(sa, labels) - metric_fn(sb, labels)
    return {"delta": float(delta), "p_a_ge_b": wins / n_boot}


def spearman_ci(x, y, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Spearman rho with bootstrap CI. For exploratory H1/H4 ONLY — report
    effect size + CI, no binary confirm/refute (methodology C5)."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    rho = spearmanr(x, y).statistic
    vals = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        vals[k] = spearmanr(x[idx], y[idx]).statistic
    lo, hi = np.nanquantile(vals, [alpha / 2, 1 - alpha / 2])
    return {"rho": float(rho), "lo": float(lo), "hi": float(hi), "n": n}


def holm_bonferroni(pvals: dict, alpha: float = 0.05) -> dict:
    """Holm-Bonferroni over a named family of p-values -> reject decisions."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, stop = {}, False
    for i, (name, p) in enumerate(items):
        if not stop and p <= alpha / (m - i):
            out[name] = True
        else:
            stop = True
            out[name] = False
    return out
