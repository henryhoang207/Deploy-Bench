"""Cluster-weighted oracle machinery on a threshold grid, so that oracle searches, cross-validated oracles and
bootstrap replicates are matrix products.

CB[c, j] = number of bona fide clips of cluster c with score < t_j; CP likewise for spoofs. For cluster weights w
(one bootstrap replicate, or a training mask), FRR(t_j) = w @ CB[:, j] / w @ nb and FAR(t_j) = 1 - w @ CP[:, j] / w @ ns,
exactly as r2lib.Oracle computes them. The grid is every unique target score plus one point above the maximum (reject
everything); t_0 = the minimum score accepts everything. A surface with more unique scores than `max_m` uses a quantile
grid instead (DF21, ASVspoof 5 eval), whose oracle can only be a hair above the exact one.
"""
from __future__ import annotations

import numpy as np


class Surface:
    def __init__(self, s, y, g, G, max_m=80_000):
        s = np.asarray(s, float)
        y = np.asarray(y, int)
        g = np.asarray(g, int)
        u = np.unique(s)
        self.exact = len(u) <= max_m
        if not self.exact:
            u = np.unique(np.quantile(s, np.linspace(0.0, 1.0, max_m)))
        self.t = np.concatenate([u, [u[-1] + 1.0]])
        M = len(self.t)
        self.G = G
        self.CB = np.zeros((G, M))
        self.CP = np.zeros((G, M))
        self.nb = np.bincount(g[y == 1], minlength=G).astype(float)
        self.ns = np.bincount(g[y == 0], minlength=G).astype(float)
        for c in range(G):
            sc = s[g == c]
            yc = y[g == c]
            self.CB[c] = np.searchsorted(np.sort(sc[yc == 1]), self.t, side="left")
            self.CP[c] = np.searchsorted(np.sort(sc[yc == 0]), self.t, side="left")

    # ----------------------------------------------------------------- curves for a weight matrix W (B x G)
    def curves(self, W):
        Nb = W @ self.nb
        Ns = W @ self.ns
        frr = (W @ self.CB) / Nb[:, None]
        far = 1.0 - (W @ self.CP) / Ns[:, None]
        return frr, far

    def oracle(self, W, chunk=100):
        """Oracle HTER (capped at 0.5) per row of W."""
        out = np.empty(W.shape[0])
        for a in range(0, W.shape[0], chunk):
            frr, far = self.curves(W[a:a + chunk])
            out[a:a + chunk] = np.minimum((0.5 * (frr + far)).min(axis=1), 0.5)
        return out

    def cv_oracle(self, W, fold, K, chunk=100):
        """Cross-validated oracle per row of W: for each fold k, the threshold that minimises HTER on the other folds
        (first minimiser on the grid) decides the clips of fold k; FRR and FAR are pooled over the out-of-fold
        decisions. `fold` gives each cluster's fold (0..K-1). Returns (hter, frr, far) per row."""
        B = W.shape[0]
        rej = np.zeros(B)
        acc = np.zeros(B)
        Nb = W @ self.nb
        Ns = W @ self.ns
        for k in range(K):
            tr = (fold != k).astype(float)
            te = 1.0 - tr
            for a in range(0, B, chunk):
                Wt = W[a:a + chunk] * tr
                We = W[a:a + chunk] * te
                frr, far = self.curves(Wt)
                j = np.argmin(0.5 * (frr + far), axis=1)
                rows = np.arange(len(j))
                rej[a:a + chunk] += (We @ self.CB)[rows, j]
                acc[a:a + chunk] += (We @ self.ns) - (We @ self.CP)[rows, j]
        frr_cv = rej / Nb
        far_cv = acc / Ns
        return 0.5 * (frr_cv + far_cv), frr_cv, far_cv

    def loso_oracle(self):
        """Leave-one-cluster-out oracle at the original sample: each cluster is decided by the threshold that minimises
        HTER on all other clusters. Returns (hter, frr, far)."""
        TB, TP = self.CB.sum(0), self.CP.sum(0)
        nbt, nst = self.nb.sum(), self.ns.sum()
        rej = acc = 0.0
        for c in range(self.G):
            frr = (TB - self.CB[c]) / (nbt - self.nb[c])
            far = 1.0 - (TP - self.CP[c]) / (nst - self.ns[c])
            j = int(np.argmin(0.5 * (frr + far)))
            rej += self.CB[c, j]
            acc += self.ns[c] - self.CP[c, j]
        frr_cv, far_cv = rej / nbt, acc / nst
        return 0.5 * (frr_cv + far_cv), frr_cv, far_cv


def random_folds(G, K, rng):
    """Random assignment of G clusters to K folds of (nearly) equal cluster counts."""
    f = np.arange(G) % K
    return rng.permutation(f)
