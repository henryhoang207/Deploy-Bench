"""Round-2 fit of the Pascu et al. logistic-regression head (D54; PREREGISTRATION_ROUND2.md).

Features come from models.wrap_pascu.load_backbone, i.e. the scoring path itself: the whole utterance through
AutoFeatureExtractor + Wav2Vec2Model('facebook/wav2vec2-xls-r-2b'), last hidden state averaged over time, float32.
Classifier as danoneata/aletheia scripts/linear_classifier.py: sklearn LogisticRegression(C=1e6, max_iter=5000,
random_state=0) on float64 features with label 1 = FAKE (the paper's text says max_iter 1,000, the code 5,000; the
code is followed). A ConvergenceWarning is recorded, not acted on.

Two fits, both registered:
  --splits train       the round-2 model: LA19 dev stays a held-out threshold source
  --splits train dev   the authors' recipe, used ONLY for the pipeline gate (LA19-eval EER vs their Table 2: 0.6%)
Features are cached per split (features_<split>.npz: utt_ids, X float32, y_fake) so the second fit reuses them.

    python train/pascu_fit.py --splits train --cache runs/pascu_lr/feats --out runs/pascu_lr/head_train.npz
    python train/pascu_fit.py --splits train dev --cache runs/pascu_lr/feats --out runs/pascu_lr/head_traindev.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def extract(split: str, cache: Path, device: str, bb_holder: dict):
    f = cache / f"features_{split}.npz"
    from deploybench.data import protocol_iter
    rows = list(protocol_iter(f"asvspoof19_{split}"))
    ids = np.array([u for u, _, _ in rows])
    if f.exists():
        z = np.load(f)
        if not np.array_equal(z["utt_ids"], ids):
            raise SystemExit(f"{f}: cached ids differ from the protocol; delete it deliberately to re-extract")
        print(f"{split}: {len(ids)} cached features", flush=True)
        return z["X"], z["y_fake"], ids
    if "bb" not in bb_holder:
        from models.wrap_pascu import load_backbone
        bb_holder["bb"] = load_backbone(device)
    bb = bb_holder["bb"]
    from score import load_wav
    X = np.empty((len(rows), bb.model.config.hidden_size), dtype=np.float32)
    y = np.empty(len(rows), dtype=np.int64)
    t0 = time.time()
    for i, (u, p, lab) in enumerate(rows):
        X[i] = bb.features(load_wav(Path(p)))
        y[i] = 1 - int(lab)                           # their label: 1 = fake
        if (i + 1) % 2000 == 0:
            el = time.time() - t0
            print(f"  {split}: {i + 1}/{len(rows)}  {el:.0f}s  {(i + 1) / el:.1f}/s", flush=True)
    cache.mkdir(parents=True, exist_ok=True)
    tmp = cache / f"features_{split}.partial.npz"
    np.savez(tmp, utt_ids=ids, X=X, y_fake=y)
    tmp.replace(f)
    return X, y, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", required=True, choices=("train", "dev"))
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.splits not in (["train"], ["train", "dev"]):
        raise SystemExit("registered fits: --splits train (the model) or --splits train dev (the pipeline gate)")
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists: refusing to overwrite a fitted head")
    from sklearn import __version__ as skv
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    holder = {}
    parts = [extract(s, Path(a.cache), a.device, holder) for s in a.splits]
    X = np.vstack([p[0] for p in parts]).astype(np.float64)
    y = np.concatenate([p[1] for p in parts])
    model = LogisticRegression(C=1e6, max_iter=5_000, random_state=0)
    t0 = time.time()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y)
    conv = [str(x.message) for x in w if issubclass(x.category, ConvergenceWarning)]
    acc = float((model.predict(X) == y).mean())
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, coef=model.coef_, intercept=model.intercept_, classes=model.classes_, n_iter=model.n_iter_)
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    log = {"splits": a.splits, "n": int(len(y)), "n_fake": int(y.sum()), "n_real": int((1 - y).sum()),
           "C": 1e6, "max_iter": 5000, "random_state": 0, "solver": model.solver, "n_iter": model.n_iter_.tolist(),
           "convergence_warnings": conv, "train_accuracy": acc, "fit_s": round(time.time() - t0, 1),
           "sklearn": skv, "head_sha256": h,
           "feature_sha256": {s: hashlib.sha256(np.ascontiguousarray(p[0]).tobytes()).hexdigest()
                              for s, p in zip(a.splits, parts)}}
    out.with_suffix(".json").write_text(json.dumps(log, indent=1))
    print("FIT_DONE " + json.dumps(log), flush=True)


if __name__ == "__main__":
    main()
