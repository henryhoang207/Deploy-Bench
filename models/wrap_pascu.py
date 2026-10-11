"""Calibration-oriented detector of Pascu et al. (Interspeech 2024; arXiv 2309.05384, code danoneata/aletheia),
round 2 (D54): a frozen wav2vec2 XLS-R 2B with a logistic regression on mean-pooled features.

Recipe verified against danoneata/aletheia @ main (2026-10-07). That repository has no licence, so the method is
re-implemented here, not vendored:
  - audio: soundfile at 16 kHz (their data.py load_audio asserts sr == 16000); 16-bit samples are exact in float32, so
    score.load_wav gives the values their float64 read casts to float32 inside the feature extractor;
  - features: AutoFeatureExtractor + Wav2Vec2Model('facebook/wav2vec2-xls-r-2b'), the WHOLE utterance, one clip per call,
    no padding or truncation (round 2's one exception, DEV-9 amendment 5: a clip under 400 samples, the encoder's
    receptive field, is repeated to 400); feature = last_hidden_state.squeeze(0).mean(0) (scripts/extract_features.py);
  - classifier: sklearn LogisticRegression(C=1e6, max_iter=5000) with label 1 = FAKE (scripts/linear_classifier.py);
    their Table 2 fits it on LA19 train + dev (evaluate_feature_type.py SPLITS).
Round 2 refits the classifier on LA19 TRAIN ONLY (train/pascu_fit.py) so that LA19 dev stays a held-out threshold
source (D54). The head is that fit's .npz (coef, intercept): a checkpoint is REQUIRED.
Score = -(w.x + b) = the log-odds of BONA FIDE: higher = bona fide, a monotone map of their P(fake) that keeps
resolution where predict_proba saturates at C = 1e6.
Backbone: vendor/wav2vec2-xls-r-2b (HF facebook/wav2vec2-xls-r-2b @ 3b6d89d). Environment: envs/r2."""
import numpy as np

from .base import VENDOR

_BACKBONE = VENDOR / "wav2vec2-xls-r-2b"
SR = 16000
#: the XLS-R convolutional feature encoder's receptive field: a shorter clip yields no frame (DEV-9 amendment 5)
MIN_SAMPLES = 400


def _tile_to_min(w, n=MIN_SAMPLES):
    """A clip shorter than n samples repeated to exactly n (score.fix_length's tiling), else the clip itself.
    DEV-9 amendment 5: two 185-sample MLAAD-5 spoofs cannot pass the encoder whole; nothing else Pascu scores is
    that short."""
    if len(w) >= n:
        return w
    if len(w) == 0:
        raise ValueError("pascu_lr: empty clip")
    return np.tile(w, int(np.ceil(n / len(w))))[:n]


class _Wrap:
    def __init__(self, fe, model, head, torch):
        self.fe, self.model, self.torch = fe, model, torch
        self.device = next(model.parameters()).device
        self.coef = np.asarray(head["coef"], dtype=np.float64).reshape(-1)
        self.intercept = float(np.asarray(head["intercept"], dtype=np.float64).reshape(-1)[0])
        if self.coef.shape[0] != model.config.hidden_size:
            raise ValueError(f"pascu_lr: head has {self.coef.shape[0]} weights, backbone hidden size is "
                             f"{model.config.hidden_size}")

    def features(self, wave):
        """Mean-pooled last hidden state of ONE whole clip (float32, as their extract_features.py stores it)."""
        t = self.torch
        w = np.asarray(wave, dtype=np.float32)
        if w.ndim != 1:
            raise ValueError("pascu_lr scores one whole utterance at a time (1-D input)")
        if len(w) < MIN_SAMPLES:
            print(f"pascu_lr: a clip of {len(w)} samples is repeated to {MIN_SAMPLES} (DEV-9 amendment 5)",
                  flush=True)
            w = _tile_to_min(w)
        inputs = self.fe(w, sampling_rate=SR, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with t.no_grad():
            h = self.model(**inputs).last_hidden_state
        return h.squeeze(0).mean(0).cpu().numpy()

    def score_batch(self, waves):
        """waves: a list of 1-D clips of any length (score.py passes one at a time for input_samples == 0)."""
        out = np.empty(len(waves), dtype=np.float64)
        for i, w in enumerate(waves):
            f = self.features(w).astype(np.float64)
            out[i] = -(f @ self.coef + self.intercept)
        return out


def load(spec, checkpoint, device):
    import torch
    from transformers import AutoFeatureExtractor, Wav2Vec2Model
    if not checkpoint:
        raise ValueError("pascu_lr: the logistic-regression head (.npz from train/pascu_fit.py) is required")
    head = dict(np.load(checkpoint))
    fe = AutoFeatureExtractor.from_pretrained(str(_BACKBONE))
    model = Wav2Vec2Model.from_pretrained(str(_BACKBONE))
    model.to(device).eval()
    return _Wrap(fe, model, head, torch)


def load_backbone(device):
    """The feature path alone (no head), for train/pascu_fit.py: guarantees fit and scoring share one code path.
    It bypasses models.base.load_model, so it turns TF32 off itself (fp32 means fp32, as in load_model)."""
    import torch
    from transformers import AutoFeatureExtractor, Wav2Vec2Model
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    fe = AutoFeatureExtractor.from_pretrained(str(_BACKBONE))
    model = Wav2Vec2Model.from_pretrained(str(_BACKBONE))
    model.to(device).eval()
    dummy = {"coef": np.zeros(model.config.hidden_size), "intercept": np.zeros(1)}
    return _Wrap(fe, model, dummy, torch)
