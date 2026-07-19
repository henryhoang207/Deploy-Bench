"""Res-TSSDNet wrapper. Vendored: vendor/end-to-end-synthetic-speech-detection.
Verified against ghua-ac/end-to-end-synthetic-speech-detection@main (Jul 2026 —
repo moved from ghuawhu/, old URL redirects): models.py class SSDNet1D; checkpoint
dict key "model_state_dict"; LABELS: bonafide=0, spoof=1 (data.label_encode) — the
OPPOSITE of AASIST. Authors eval on 6-s clips (96000 samples), repeat-tile short /
truncate-from-start long, softmax genuine prob as score (test.py).
Note: pretrained eval EER differs by OS (soundfile flac bug): Linux 1.6590%."""
import numpy as np
from .base import VENDOR, load_vendor_module


class _Wrap:
    def __init__(self, model, torch):
        self.model, self.torch = model, torch

    def score_batch(self, wave_batch):
        t = self.torch
        with t.no_grad():
            x = t.tensor(np.asarray(wave_batch, dtype=np.float32)).unsqueeze(1)  # [B,1,T]
            out = self.model(x)                          # [B, 2], index 0 = bona fide!
            # Authors score with softmax(out)[:, 0] (genuine prob). The logit
            # difference out[:,0]-out[:,1] = logit of that prob — a strictly
            # monotone transform, so EER is identical to the authors' score.
            return (out[:, 0] - out[:, 1]).cpu().numpy()


def load(spec, checkpoint, device):
    import torch
    # spec-load: vendor file models.py would clash with our `models` package.
    mod = load_vendor_module(
        "vendor_tssdnet_models",
        VENDOR / "end-to-end-synthetic-speech-detection" / "models.py")
    model = mod.SSDNet1D()
    model.load_state_dict(torch.load(checkpoint, map_location=device)
                          .get("model_state_dict", torch.load(checkpoint, map_location=device)))
    model.to(device).eval()
    w = _Wrap(model, torch)

    def _export():
        import torch as t

        class ScoreNet(t.nn.Module):
            def __init__(self):
                super().__init__()
                self.m = model

            def forward(self, x):
                out = self.m(x.unsqueeze(1))
                return out[:, 0] - out[:, 1]      # index 0 = bona fide (verified)
        return ScoreNet()
    w.export_module = _export
    return w
