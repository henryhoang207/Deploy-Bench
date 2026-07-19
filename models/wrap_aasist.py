"""AASIST / AASIST-L wrapper. Vendored repo: vendor/aasist (clovaai/aasist).
Verified against clovaai/aasist@main (Jul 2026): models/AASIST.py class Model,
Model(config["model_config"]), forward returns (hidden, logits), eval score =
logits[:, 1] (bona fide logit — main.py produce_evaluation_file), eval padding =
repeat-tile short / truncate-from-start long (data_utils.pad, 64600)."""
import json
import numpy as np
from .base import VENDOR, load_vendor_module

_CONF = {"aasist": "AASIST.conf", "aasist_l": "AASIST-L.conf"}


class _Wrap:
    def __init__(self, model, torch):
        self.model, self.torch = model, torch

    def score_batch(self, wave_batch):
        t = self.torch
        with t.no_grad():
            x = t.tensor(np.asarray(wave_batch, dtype=np.float32))
            _, out = self.model(x)                      # repo returns (hidden, logits)
            # Authors' eval score is the RAW bona-fide logit out[:, 1] (main.py).
            # NOT out[:,1]-out[:,0]: the difference is not a monotone transform of
            # out[:,1], so it would change EER vs the published numbers.
            score = out[:, 1].cpu().numpy()
        return score


class _ExportModule:
    """Factory: single-output scoring graph for ONNX/ExecuTorch export."""

    def __new__(cls, model, torch):
        class ScoreNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.m = model

            def forward(self, x):
                _, out = self.m(x)
                return out[:, 1]                  # bona-fide logit = authors' score
        return ScoreNet()


def load(spec, checkpoint, device):
    import torch
    # spec-load: vendor package is ALSO named `models` -> clashes with ours.
    mod = load_vendor_module("vendor_aasist_AASIST",
                             VENDOR / "aasist" / "models" / "AASIST.py")
    conf = json.load(open(VENDOR / "aasist" / "config" / _CONF[spec.name]))
    model = mod.Model(conf["model_config"])
    ckpt = checkpoint or str(VENDOR / "aasist" / "models" / "weights" /
                             ("AASIST.pth" if spec.name == "aasist" else "AASIST-L.pth"))
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    w = _Wrap(model, torch)
    w.export_module = lambda: _ExportModule(model, torch)
    return w
