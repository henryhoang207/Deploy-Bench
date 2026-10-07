"""AASIST / AASIST-L wrapper. Vendored repo: vendor/aasist (clovaai/aasist).
Verified against clovaai/aasist@main (Jul 2026): models/AASIST.py class Model,
Model(config["model_config"]), forward returns (hidden, logits), eval score =
logits[:, 1] (bona fide logit — main.py produce_evaluation_file), eval padding =
repeat-tile short / truncate-from-start long (data_utils.pad, 64600).

Round 2 (D54/D55): as5_aasist is the ASVspoof 5 baseline B02 (asvspoof-challenge/asvspoof5 @ fe23d30,
Baseline-AASIST): its models/AASIST.py is byte-identical to clovaai's, its config's model_config equals AASIST.conf's,
bonafide = 1 (data_utils.genSpoof_list) and the eval score is batch_out[:, 1] (main.py), so the same _Wrap applies;
it is loaded from its own vendor dir with the released models/weights/AASIST/best.pth. aasist_l_r2 / aasist_l_trim are
our training runs of the clovaai AASIST-L recipe (train/aasist_r2_train.py): one file per run, so a checkpoint is
REQUIRED (no default; refusing beats silently scoring the released AASIST-L under a run's name)."""
import json
import numpy as np
from .base import VENDOR, load_vendor_module

# name -> (vendor dir, config, default checkpoint; None = a checkpoint must be given)
_CONF = {
    "aasist":        ("aasist", "config/AASIST.conf", "models/weights/AASIST.pth"),
    "aasist_l":      ("aasist", "config/AASIST-L.conf", "models/weights/AASIST-L.pth"),
    "as5_aasist":    ("asvspoof5/Baseline-AASIST", "config/AASIST_ASVspoof5.conf", "models/weights/AASIST/best.pth"),
    "aasist_l_r2":   ("aasist", "config/AASIST-L.conf", None),
    "aasist_l_trim": ("aasist", "config/AASIST-L.conf", None),
}


class _Wrap:
    def __init__(self, model, torch):
        self.model, self.torch = model, torch
        self.device = next(model.parameters()).device  # A1: batch goes where the weights are

    def score_batch(self, wave_batch):
        t = self.torch
        with t.no_grad():
            x = t.tensor(np.asarray(wave_batch, dtype=np.float32), device=self.device)
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
    vdir, conf_rel, ck_rel = _CONF[spec.name]
    base = VENDOR / vdir
    if checkpoint is None and ck_rel is None:
        raise ValueError(f"{spec.name}: a checkpoint is required (one training run per file)")
    # spec-load: vendor package is ALSO named `models` -> clashes with ours.
    alias = "vendor_aasist_AASIST" if vdir == "aasist" else "vendor_as5_AASIST"
    mod = load_vendor_module(alias, base / "models" / "AASIST.py")
    conf = json.load(open(base / conf_rel))
    model = mod.Model(conf["model_config"])
    ckpt = checkpoint or str(base / ck_rel)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    w = _Wrap(model, torch)
    w.export_module = lambda: _ExportModule(model, torch)
    return w
