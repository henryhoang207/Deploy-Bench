"""Train WavLM head variants (wavlm_trunc, wavlm_mhfa) on ASVspoof2019 LA.
Backbone frozen; only head params train. Convention: class 1 = bona fide
(MUST match models/wrap_wavlm.py). 3 seeds per phase_1 step_04.

    python train/train_heads.py --variant wavlm_trunc --seed 0 \
        --out checkpoints/wavlm_trunc_s0.pth [--epochs 10]
"""
import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["wavlm_trunc", "wavlm_mhfa"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader, Dataset
    from models.base import SPECS
    from models.wrap_wavlm import _build
    from deploybench.data import protocol_iter
    from score import load_wav, fix_length

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    n = SPECS[a.variant].input_samples

    class DS(Dataset):
        def __init__(self, part):
            # D16: heads train on AS5 ONLY (tau from AS5 dev). BLOCKED on the
            # ASVspoof5 loader (deliberate gap until real tree exists) — swap
            # the dataset name below once deploybench.data supports asvspoof5_*.
            self.items = list(protocol_iter(f"asvspoof5_{part}"))

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            _, p, y = self.items[i]
            return fix_length(load_wav(Path(p)), n), y

    net = _build(a.variant, torch, a.device).to(a.device)
    head_params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.Adam(head_params, lr=a.lr)
    # authors-style class weighting (bonafide-heavy, cf. AASIST/RawNet2 WCE)
    crit = torch.nn.CrossEntropyLoss(
        weight=torch.tensor([0.1, 0.9]).to(a.device))

    tr = DataLoader(DS("train"), batch_size=a.batch, shuffle=True,
                    drop_last=True, num_workers=4)
    dv = DataLoader(DS("dev"), batch_size=a.batch, num_workers=4)

    best = 1e9
    for ep in range(a.epochs):
        net.train()
        # frozen backbone must NOT run in train mode: dropout would corrupt
        # the features the head learns from (backbone params are frozen anyway)
        net.backbone.eval()
        for x, y in tr:
            x, y = x.float().to(a.device), y.to(a.device)
            loss = crit(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = sum(crit(net(x.float().to(a.device)), y.to(a.device)).item()
                     for x, y in dv) / len(dv)
        print(f"epoch {ep}: dev_loss {vl:.4f}")
        if vl < best:
            best = vl
            head_sd = {k: v for k, v in net.state_dict().items()
                       if not k.startswith("backbone.")}
            torch.save({"head": head_sd, "variant": a.variant,
                        "seed": a.seed}, a.out)
    print(f"best dev_loss {best:.4f} -> {a.out}")


if __name__ == "__main__":
    main()
