"""Round-2 training runs of the clovaai AASIST-L recipe (D55; PREREGISTRATION_ROUND2.md, R2-T).

A minimal, logged patch of vendor/aasist/main.py @ a04c9863 (clovaai/aasist). Everything that defines the model
is the authors' code, imported from the vendor tree through models.base.load_vendor_module: Model
(models/AASIST.py), Dataset_ASVspoof2019_train / _devNeval, pad / pad_random and genSpoof_list (data_utils.py),
create_optimizer / set_seed / seed_worker / str_to_bool (utils.py), compute_eer (evaluation.py), torchcontrib SWA,
and config/AASIST-L.conf (100 epochs, batch 24, Adam 1e-4 with per-step cosine annealing to 5e-6, weight decay
1e-4, cross-entropy weights [0.1, 0.9] for [spoof, bona fide], cudnn deterministic, no benchmark). The epoch loop,
the dev scoring, the selection rule and the final SWA swap + BN update are transcribed line for line, including
the recipe's initial best_dev_eer = 1. against a dev EER in PERCENT (calculate_tDCF_EER returns eer * 100): only
epochs with dev EER <= 1% are saved and enter SWA, and if none does, the final model is the last epoch.

Changes, all declared in PREREGISTRATION_ROUND2.md before any run:
  1. No evaluation set anywhere. The recipe scores LA19 eval whenever dev EER improves (eval_all_best) and again
     after SWA, and writes best.pth by EVAL t-DCF; here there is no eval loader and best.pth is never written.
  2. The run's model is the recipe's final model, swa.pth (the only output it selects on dev alone). The last-epoch
     weights are also saved, as last.pth, for the registered secondary comparison.
  3. Dev EER = the vendor's compute_eer on the dev scores, * 100: the number the recipe selects on. The dev t-DCF
     (needs ASV scores; the recipe only logs it) is not computed.
  4. --trim vad35: train and dev audio are trimmed by deploybench.vad.trim (the DEV-8 rule, on the float32 values)
     before pad_random / pad. Without --trim the vendor datasets are used unchanged.
  5. TF32 off (as models.base.load_model). Environment envs/r2 (torch 2.2.1+cu118).
  6. --database-path and --out-dir; a JSON log line per epoch; refuses to reuse a run directory.

    python train/aasist_r2_train.py --seed 1 --database-path $DATA_ROOT/LA --out-dir runs/aasist_l_r2_s1
    python train/aasist_r2_train.py --seed 1 --trim vad35 --database-path $DATA_ROOT/LA --out-dir runs/aasist_l_trim_s1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from models.base import VENDOR, load_vendor_module  # noqa: E402

AASIST_DIR = VENDOR / "aasist"
CONFIG = AASIST_DIR / "config" / "AASIST-L.conf"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--database-path", required=True, help="the LA directory (ASVspoof2019_LA_train/, _dev/, _cm_protocols/)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--trim", choices=("none", "vad35"), default="none")
    ap.add_argument("--epochs", type=int, default=None, help="SMOKE ONLY: fewer epochs (never for a registered run)")
    ap.add_argument("--max-steps", type=int, default=None, help="SMOKE ONLY: stop each epoch after N steps, no dev")
    a = ap.parse_args()

    out = Path(a.out_dir)
    if out.exists():
        raise SystemExit(f"{out} exists: one directory per run, refusing to reuse it")
    smoke = a.epochs is not None or a.max_steps is not None
    (out / "weights").mkdir(parents=True)

    import soundfile as sf
    import torch
    import torch.nn as nn
    from torch import Tensor
    from torch.utils.data import DataLoader
    from torchcontrib.optim import SWA

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    du = load_vendor_module("vendor_aasist_data_utils", AASIST_DIR / "data_utils.py")
    ut = load_vendor_module("vendor_aasist_utils", AASIST_DIR / "utils.py")
    ev = load_vendor_module("vendor_aasist_evaluation", AASIST_DIR / "evaluation.py")
    am = load_vendor_module("vendor_aasist_AASIST", AASIST_DIR / "models" / "AASIST.py")

    with open(CONFIG, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    if "freq_aug" not in config:
        config["freq_aug"] = "False"
    num_epochs = a.epochs if a.epochs is not None else config["num_epochs"]

    ut.set_seed(a.seed, config)                       # random, numpy, torch, cuda; cudnn flags from the config

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("GPU not detected!")
    model = am.Model(model_config).to(device)

    db = Path(a.database_path)
    prot = db / "ASVspoof2019_LA_cm_protocols"
    d_label_trn, file_train = du.genSpoof_list(dir_meta=prot / "ASVspoof2019.LA.cm.train.trn.txt",
                                               is_train=True, is_eval=False)
    d_label_dev, file_dev = du.genSpoof_list(dir_meta=prot / "ASVspoof2019.LA.cm.dev.trl.txt",
                                             is_train=False, is_eval=False)
    trn_dir, dev_dir = db / "ASVspoof2019_LA_train/", db / "ASVspoof2019_LA_dev/"

    trim_counts = {"train": {}, "dev": {}}
    if a.trim == "vad35":
        from deploybench.vad import trim as vad_trim

        class TrainSet(du.Dataset_ASVspoof2019_train):
            def __getitem__(self, index):
                key = self.list_IDs[index]
                X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
                X, _st = vad_trim(X.astype(np.float32))
                X_pad = du.pad_random(X, self.cut)
                return Tensor(X_pad), self.labels[key]

        class DevSet(du.Dataset_ASVspoof2019_devNeval):
            def __getitem__(self, index):
                key = self.list_IDs[index]
                X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
                X, _st = vad_trim(X.astype(np.float32))
                X_pad = du.pad(X, self.cut)
                return Tensor(X_pad), key

        # status counts once, outside the loaders (deterministic, so identical to what the loaders apply).
        # pad_random calls np.random.randint(x_len - 64600), which raises for a clip of exactly 64,600 samples;
        # check the trimmed lengths up front so such a clip stops the run before training, not in epoch k.
        exact = []
        for part, ids, d in (("train", file_train, trn_dir), ("dev", file_dev, dev_dir)):
            for key in ids:
                x, _ = sf.read(str(d / f"flac/{key}.flac"))
                _y, st = vad_trim(x.astype(np.float32))
                trim_counts[part][st] = trim_counts[part].get(st, 0) + 1
                if part == "train" and len(_y) == 64600:
                    exact.append(key)
        if exact:
            raise SystemExit(f"{len(exact)} trimmed train clips have exactly 64,600 samples (pad_random would "
                             f"raise): {exact[:5]}")
    else:
        TrainSet, DevSet = du.Dataset_ASVspoof2019_train, du.Dataset_ASVspoof2019_devNeval

    # loaders exactly as main.get_loader (num_workers 0, pin_memory, the seeded generator for shuffling)
    train_set = TrainSet(list_IDs=file_train, labels=d_label_trn, base_dir=trn_dir)
    gen = torch.Generator()
    gen.manual_seed(a.seed)
    trn_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, drop_last=True,
                            pin_memory=True, worker_init_fn=ut.seed_worker, generator=gen)
    dev_set = DevSet(list_IDs=file_dev, base_dir=dev_dir)
    dev_loader = DataLoader(dev_set, batch_size=config["batch_size"], shuffle=False, drop_last=False,
                            pin_memory=True)

    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = ut.create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    meta = {"seed": a.seed, "trim": a.trim, "smoke": smoke, "config": str(CONFIG), "config_sha256": _sha256(CONFIG),
            "vendor_main_sha256": _sha256(AASIST_DIR / "main.py"), "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "host": platform.node(),
            "n_train": len(file_train), "n_dev": len(file_dev), "trim_counts": trim_counts,
            "nb_params": int(sum(p.view(-1).size()[0] for p in model.parameters()))}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1))
    log = open(out / "epochs.jsonl", "a")

    def dev_scores():
        model.eval()
        keys, scores = [], []
        for batch_x, utt_id in dev_loader:
            batch_x = batch_x.to(device)
            with torch.no_grad():
                _, batch_out = model(batch_x)
                batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
            keys.extend(utt_id)
            scores.extend(batch_score.tolist())
        return keys, np.asarray(scores)

    def train_epoch():                                  # main.train_epoch, transcribed
        running_loss, num_total, ii = 0, 0.0, 0
        model.train()
        weight = torch.FloatTensor([0.1, 0.9]).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight)
        for batch_x, batch_y in trn_loader:
            batch_size = batch_x.size(0)
            num_total += batch_size
            ii += 1
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            _, batch_out = model(batch_x, Freq_aug=ut.str_to_bool(config["freq_aug"]))
            batch_loss = criterion(batch_out, batch_y)
            running_loss += batch_loss.item() * batch_size
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
                scheduler.step()
            elif scheduler is None:
                pass
            else:
                raise ValueError("scheduler error, got:{}".format(scheduler))
            if a.max_steps is not None and ii >= a.max_steps:
                break
        return running_loss / num_total

    best_dev_eer = 1.                                   # the recipe's value; dev_eer below is in PERCENT
    n_swa_update = 0
    for epoch in range(num_epochs):
        t0 = time.time()
        running_loss = train_epoch()
        rec = {"epoch": epoch, "loss": running_loss, "train_s": round(time.time() - t0, 1)}
        if a.max_steps is None:
            keys, sc = dev_scores()
            lab = np.array([d_label_dev[k] for k in keys])
            dev_eer = 100 * ev.compute_eer(sc[lab == 1], sc[lab == 0])[0]
            rec["dev_eer"] = float(dev_eer)
            if best_dev_eer >= dev_eer:
                best_dev_eer = dev_eer
                torch.save(model.state_dict(), out / "weights" / "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer))
                optimizer_swa.update_swa()
                n_swa_update += 1
                rec["swa_snapshot"] = n_swa_update
        rec["total_s"] = round(time.time() - t0, 1)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)

    torch.save(model.state_dict(), out / "weights" / "last.pth")
    if n_swa_update > 0:
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(trn_loader, model, device=device)
    torch.save(model.state_dict(), out / "weights" / "swa.pth")
    final = {"n_swa_update": n_swa_update, "best_dev_eer_pct": float(best_dev_eer),
             "swa_sha256": _sha256(out / "weights" / "swa.pth"), "last_sha256": _sha256(out / "weights" / "last.pth"),
             "final_model": "swa.pth = SWA of the dev-selected snapshots" if n_swa_update else
                            "swa.pth = last epoch (no epoch reached dev EER <= 1%, so the recipe never updated SWA)"}
    (out / "run_final.json").write_text(json.dumps(final, indent=1))
    print("RUN_DONE " + json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
