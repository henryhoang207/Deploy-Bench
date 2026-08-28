"""Res-TSSDNet smoke test — run this BEFORE the 71k-utterance eval sweep.

Checks, in order, the things that have historically failed SILENTLY:
  1. vendor clone present, commit SHA recorded (repo moved ghuawhu -> ghua-ac)
  2. pretrained checkpoint found (it ships INSIDE the repo, under pretrained/)
  3. spec.input_samples == 96000 (SSDNet1D hardcodes max_pool1d k=375)
  4. wrapper loads; raw model emits [B,2]; export_module() emits [B] (D13)
  5. padding contract: repeat-tile short, truncate-from-START long (D10)
  6. SCORE SIGN on real 19LA dev clips: bona fide must score HIGHER than spoof.
     TSSDNet is bonafide = class index 0 — the bug that turns EER 1.7% into
     98.3% without ever raising an exception (wrapper_audit.md finding 2).

    python scripts/smoke_tssdnet.py                        # 200 dev clips, cpu
    python scripts/smoke_tssdnet.py --n 400 --device cuda
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

VENDOR_DIR = REPO / "vendor" / "end-to-end-synthetic-speech-detection"
UPSTREAM = "https://github.com/ghua-ac/end-to-end-synthetic-speech-detection"
# Author's Res-TSSDNet checkpoint, verified present in the repo tree at commit
# 36c9705 (4,251,839 bytes). Its filename contains '%' — quote it in shells.
CKPT_GLOB = "Res_TSSDNet*.pth"

FAIL: list = []
WARN: list = []


def ok(msg):
    print(f"  [ok]   {msg}")


def bad(msg):
    print(f"  [FAIL] {msg}")
    FAIL.append(msg)


def warn(msg):
    print(f"  [warn] {msg}")
    WARN.append(msg)


def step(n, title):
    print(f"\n=== {n}. {title}")


def find_checkpoint(explicit):
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    for d in (VENDOR_DIR / "pretrained", VENDOR_DIR / "trained_models", VENDOR_DIR):
        hits = sorted(d.glob(CKPT_GLOB)) if d.exists() else []
        if hits:
            return hits[0]
    return None


def vendor_sha():
    try:
        out = subprocess.run(["git", "-C", str(VENDOR_DIR), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or f"(git failed: {out.stderr.strip()[:80]})"
    except Exception as e:                                     # noqa: BLE001
        return f"(unavailable: {type(e).__name__})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200,
                    help="dev clips to score, sampled BALANCED (n/2 per class)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--dataset", default="asvspoof19_dev")
    ap.add_argument("--out", default=None, help="write a JSON summary here")
    a = ap.parse_args()

    prov = {"upstream": UPSTREAM, "device": a.device,
            "python": sys.version.split()[0]}

    step(1, "vendor clone")
    if not VENDOR_DIR.exists():
        bad(f'missing {VENDOR_DIR}\n         git clone {UPSTREAM} "{VENDOR_DIR}"')
        print("\nSMOKE: FAIL (cannot continue)")
        return 1
    if not (VENDOR_DIR / "models.py").exists():
        bad(f"{VENDOR_DIR}/models.py absent — wrong clone?")
    prov["vendor_sha"] = vendor_sha()
    ok(f"{VENDOR_DIR}  HEAD={prov['vendor_sha']}")
    if not prov["vendor_sha"].startswith("36c9705"):
        warn("HEAD is not 36c9705 (last upstream commit, 2021-10-27) — record "
             "the SHA you used in the provenance appendix")

    step(2, "pretrained checkpoint")
    ckpt = find_checkpoint(a.checkpoint)
    if ckpt is None:
        bad(f"no {CKPT_GLOB} under {VENDOR_DIR}/pretrained — the checkpoint "
            "ships inside the repo; a shallow clone can miss it")
        print("\nSMOKE: FAIL (cannot continue)")
        return 1
    prov["checkpoint"] = str(ckpt)
    prov["checkpoint_bytes"] = ckpt.stat().st_size
    ok(f"{ckpt.name}  ({prov['checkpoint_bytes']} bytes)")
    if prov["checkpoint_bytes"] != 4_251_839:
        warn("checkpoint size != 4,251,839 bytes (the upstream Res-TSSDNet "
             "eEER-1.64% file) — confirm you have the right .pth")

    step(3, "spec")
    from models.base import SPECS
    spec = SPECS["tssdnet"]
    prov["input_samples"] = spec.input_samples
    if spec.input_samples != 96000:
        bad(f"input_samples={spec.input_samples}, must be 96000 (D11)")
    else:
        ok("input_samples = 96000 (6 s @ 16 kHz; max_pool1d k=375 = 96000/4^4)")

    step(4, "wrapper load + output shapes")
    try:
        import torch
        from models.base import load_model
        prov["torch"] = torch.__version__
        wrap = load_model("tssdnet", str(ckpt), a.device)
        ok(f"load_model ok (torch {torch.__version__})")
        with torch.no_grad():
            raw = wrap.model(torch.zeros(2, 1, spec.input_samples, device=a.device))
        if tuple(raw.shape) != (2, 2):
            bad(f"raw model output {tuple(raw.shape)}, expected (2, 2)")
        else:
            ok("raw model output [B,2]")
        em = wrap.export_module().eval()
        with torch.no_grad():
            s = em(torch.zeros(2, spec.input_samples, device=a.device))
        if tuple(s.shape) != (2,):
            bad(f"export_module output {tuple(s.shape)}, expected (2,) — D13 "
                "requires one score per clip")
        else:
            ok("export_module output [B] (single-score contract, D13)")
        # export_module must reproduce the wrapper's score exactly, else the
        # parity reference and the exported artifact measure different things.
        probe = (np.random.default_rng(0)
                 .standard_normal((2, spec.input_samples)).astype("float32") * 0.05)
        sb = wrap.score_batch(probe)
        with torch.no_grad():
            se = em(torch.tensor(probe, device=a.device)).cpu().numpy()
        d = float(np.max(np.abs(sb - se)))
        prov["score_vs_export_max_abs_delta"] = d
        (ok if d < 1e-5 else bad)(f"score_batch vs export_module max|d| = {d:.3e}")
    except Exception as e:                                     # noqa: BLE001
        bad(f"{type(e).__name__}: {e}")
        print("\nSMOKE: FAIL (cannot continue)")
        return 1

    step(5, "padding contract")
    from score import fix_length
    n = spec.input_samples
    short = np.arange(1000, dtype="float32")
    p = fix_length(short, n)
    if len(p) == n and np.array_equal(p[:1000], short) and np.array_equal(p[1000:2000], short):
        ok("short clip repeat-tiled")
    else:
        bad("short-clip padding is not repeat-tile (D10)")
    q = fix_length(np.arange(n + 5000, dtype="float32"), n)
    if len(q) == n and q[0] == 0.0 and q[-1] == n - 1:
        ok("long clip truncated FROM THE START (not centre-cropped)")
    else:
        bad("long-clip truncation is not start-truncate (D10)")

    step(6, f"score sign on {a.n} real {a.dataset} clips")
    try:
        from deploybench.data import protocol_iter, DATA_ROOT
        from score import load_wav
    except Exception as e:                                     # noqa: BLE001
        bad(f"data layer: {type(e).__name__}: {e}")
        return 1
    prov["data_root"] = str(DATA_ROOT)
    print(f"  DATA_ROOT={DATA_ROOT}")
    # BALANCED sample, not a protocol-order prefix: the 19LA dev protocol lists
    # all 2,548 bona fide rows BEFORE any spoof row, so any prefix under 2,548
    # is single-class and the sign check below cannot run at all.
    quota = {0: a.n // 2, 1: a.n - a.n // 2}
    waves, labels = [], []
    try:
        for uid, path, lab in protocol_iter(a.dataset):
            if quota.get(lab, 0) <= 0:
                if all(q <= 0 for q in quota.values()):
                    break
                continue
            quota[lab] -= 1
            waves.append(fix_length(load_wav(Path(path)), n))
            labels.append(lab)
    except Exception as e:                                     # noqa: BLE001
        bad(f"reading {a.dataset}: {type(e).__name__}: {e}\n"
            f"         run:  python -m deploybench.data --check "
            f"--datasets {a.dataset}")
        print("\nSMOKE: FAIL")
        return 1
    labels = np.asarray(labels)
    if len(waves) == 0 or labels.min() == labels.max():
        bad(f"need both classes in the {a.n}-clip balanced sample; got "
            f"{np.bincount(labels, minlength=2).tolist()} [spoof, bona] — the "
            f"{a.dataset} protocol appears to be single-class")
        print("\nSMOKE: FAIL")
        return 1
    scores = np.concatenate([wrap.score_batch(np.stack(waves[i:i + 32]))
                             for i in range(0, len(waves), 32)])
    if not np.all(np.isfinite(scores)):
        bad(f"{int((~np.isfinite(scores)).sum())} non-finite scores")
    mb, ms = float(scores[labels == 1].mean()), float(scores[labels == 0].mean())
    prov.update({"n_scored": len(scores), "n_bona": int((labels == 1).sum()),
                 "mean_score_bona": mb, "mean_score_spoof": ms})
    print(f"  n={len(scores)}  bona={int((labels == 1).sum())} "
          f"spoof={int((labels == 0).sum())}")
    print(f"  mean score  bona={mb:+.4f}   spoof={ms:+.4f}")
    if mb > ms:
        ok("bona fide score HIGHER than spoof — project sign contract holds")
    else:
        bad("SIGN INVERTED: spoof scores higher than bona fide. Do NOT 'fix' "
            "this by flipping the wrapper — check the protocol label column "
            "and the checkpoint first (wrapper_audit.md finding 2).")

    print("\n" + json.dumps(prov, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"provenance": prov, "failures": FAIL, "warnings": WARN}, indent=2))
        print(f"summary -> {a.out}")
    print(f"\nSMOKE: {'FAIL' if FAIL else 'PASS'}"
          f"   ({len(FAIL)} failures, {len(WARN)} warnings)")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
