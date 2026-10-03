"""Score an ExecuTorch .pte artifact (D48 / DEVIATIONS.md DEV-7): the desktop
ExecuTorch leg, the counterpart of score_onnx.py.

Same preprocessing as score.py (load_wav / fix_length / read_manifest), so the only
difference from the PyTorch reference CSV is the runtime. The artifact must sit next to
its to_executorch.py sidecar: the sidecar must say converts=true, and its sha256 and
precision must match the file and --precision. Otherwise this refuses, so a label can
never disagree with the artifact.

    python score_pte.py --pte exports/et/tssdnet_fp32.pte --model tssdnet \
        --dataset asvspoof19_eval --manifest configs/parity_manifest_tssdnet.txt \
        --precision fp32 --out results/.../parity_test_fp32_executorch.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import fix_length, load_wav, read_manifest  # noqa: E402


def sidecar_for(pte: Path) -> Path:
    return pte.with_name(f"conversion_log_{pte.stem}_executorch.json")


def check_artifact(pte: Path, precision: str) -> dict:
    from deploybench.manifests import file_sha256
    side = sidecar_for(pte)
    if not side.is_file():
        raise SystemExit(f"no sidecar {side.name} next to {pte}: refusing to score an "
                         "artifact of unknown provenance")
    log = json.loads(side.read_text())
    if not log.get("converts"):
        raise SystemExit(f"{side.name} says converts={log.get('converts')}: {log.get('error')}")
    if log.get("precision") != precision:
        raise SystemExit(f"{pte.name} is {log.get('precision')}; refusing to label it {precision!r}")
    sha = file_sha256(pte)
    if sha != log.get("sha256"):
        raise SystemExit(f"{pte.name} sha256 {sha[:12]} != sidecar {str(log.get('sha256'))[:12]}")
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pte", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--precision", required=True, choices=["fp32", "fp16", "int8"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--condition", default="clean")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--progress", type=int, default=0)
    a = ap.parse_args()

    pte = Path(a.pte)
    log = check_artifact(pte, a.precision)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"REFUSING to overwrite {out}")

    import torch
    from executorch.runtime import Runtime
    from deploybench.data import native_conditions, protocol_iter
    from models.base import SPECS
    n = SPECS[a.model].input_samples
    if log.get("model") != a.model or log.get("input_samples") != n:
        raise SystemExit(f"sidecar is {log.get('model')}/{log.get('input_samples')}, "
                         f"expected {a.model}/{n}")
    native = native_conditions(a.dataset)
    method = Runtime.get().load_program(str(pte)).load_method("forward")

    def utts():
        if a.manifest:
            want = read_manifest(a.manifest)
            index = {u: (u, p, l) for u, p, l in protocol_iter(a.dataset)}
            miss = [u for u in want if u not in index]
            if miss:
                raise SystemExit(f"{len(miss)} manifest utt_ids absent from {a.dataset} "
                                 f"(first: {miss[:3]})")
            for u in want:
                yield index[u]
        else:
            yield from protocol_iter(a.dataset)

    rows, t0 = [], time.time()
    for i, (uid, path, lab) in enumerate(utts(), 1):
        x = torch.from_numpy(fix_length(load_wav(Path(path)), n)).unsqueeze(0)
        v = float(method.execute([x])[0].reshape(-1)[0])
        rows.append((uid, lab, v))
        if a.progress and i % a.progress == 0:
            el = time.time() - t0
            print(f"  {i} utts  {el:.0f}s  {i/max(el,1e-9):.1f} utt/s", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "label", "score", "model", "dataset",
                    "precision", "runtime", "condition", "seed"])
        for u, l, v in rows:
            cond = native[u] if native is not None else a.condition
            w.writerow([u, l, v, a.model, a.dataset, a.precision, "executorch", cond, a.seed])
    print(f"wrote {len(rows)} scores -> {out}")
    import score as _score
    print(f"FLAC files decoded via the MD5-verified fallback (D34): {_score.FLAC_FALLBACKS}")


if __name__ == "__main__":
    main()
