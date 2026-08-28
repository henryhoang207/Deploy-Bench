"""Stage P3 preprocessing gate: Kotlin (app) vs Python (reference), desktop-only.

The Kotlin side is produced by android/harness/PreprocParityMain.kt running the
REAL app sources (Preprocess.kt + WavIo.kt) over PCM16 WAV copies of the parity
manifest. The Python side is score.py's own load_wav + fix_length reading the
ORIGINAL FLAC. Two different decoders, two different resamplers, one contract:
after fixLength the two waveforms must be the same audio.

Reading the original FLAC on this side is deliberate. If both sides read the WAV
copies, a transcode defect would cancel out and go unseen; make_device_wavs.py
already proved the transcode is sample-exact, so the remaining difference is
purely the preprocessing logic under test.

GATE: methodology Part C B5 makes decision flips the primary gate and feature
drift a DIAGNOSTIC reported at 1e-3 resolution. So this script reports max|delta|
and fails only on a mismatch large enough to be a logic error rather than a
decode nuance -- default 1e-3, overridable.

    python tests/preproc_parity.py --bin device_wavs.f32 --index device_wavs_index.csv \
        --flac-dir %DATA_ROOT%/LA/ASVspoof2019_LA_eval/flac
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from score import fix_length, load_wav          # noqa: E402  the reference


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="float32 blob from the Kotlin harness")
    ap.add_argument("--index", required=True)
    ap.add_argument("--flac-dir", default=None, help="reference audio dir")
    # The corpus side is FLAC, but the synthetic resampler probe
    # (scripts/make_resample_probe.py) is WAV -- and there the WAV is the ONLY
    # source, so both sides reading it is correct: it isolates the resampler.
    ap.add_argument("--ref-ext", default="flac", choices=["flac", "wav"])
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()

    flac_dir = a.flac_dir
    if flac_dir is None:
        root = os.environ.get("DATA_ROOT")
        if not root:
            print("ERROR: pass --flac-dir or set DATA_ROOT", file=sys.stderr)
            return 2
        flac_dir = Path(root) / "LA" / "ASVspoof2019_LA_eval" / "flac"
    flac_dir = Path(flac_dir)

    rows = [ln.split(",") for ln in
            Path(a.index).read_text(encoding="utf-8-sig").strip().splitlines()[1:]]
    blob = np.memmap(a.bin, dtype="<f4", mode="r")
    print(f"clips : {len(rows)}   blob : {blob.nbytes:,} bytes   tol : {a.tol}")

    worst, n_over, max_all = [], 0, 0.0
    for utt, offset, n, _src_n, _src_sr in rows:
        off, n = int(offset) // 4, int(n)
        kot = np.asarray(blob[off:off + n], dtype=np.float64)

        py = fix_length(load_wav(flac_dir / f"{utt}.{a.ref_ext}"), n).astype(np.float64)
        d = float(np.abs(kot - py).max())
        max_all = max(max_all, d)
        if d > a.tol:
            n_over += 1
        worst.append((d, utt))

    worst.sort(reverse=True)
    print(f"\nmax|delta| over all clips : {max_all:.6e}")
    print(f"clips over tol            : {n_over} / {len(rows)}")
    print("\nworst clips:")
    for d, utt in worst[:a.top]:
        print(f"  {utt}  {d:.6e}")

    if n_over:
        print(f"\nFAIL -- {n_over} clips exceed {a.tol}. The app and the reference "
              f"are not preprocessing the same audio; fix before any device run.")
        return 1
    print("\nPASS -- Kotlin and Python preprocessing agree within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
