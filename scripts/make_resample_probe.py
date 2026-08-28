"""Synthetic probe set for the ONE preprocessing path with no evidence: resampling.

The 1,000-clip parity manifest is entirely 16 kHz, so both sides skip resampling
and the P3 preprocessing PASS says nothing about Preprocess.kt::resampleTo16k.
That matters because the two implementations are NOT the same filter:

  scipy resample_poly : Kaiser window, half-length 10*max(up,down)
  Preprocess.kt       : Hamming window, half-length 64*max(up,down)

Same cutoff (1/max(up,down) in Nyquist units), different window and different
length -- and possibly different group-delay compensation. The Kotlin docstring
claims they match "closely enough for the decision-flip parity gate", which has
never been tested. This generates the input that tests it.

Signals are chosen to be hostile to a resampler rather than realistic: a full-band
chirp exposes passband ripple and stopband leakage across the whole spectrum, a
near-Nyquist tone probes the anti-alias transition, and noise probes broadband
behaviour. Durations straddle 96,000 samples so both the tile and truncate
branches of fixLength are exercised at every rate.

Written as PCM16 WAV, so the Kotlin and Python sides read byte-identical input
and the ONLY difference left is the resampler itself.

    python scripts/make_resample_probe.py --out-dir resample_probe \
        --manifest configs/resample_probe_manifest.txt
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

RATES = [16000, 8000, 22050, 44100, 48000]   # 16000 is the control: must be exact
DURATIONS = [3.0, 7.0]                       # < and > the 6 s / 96,000-sample input
AMP = 0.8                                    # headroom so int16 never clips


def chirp(t: np.ndarray, sr: int) -> np.ndarray:
    f0, f1 = 50.0, 0.45 * sr
    k = (f1 - f0) / t[-1]
    return np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t))


def tones(t: np.ndarray, sr: int) -> np.ndarray:
    y = np.zeros_like(t)
    for f in (110.0, 440.0, 1000.0, 3500.0, 0.40 * sr):
        y += np.sin(2 * np.pi * f * t)
    return y / 5.0


def near_nyquist(t: np.ndarray, sr: int) -> np.ndarray:
    return np.sin(2 * np.pi * (0.47 * sr) * t)


def noise(t: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(t.size) * 0.3


SIGNALS = {"chirp": chirp, "tones": tones, "nyq": near_nyquist}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--seed", type=int, default=20260727)
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    names = []
    for sr in RATES:
        for dur in DURATIONS:
            t = np.arange(int(round(sr * dur)), dtype=np.float64) / sr
            for kind, fn in SIGNALS.items():
                sig = fn(t, sr)
                names.append(_write(out_dir, f"probe_{kind}_{sr}_{dur:g}s", sig, sr))
            names.append(_write(out_dir, f"probe_noise_{sr}_{dur:g}s",
                                noise(t, sr, rng), sr))

    Path(a.manifest).write_text(
        "# synthetic resampler probe -- regenerate with scripts/make_resample_probe.py\n"
        f"# seed: {a.seed}\n" + "\n".join(names) + "\n", encoding="utf-8")

    print(f"wrote {len(names)} clips to {out_dir}")
    print(f"manifest: {a.manifest}")
    print(f"rates: {RATES}   durations: {DURATIONS}")
    return 0


def _write(out_dir: Path, name: str, sig: np.ndarray, sr: int) -> str:
    sig = np.clip(sig * AMP, -1.0, 1.0)
    # Quantise to int16 explicitly, then write int16: both sides then start from
    # exactly these samples and no float rounding difference can be blamed.
    pcm = np.round(sig * 32767.0).astype(np.int16)
    sf.write(str(out_dir / f"{name}.wav"), pcm, sr, subtype="PCM_16")
    return name


if __name__ == "__main__":
    raise SystemExit(main())
