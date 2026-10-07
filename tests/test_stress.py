"""Stress suite smoke tests with synthetic audio (runs wherever ffmpeg exists)."""
import shutil
import numpy as np
import pytest
import soundfile as sf
from pathlib import Path
from deploybench.stress import apply_condition, has_encoder, TRAIN_CONDITIONS, EVAL_CONDITIONS

ffmpeg = shutil.which("ffmpeg") is not None


@pytest.fixture
def wav(tmp_path):
    p = tmp_path / "utt1.wav"
    t = np.linspace(0, 2, 32000, endpoint=False)
    sf.write(str(p), (0.3 * np.sin(2 * np.pi * 440 * t)).astype("float32"), 16000)
    return p


@pytest.mark.skipif(not ffmpeg, reason="ffmpeg missing")
@pytest.mark.parametrize("cond", ["opus6k", "opus24k", "mp3_32", "amr4.75"])
def test_codec_roundtrip_changes_signal(wav, tmp_path, cond):
    if cond == "amr4.75" and not has_encoder("libopencore_amrnb"):
        pytest.skip("ffmpeg build lacks AMR encoder")
    dst = tmp_path / cond / "utt1.wav"
    apply_condition(wav, dst, cond)
    out, sr = sf.read(str(dst))
    assert sr == 16000 and len(out) > 16000
    ref, _ = sf.read(str(wav))
    n = min(len(ref), len(out))
    assert not np.allclose(ref[:n], out[:n], atol=1e-4)   # codec actually did something


def test_noise_hits_target_snr(wav, tmp_path):
    musan = tmp_path / "musan" / "noise" / "free"
    musan.mkdir(parents=True)
    rng = np.random.default_rng(0)
    sf.write(str(musan / "n1.wav"), rng.normal(0, 0.1, 48000).astype("float32"), 16000)
    dst = tmp_path / "snr10" / "utt1.wav"
    apply_condition(wav, dst, "snr10", musan_dir=tmp_path / "musan")
    ref, _ = sf.read(str(wav))
    out, _ = sf.read(str(dst))
    noise = out[:len(ref)] - ref
    snr = 10 * np.log10(np.mean(ref ** 2) / np.mean(noise ** 2))
    assert abs(snr - 10) < 1.0


def test_noise_is_deterministic(wav, tmp_path):
    musan = tmp_path / "musan" / "noise" / "free"
    musan.mkdir(parents=True)
    rng = np.random.default_rng(0)
    sf.write(str(musan / "n1.wav"), rng.normal(0, 0.1, 48000).astype("float32"), 16000)
    d1, d2 = tmp_path / "a" / "u.wav", tmp_path / "b" / "u.wav"
    apply_condition(wav, d1, "snr10", musan_dir=tmp_path / "musan")
    apply_condition(wav, d2, "snr10", musan_dir=tmp_path / "musan")
    a, _ = sf.read(str(d1)); b, _ = sf.read(str(d2))
    assert np.array_equal(a, b)


def test_train_conditions_barred_from_eval(wav, tmp_path):
    for cond in TRAIN_CONDITIONS:
        with pytest.raises(ValueError):
            apply_condition(wav, tmp_path / "x.wav", cond)


def test_ir_convolution(wav, tmp_path):
    ird = tmp_path / "irs"
    ird.mkdir()
    ir = np.zeros(800, dtype="float32"); ir[0] = 1.0; ir[400] = 0.5   # echo
    sf.write(str(ird / "ir1.wav"), ir, 16000)
    dst = tmp_path / "mic_ir" / "utt1.wav"
    apply_condition(wav, dst, "mic_ir", ir_dir=ird)
    out, _ = sf.read(str(dst))
    ref, _ = sf.read(str(wav))
    assert not np.allclose(ref, out[:len(ref)], atol=1e-4)


def test_ir_at_another_sample_rate_is_refused(wav, tmp_path):
    """Round-2 review (2026-10-07): an IR read at 48 kHz used to be convolved as if it were 16 kHz."""
    ird = tmp_path / "irs48"
    ird.mkdir()
    ir = np.zeros(2400, dtype="float32"); ir[0] = 1.0
    sf.write(str(ird / "ir48.wav"), ir, 48000)
    with pytest.raises(ValueError, match="IR sampled at 48000"):
        apply_condition(wav, tmp_path / "mic_ir" / "utt1.wav", "mic_ir", ir_dir=ird)
