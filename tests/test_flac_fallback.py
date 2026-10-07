"""D34: the FLAC fallback decoder accepts samples ONLY when they match the
file's STREAMINFO MD5. No ffmpeg needed: the decoder call is stubbed."""
import hashlib
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import score  # noqa: E402


def _flac_head(md5: bytes, sr=16000, ch=1, bps=16, total=4):
    si = (4096).to_bytes(2, "big") * 2 + bytes(6)
    x = (sr << 44) | ((ch - 1) << 41) | ((bps - 1) << 36) | total
    si += x.to_bytes(8, "big") + md5
    return b"fLaC" + bytes([0x80]) + (34).to_bytes(3, "big") + si


@pytest.fixture
def stub_ffmpeg(monkeypatch):
    raw = np.array([0, 1000, -32768, 32767], dtype="<i2").tobytes()
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg",
                        types.SimpleNamespace(get_ffmpeg_exe=lambda: "ffmpeg"))
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(stdout=raw))
    return raw


def test_fallback_accepts_md5_verified_samples(tmp_path, stub_ffmpeg):
    p = tmp_path / "a.flac"
    p.write_bytes(_flac_head(hashlib.md5(stub_ffmpeg).digest()))
    before = score.FLAC_FALLBACKS
    wave, sr = score._flac_md5_verified(p)
    assert sr == 16000 and wave.dtype == np.float32
    assert np.array_equal(wave, np.array([0, 1000, -32768, 32767]) / 32768.0)
    assert score.FLAC_FALLBACKS == before + 1


@pytest.mark.parametrize("md5", [bytes(16), b"\x01" * 16])   # unset / mismatch
def test_fallback_refuses_unverifiable_audio(tmp_path, stub_ffmpeg, md5):
    p = tmp_path / "b.flac"
    p.write_bytes(_flac_head(md5))
    with pytest.raises(ValueError):
        score._flac_md5_verified(p)


def test_fallback_refuses_non_mono16(tmp_path, stub_ffmpeg):
    p = tmp_path / "c.flac"
    p.write_bytes(_flac_head(hashlib.md5(stub_ffmpeg).digest(), ch=2))
    with pytest.raises(ValueError):
        score._flac_md5_verified(p)


# ---- DEV-9 amendment 3: native FLAC decodes are MD5-verified (libsndfile 1.2.0 flipped -32768 to +1.0) ----
sf = pytest.importorskip("soundfile")


def _write_flac(p, pcm):
    sf.write(str(p), pcm, 16000, format="FLAC", subtype="PCM_16")


def test_native_flac_is_md5_verified_and_maps_min_to_minus_one(tmp_path):
    rng = np.random.default_rng(0)
    pcm = rng.integers(-32768, 32768, 16000).astype(np.int16)
    pcm[100] = -32768
    p = tmp_path / "v.flac"
    _write_flac(p, pcm)
    before = score.FLAC_VERIFIED
    w = score.load_wav(p)
    assert score.FLAC_VERIFIED == before + 1
    assert w.dtype == np.float32 and w[100] == -1.0
    assert np.array_equal(w, pcm.astype(np.float32) / np.float32(32768.0))


def test_verified_path_is_bit_identical_to_libsndfile_float(tmp_path):
    rng = np.random.default_rng(1)
    pcm = (rng.standard_normal(40000) * 3000).astype(np.int16)
    p = tmp_path / "b.flac"
    _write_flac(p, pcm)
    ref, _ = sf.read(str(p), dtype="float32")
    assert np.array_equal(score.load_wav(p), ref)


def test_unset_md5_is_decoded_and_counted_unverified(tmp_path):
    pcm = np.arange(-2000, 2000, dtype=np.int16)
    p = tmp_path / "u.flac"
    _write_flac(p, pcm)
    b = bytearray(p.read_bytes())
    b[26:42] = bytes(16)                                   # STREAMINFO MD5 := unset
    p.write_bytes(bytes(b))
    before = score.FLAC_UNVERIFIED
    w = score.load_wav(p)
    assert score.FLAC_UNVERIFIED == before + 1 and np.array_equal(w, pcm.astype(np.float32) / 32768.0)


def test_md5_mismatch_is_refused(tmp_path, stub_ffmpeg):
    pcm = np.arange(-2000, 2000, dtype=np.int16)
    p = tmp_path / "m.flac"
    _write_flac(p, pcm)
    b = bytearray(p.read_bytes())
    b[26:42] = b"\x01" * 16                                # STREAMINFO MD5 := wrong
    p.write_bytes(bytes(b))
    with pytest.raises(ValueError, match="does not match the STREAMINFO"):
        score.load_wav(p)
