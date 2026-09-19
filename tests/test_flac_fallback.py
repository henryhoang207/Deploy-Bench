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
