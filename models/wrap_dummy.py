"""Deterministic dummy model: end-to-end pipeline tests without torch/checkpoints.
Score = mean absolute amplitude (arbitrary but deterministic + input-sensitive)."""
import numpy as np


class _Dummy:
    def score_batch(self, wave_batch: np.ndarray) -> np.ndarray:
        return np.tanh(np.abs(np.asarray(wave_batch, dtype=np.float32)).mean(axis=-1) * 10.0)


def load(spec, checkpoint, device):
    return _Dummy()
