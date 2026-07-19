import json
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from parity import compare


def _df(scores):
    return pd.DataFrame({"utt_id": [f"u{i}" for i in range(len(scores))],
                         "score": scores})


def test_identical_passes():
    r = compare(_df([0.1, 0.9, -0.5]), _df([0.1, 0.9, -0.5]), {"eer": 0.0})
    assert r["pass"] and r["max_abs_delta"] == 0


def test_small_drift_no_flip_passes():
    r = compare(_df([0.5, -0.5]), _df([0.51, -0.49]), {"eer": 0.0})
    assert r["pass"]                       # drift without decision change is OK


def test_flip_fails():
    r = compare(_df([0.01, -0.5]), _df([-0.01, -0.5]), {"eer": 0.0})
    assert not r["pass"] and r["decision_flips"]["eer"] == 1
