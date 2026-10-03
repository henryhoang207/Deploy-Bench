"""score_pte.check_artifact: the .pte label and provenance guard (DEV-7). No executorch needed."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deploybench.manifests import file_sha256  # noqa: E402
from score_pte import check_artifact, sidecar_for  # noqa: E402


def _art(tmp_path, **over):
    pte = tmp_path / "tssdnet_fp32.pte"
    pte.write_bytes(b"\x00pte")
    log = {"model": "tssdnet", "precision": "fp32", "converts": True,
           "sha256": file_sha256(pte), "input_samples": 96000}
    log.update(over)
    sidecar_for(pte).write_text(json.dumps(log))
    return pte


def test_accepts_matching_sidecar(tmp_path):
    assert check_artifact(_art(tmp_path), "fp32")["model"] == "tssdnet"


def test_sidecar_name():
    assert sidecar_for(Path("x/aasist_int8_seed3.pte")).name == \
        "conversion_log_aasist_int8_seed3_executorch.json"


@pytest.mark.parametrize("over,label,msg", [
    ({}, "int8", "refusing to label"),
    ({"converts": False, "error": "boom"}, "fp32", "converts=False"),
    ({"sha256": "0" * 64}, "fp32", "sha256"),
])
def test_refusals(tmp_path, over, label, msg):
    with pytest.raises(SystemExit, match=msg):
        check_artifact(_art(tmp_path, **over), label)


def test_missing_sidecar(tmp_path):
    pte = tmp_path / "a_fp32.pte"
    pte.write_bytes(b"x")
    with pytest.raises(SystemExit, match="no sidecar"):
        check_artifact(pte, "fp32")


def test_select_et_canonical_lowest_qualifying_seed(tmp_path):
    import csv as _csv
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "convert"))
    from select_et_canonical import select
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(1000, int), np.zeros(1000, int)]

    def write(p, noise):
        p.parent.mkdir(parents=True, exist_ok=True)
        s = y * 2.0 + rng.normal(0, noise, y.size)
        with open(p, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["utt_id", "label", "score"])
            w.writerows((f"u{i}", int(l), float(v)) for i, (l, v) in enumerate(zip(y, s)))
    write(tmp_path / "diag_fp32_executorch.csv", 0.8)
    write(tmp_path / "int8_seed1" / "diag_int8_executorch.csv", 2.0)   # far outside budget
    write(tmp_path / "int8_seed3" / "diag_int8_executorch.csv", 0.8)   # within budget
    write(tmp_path / "int8_seed4" / "diag_int8_executorch.csv", 0.8)
    r = select(tmp_path)
    assert r["seeds"][1]["within_budget"] is False
    assert "missing" in r["seeds"][2]["status"]                         # reported, not replaced
    assert r["canonical_seed"] == 3
