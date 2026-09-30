"""H5 path: stress.py --dataset writes a labelled-by-source manifest; the scorers read it
through iter_stressed. Every label a scorer writes must describe audio it scored."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sf = pytest.importorskip("soundfile")
from deploybench.stress import MANIFEST_COLS, iter_stressed  # noqa: E402
import score_cell  # noqa: E402

UTTS = [("LA_E_0000001", "bonafide"), ("LA_E_0000002", "spoof"), ("LA_E_0000003", "spoof")]


@pytest.fixture()
def root(tmp_path):
    proto = tmp_path / "LA" / "ASVspoof2019_LA_cm_protocols"
    proto.mkdir(parents=True)
    (proto / "ASVspoof2019.LA.cm.eval.trl.txt").write_text(
        "".join(f"LA_0001 {u} - {'-' if k == 'bonafide' else 'A07'} {k}\n" for u, k in UTTS))
    flac = tmp_path / "LA" / "ASVspoof2019_LA_eval" / "flac"
    flac.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for u, _ in UTTS:
        sf.write(str(flac / f"{u}.flac"), rng.uniform(-.3, .3, 8000).astype(np.float32), 16000,
                 format="FLAC")
    return tmp_path


def _env(root):
    return dict(os.environ, DATA_ROOT=str(root), PYTHONPATH=str(REPO))


def _stress(root, out, *extra):
    return subprocess.run([sys.executable, "-m", "deploybench.stress", "--dataset",
                           "asvspoof19_eval", "--out", str(out), *extra],
                          capture_output=True, text=True, cwd=str(REPO), env=_env(root))


def test_dataset_mode_writes_a_sourced_manifest_and_provenance(root, tmp_path):
    out = tmp_path / "st"
    r = _stress(root, out, "--conditions", "clean")
    assert r.returncode == 0, r.stderr
    rows = list(csv.DictReader(open(out / "manifest.csv")))
    assert tuple(rows[0]) == MANIFEST_COLS
    assert {x["source_dataset"] for x in rows} == {"asvspoof19_eval"}
    assert all(x["path"].endswith(".flac") for x in rows)       # clean keeps the container
    prov = json.loads((out / "manifest.json").read_text())
    assert prov["n_utts"] == 3 and prov["conditions"] == ["clean"]
    again = _stress(root, out, "--conditions", "clean")
    assert again.returncode != 0 and "generated once" in again.stderr


@pytest.mark.parametrize("cond", ["opus16k", "mp3_128", "nonsense"])
def test_stress_refuses_train_only_and_unknown_conditions(root, tmp_path, cond):
    r = _stress(root, tmp_path / "st", "--conditions", cond)
    assert r.returncode != 0 and "not evaluation conditions" in r.stderr


def test_df21_is_not_a_stress_source(root, tmp_path):
    r = subprocess.run([sys.executable, "-m", "deploybench.stress", "--dataset",
                        "asvspoof21_df_eval", "--out", str(tmp_path / "x"), "--conditions", "clean"],
                       capture_output=True, text=True, cwd=str(REPO), env=_env(root))
    assert r.returncode != 0 and "native per-trial codecs" in r.stderr


def _manifest(tmp_path, rows):
    p = tmp_path / "m.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(MANIFEST_COLS)
        w.writerows(rows)
    return p


def _wav(tmp_path, u):
    p = tmp_path / f"{u}.wav"
    sf.write(str(p), np.zeros(1600, np.float32), 16000)
    return str(p)


def test_iter_stressed_joins_labels_from_the_source_protocol(root, tmp_path):
    rows = [(u, "encodec6k", _wav(tmp_path, u), "asvspoof19_eval") for u, _ in UTTS]
    got = list(iter_stressed(_manifest(tmp_path, rows), "asvspoof19_eval", "encodec6k", root))
    assert [(u, lab) for u, _p, lab in got] == [("LA_E_0000001", 1), ("LA_E_0000002", 0),
                                                ("LA_E_0000003", 0)]


@pytest.mark.parametrize("case,match", [
    ("train", "train-only"), ("source", "not 'asvspoof19_eval'"), ("dup", "duplicate"),
    ("unknown", "not in the"), ("missing", "missing"), ("absent", "no rows"),
])
def test_iter_stressed_refuses(root, tmp_path, case, match):
    u0, u1 = UTTS[0][0], UTTS[1][0]
    cond, rows = "encodec6k", [(u0, "encodec6k", _wav(tmp_path, u0), "asvspoof19_eval")]
    if case == "train":
        cond = "opus16k"
    elif case == "source":
        rows = [(u0, "encodec6k", _wav(tmp_path, u0), "itw")]
    elif case == "dup":
        rows = rows * 2
    elif case == "unknown":
        rows = [("LA_E_9999999", "encodec6k", _wav(tmp_path, "LA_E_9999999"), "asvspoof19_eval")]
    elif case == "missing":
        rows = [(u1, "encodec6k", str(tmp_path / "gone.wav"), "asvspoof19_eval")]
    elif case == "absent":
        cond = "mp3_32"
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        list(iter_stressed(_manifest(tmp_path, rows), "asvspoof19_eval", cond, root))


def test_folder_mode_manifest_cannot_be_scored(tmp_path):
    p = tmp_path / "old.csv"
    p.write_text("utt_id,condition,path\nx,encodec6k,x.wav\n")
    with pytest.raises(ValueError, match="columns"):
        list(iter_stressed(p, "asvspoof19_eval", "encodec6k"))


def test_score_py_scores_stressed_audio_with_the_stress_label(root, tmp_path):
    rows = [(u, "encodec6k", _wav(tmp_path, u), "asvspoof19_eval") for u, _ in UTTS]
    man, out = _manifest(tmp_path, rows), tmp_path / "scores.csv"
    r = subprocess.run([sys.executable, "score.py", "--model", "dummy", "--dataset",
                        "asvspoof19_eval", "--stress-manifest", str(man), "--condition",
                        "encodec6k", "--out", str(out)],
                       capture_output=True, text=True, cwd=str(REPO), env=_env(root))
    assert r.returncode == 0, r.stderr[-1500:]
    got = list(csv.DictReader(open(out)))
    assert [(x["utt_id"], x["label"], x["condition"], x["dataset"]) for x in got] == \
        [(u, "1" if k == "bonafide" else "0", "encodec6k", "asvspoof19_eval") for u, k in UTTS]


def test_score_py_still_refuses_a_stress_label_on_clean_audio(root, tmp_path):
    r = subprocess.run([sys.executable, "score.py", "--model", "dummy", "--dataset",
                        "asvspoof19_eval", "--condition", "encodec6k", "--out",
                        str(tmp_path / "x.csv")],
                       capture_output=True, text=True, cwd=str(REPO), env=_env(root))
    assert r.returncode != 0 and "--stress-manifest" in r.stderr


def test_score_cell_forwards_the_stress_manifest(tmp_path):
    man = tmp_path / "m.csv"
    man.write_text("x")
    cell = dict(model="tssdnet", precision="fp32", runtime="pytorch", dataset="asvspoof19_eval",
                condition="encodec6k", seed=0, out="x.csv", stress_manifest=str(man))
    cmd = score_cell.build_cmd(cell)
    assert cmd[cmd.index("--stress-manifest") + 1] == str(man)
    with pytest.raises(score_cell.Refused):
        score_cell.build_cmd(dict(cell, stress_manifest=str(tmp_path / "absent.csv")))
    cell.pop("stress_manifest")
    with pytest.raises(score_cell.Refused, match="stress_manifest"):
        score_cell.build_cmd(cell)
