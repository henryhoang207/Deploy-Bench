"""Parser tests on synthetic protocol files mirroring official layouts."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deploybench.data import (ASVSPOOF19_PROTOCOL_DIRS, check,  # noqa: E402
                              native_conditions, protocol_iter)


def _mk(tmp_path, rel, text):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def test_asvspoof19(tmp_path):
    _mk(tmp_path, "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
        "LA_0069 LA_D_1047731 - - bonafide\n"
        "LA_0069 LA_D_1105538 - A01 spoof\n")
    rows = list(protocol_iter("asvspoof19_dev", root=tmp_path))
    assert rows[0][0] == "LA_D_1047731" and rows[0][2] == 1
    assert rows[1][2] == 0
    assert rows[0][1].name == "LA_D_1047731.flac" and "flac" in str(rows[0][1])


@pytest.mark.parametrize("proto_dir", ASVSPOOF19_PROTOCOL_DIRS)
def test_asvspoof19_every_protocol_dir_candidate_resolves(tmp_path, proto_dir):
    """PIN of the 19LA layouts the loader currently accepts. `check()` and
    `_iter_asvspoof19` share ASVSPOOF19_PROTOCOL_DIRS, so this also guarantees
    they can never disagree. STEP 0.3: once the user's real tree is known,
    trim/extend the tuple and this test pins the resolved layout."""
    _mk(tmp_path, f"{proto_dir}/ASVspoof2019.LA.cm.dev.trl.txt",
        "LA_0069 LA_D_1047731 - - bonafide\n"
        "LA_0069 LA_D_1105538 - A01 spoof\n")
    rows = list(protocol_iter("asvspoof19_dev", root=tmp_path))
    assert [r[0] for r in rows] == ["LA_D_1047731", "LA_D_1105538"]
    assert [r[2] for r in rows] == [1, 0]
    assert rows[0][1] == (tmp_path / "LA" / "ASVspoof2019_LA_dev" / "flac"
                          / "LA_D_1047731.flac")


def test_check_reports_instead_of_raising(tmp_path, capsys):
    """--check runs BEFORE the layout is known good; it must never raise."""
    assert check(tmp_path, only={"asvspoof19_dev"}) == 1     # audio dir absent
    out = capsys.readouterr().out
    assert "MISS" in out and "protocol=MISS" in out
    for cand in ASVSPOOF19_PROTOCOL_DIRS:                    # tried paths shown
        assert str(tmp_path / cand.replace("/", "\\")) in out or cand in out


# 13-column DF21 keys lines in the official format (eval-package README).
_DF21 = (
    "LA_0023 DF_E_2000011 nocodec asvspoof A14 spoof notrim eval traditional_vocoder - - - -\n"
    "LA_0047 DF_E_2000021 low_mp3 vcc2018 - bonafide notrim eval bonafide - - - -\n"
    "LA_0023 DF_E_2000031 high_ogg asvspoof A16 spoof notrim progress traditional_vocoder - - - -\n"
    "LA_0047 DF_E_2000041 mp3m4a asvspoof - bonafide trim hidden bonafide - - - -\n"
    "LA_0023 DF_E_2000051 oggm4a vcc2020 A19 spoof notrim eval unknown - - - -\n")


def test_asvspoof21_df_eval_subset_only(tmp_path):
    """B4: published DF21 numbers pool the EVAL subset; progress/hidden trials
    must not leak into the pool. Label comes from the `label` column only."""
    _mk(tmp_path, "keys/DF/CM/trial_metadata.txt", _DF21)
    rows = list(protocol_iter("asvspoof21_df_eval", root=tmp_path))
    assert [r[0] for r in rows] == ["DF_E_2000011", "DF_E_2000021", "DF_E_2000051"]
    assert [r[2] for r in rows] == [0, 1, 0]
    assert rows[1][1] == tmp_path / "ASVspoof2021_DF_eval" / "flac" / "DF_E_2000021.flac"


def test_asvspoof21_df_native_condition_is_compr(tmp_path):
    """B4: the codec condition survives into the score file."""
    _mk(tmp_path, "keys/DF/CM/trial_metadata.txt", _DF21)
    assert native_conditions("asvspoof21_df_eval", root=tmp_path) == {
        "DF_E_2000011": "nocodec", "DF_E_2000021": "low_mp3",
        "DF_E_2000051": "oggm4a"}
    assert native_conditions("itw", root=tmp_path) is None


@pytest.mark.parametrize("bad", [
    "LA_0023 DF_E_2000011 nocodec asvspoof A14 spoof notrim eval\n",            # 8 cols
    "LA_0023 DF_E_2000011 nocodec asvspoof A14 fake notrim eval x - - - -\n",   # label
    "LA_0023 DF_E_2000011 nocodec asvspoof A14 spoof notrim test x - - - -\n",  # subset
])
def test_asvspoof21_df_malformed_lines_raise(tmp_path, bad):
    """A skipped or misparsed trial silently changes a pooled EER -- refuse."""
    _mk(tmp_path, "keys/DF/CM/trial_metadata.txt", _DF21 + bad)
    with pytest.raises(ValueError):
        list(protocol_iter("asvspoof21_df_eval", root=tmp_path))


def test_itw(tmp_path):
    _mk(tmp_path, "release_in_the_wild/meta.csv",
        "file,speaker,label\n0.wav,Alec Guinness,spoof\n1.wav,Ayn Rand,bona-fide\n")
    rows = list(protocol_iter("itw", root=tmp_path))
    assert [r[2] for r in rows] == [0, 1]
    assert rows[0][0] == "0"


def test_mlaad(tmp_path):
    _mk(tmp_path, "mlaad/meta.csv",
        "path|language|model_name\nfake/de/x1.wav|de|xtts_v2\n")
    rows = list(protocol_iter("mlaad", root=tmp_path))
    assert rows[0][2] == 0 and rows[0][0] == "x1"
