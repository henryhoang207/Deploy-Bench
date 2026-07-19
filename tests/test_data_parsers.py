"""Parser tests on synthetic protocol files mirroring official layouts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deploybench.data import protocol_iter  # noqa: E402


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


def test_asvspoof21_df(tmp_path):
    _mk(tmp_path, "ASVspoof2021_DF_eval/trial_metadata.txt",
        "LA_0023 DF_E_2000011 nocodec asvspoof A14 spoof notrim eval\n"
        "LA_0047 DF_E_2000021 low_mp3 vcc2018 - bonafide notrim eval\n")
    rows = list(protocol_iter("asvspoof21_df_eval", root=tmp_path))
    assert [r[2] for r in rows] == [0, 1]
    assert rows[1][0] == "DF_E_2000021"


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
