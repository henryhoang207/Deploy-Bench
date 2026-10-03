"""Parser tests on synthetic protocol files mirroring official layouts."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deploybench import data as _D                             # noqa: E402
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


def _classes(d, A=(), B=()):
    """Write synthetic D34 class lists with correct sha256 headers."""
    import hashlib
    d.mkdir(parents=True, exist_ok=True)
    for cls, ids in (("A", A), ("B", B)):
        ids = sorted(ids)
        h = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        (d / f"df21_class{cls}.txt").write_text(
            f"# synthetic\n# sha256(sorted ids joined by newline): {h}\n"
            + "".join(i + "\n" for i in ids))


@pytest.fixture
def no_df21_classes(tmp_path, monkeypatch):
    _classes(tmp_path / "cls")
    monkeypatch.setattr(_D, "DF21_CLASS_DIR", tmp_path / "cls")


# 13-column DF21 keys lines in the official format (eval-package README).
_DF21 = (
    "LA_0023 DF_E_2000011 nocodec asvspoof A14 spoof notrim eval traditional_vocoder - - - -\n"
    "LA_0047 DF_E_2000021 low_mp3 vcc2018 - bonafide notrim eval bonafide - - - -\n"
    "LA_0023 DF_E_2000031 high_ogg asvspoof A16 spoof notrim progress traditional_vocoder - - - -\n"
    "LA_0047 DF_E_2000041 mp3m4a asvspoof - bonafide trim hidden bonafide - - - -\n"
    "LA_0023 DF_E_2000051 oggm4a vcc2020 A19 spoof notrim eval unknown - - - -\n")


def test_asvspoof21_df_eval_subset_only(tmp_path, no_df21_classes):
    """B4: published DF21 numbers pool the EVAL subset; progress/hidden trials
    must not leak into the pool. Label comes from the `label` column only."""
    _mk(tmp_path, "keys/DF/CM/trial_metadata.txt", _DF21)
    rows = list(protocol_iter("asvspoof21_df_eval", root=tmp_path))
    assert [r[0] for r in rows] == ["DF_E_2000011", "DF_E_2000021", "DF_E_2000051"]
    assert [r[2] for r in rows] == [0, 1, 0]
    assert rows[1][1] == tmp_path / "ASVspoof2021_DF_eval" / "flac" / "DF_E_2000021.flac"


def test_asvspoof21_df_native_condition_is_compr(tmp_path, no_df21_classes):
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
def test_asvspoof21_df_malformed_lines_raise(tmp_path, no_df21_classes, bad):
    """A skipped or misparsed trial silently changes a pooled EER -- refuse."""
    _mk(tmp_path, "keys/DF/CM/trial_metadata.txt", _DF21 + bad)
    with pytest.raises(ValueError):
        list(protocol_iter("asvspoof21_df_eval", root=tmp_path))


def test_df21_surfaces_exclude_integrity_classes(tmp_path, monkeypatch):
    """D34: the verified surface drops class B only; the clean surface drops
    A and B. Conditions follow the same surface."""
    _mk(tmp_path, "keys/DF/CM/trial_metadata.txt", _DF21)
    _classes(tmp_path / "cls", A=["DF_E_2000021"], B=["DF_E_2000051"])
    monkeypatch.setattr(_D, "DF21_CLASS_DIR", tmp_path / "cls")
    ver = [r[0] for r in protocol_iter("asvspoof21_df_eval", root=tmp_path)]
    cln = [r[0] for r in protocol_iter("asvspoof21_df_eval_clean", root=tmp_path)]
    assert ver == ["DF_E_2000011", "DF_E_2000021"]
    assert cln == ["DF_E_2000011"]
    assert native_conditions("asvspoof21_df_eval_clean", root=tmp_path) == {
        "DF_E_2000011": "nocodec"}


def test_df21_class_list_integrity_is_enforced(tmp_path, monkeypatch):
    """An edited or missing class list silently changes the eval surface."""
    _mk(tmp_path, "keys/DF/CM/trial_metadata.txt", _DF21)
    d = tmp_path / "cls"
    _classes(d, A=["DF_E_2000021"], B=["DF_E_2000051"])
    monkeypatch.setattr(_D, "DF21_CLASS_DIR", d)
    (d / "df21_classB.txt").write_text((d / "df21_classB.txt").read_text() + "DF_E_2000011\n")
    with pytest.raises(ValueError):
        list(protocol_iter("asvspoof21_df_eval", root=tmp_path))
    (d / "df21_classB.txt").unlink()
    with pytest.raises(FileNotFoundError):
        list(protocol_iter("asvspoof21_df_eval", root=tmp_path))


def test_committed_df21_class_lists():
    """The committed lists are the D34 definition: sizes, disjointness, and
    the recorded sha256 (checked inside _df21_class_ids)."""
    a, b = _D._df21_class_ids("A"), _D._df21_class_ids("B")
    assert (len(a), len(b)) == (149983, 78420)
    assert not a & b


def test_itw(tmp_path):
    _mk(tmp_path, "release_in_the_wild/meta.csv",
        "file,speaker,label\n0.wav,Alec Guinness,spoof\n1.wav,Ayn Rand,bona-fide\n")
    rows = list(protocol_iter("itw", root=tmp_path))
    assert [r[2] for r in rows] == [0, 1]
    assert rows[0][0] == "0"


@pytest.mark.parametrize("label", ["", "fake", "Bona fide"])
def test_itw_refuses_unknown_labels(tmp_path, label):
    _mk(tmp_path, "release_in_the_wild/meta.csv",
        f"file,speaker,label\n0.wav,Alec Guinness,{label}\n")
    with pytest.raises(ValueError, match="unusable row"):
        list(protocol_iter("itw", root=tmp_path))


def test_mlaad(tmp_path):
    _mk(tmp_path, "mlaad5/protocol.csv",
        "utt_id,path,label,lang,system\n"
        "mlaad__fake_de_xtts_x1,mlaad/fake/de/xtts/x1.wav,0,de,xtts\n"
        "mailabs__de_DE_b_y1,mailabs/de_DE/b/y1.wav,1,de,bonafide\n")
    rows = list(protocol_iter("mlaad", root=tmp_path))
    assert [r[2] for r in rows] == [0, 1] and rows[0][0] == "mlaad__fake_de_xtts_x1"
    assert rows[1][1] == tmp_path / "mailabs/de_DE/b/y1.wav"


@pytest.mark.parametrize("bad", ["2", "", "spoof"])
def test_mlaad_refuses_unknown_labels(tmp_path, bad):
    _mk(tmp_path, "mlaad5/protocol.csv",
        f"utt_id,path,label,lang,system\nu,mlaad/x.wav,{bad},de,s\n")
    with pytest.raises(ValueError, match="unusable row"):
        list(protocol_iter("mlaad", root=tmp_path))


def test_build_mlaad_protocol(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import build_mlaad_protocol as B
    for i in range(3):                                   # 3 de spoof clips, 2 systems
        _mk(tmp_path, f"mlaad/fake/de/sys{i % 2}/same_name.wav" if i < 2
            else "mlaad/fake/de/sys0/other.wav", "")
    for i in range(5):
        _mk(tmp_path, f"mailabs/de_DE/by_book/female/eva/book/wavs/f{i}.wav", "")
    _mk(tmp_path, "mlaad/fake/pl/sysA/z.wav", "")         # pl: 2 spoof, 1 bona fide
    _mk(tmp_path, "mlaad/fake/pl/sysA/z2.wav", "")
    _mk(tmp_path, "mailabs/pl_PL/wavs/p0.wav", "")
    _mk(tmp_path, "mailabs/pl_PL/wavs/._p0.wav", "")       # AppleDouble metadata, never audio
    _mk(tmp_path, "mlaad/fake/pl/sysA/._z.wav", "")
    rows, summ = B.build(tmp_path, tmp_path / "mlaad", tmp_path / "mailabs", langs=("de", "pl"))
    ids = [r[0] for r in rows]
    assert len(ids) == len(set(ids)) == 3 + 3 + 2 + 1      # same stem in two systems stays distinct
    assert summ["languages"]["de"] == {"spoof": 3, "bonafide": 3, "bonafide_pool": 5,
                                       "systems": 2, "imbalanced": False}
    assert summ["languages"]["pl"]["imbalanced"] is True
    again, _ = B.build(tmp_path, tmp_path / "mlaad", tmp_path / "mailabs", langs=("de", "pl"))
    assert again == rows                                  # deterministic draw
    de_only, _ = B.build(tmp_path, tmp_path / "mlaad", tmp_path / "mailabs", langs=("pl", "de"))
    assert [r for r in rows if r[3] == "de"] == [r for r in de_only if r[3] == "de"]  # order-free draw
    assert all(" " not in u and "#" not in u for u in ids)
    assert not any("._" in r[1] for r in rows)
