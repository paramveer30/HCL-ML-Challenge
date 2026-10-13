from pathlib import Path

import pytest

from meldrt.data.meld import fix_text, load_split

HEADER = "Sr No.,Utterance,Speaker,Emotion,Sentiment,Dialogue_ID,Utterance_ID,Season,Episode,StartTime,EndTime\n"


@pytest.mark.parametrize("raw, clean", [
    ("You must\u2019ve had your hands full.", "You must've had your hands full."),
    ("\u2018Kay.", "'Kay."),
    ("\u201cHi\u201d", '"Hi"'),
    ("Well\u2026", "Well..."),
    ("No\u2014wait", "No--wait"),
    ("1\u20132", "1-2"),
])
def test_fix_text_replaces_curly_punctuation(raw, clean):
    assert fix_text(raw) == clean


@pytest.mark.parametrize("raw, clean", [
    ("my company\x92s transition", "my company's transition"),
    ("Well\x85", "Well..."),
    ("No\x97wait", "No--wait"),
])
def test_fix_text_replaces_cp1252_leftovers(raw, clean):
    assert fix_text(raw) == clean


@pytest.mark.parametrize("good", ["\u2019", "\u2018", "\u2026", "\u2014"])
def test_fix_text_repairs_utf8_read_as_cp1252(good):
    junk = good.encode("utf-8").decode("cp1252")
    assert fix_text(f"a{junk}b") == fix_text(f"a{good}b")


def test_fix_text_strips_and_collapses_spaces():
    assert fix_text("  Oh my   God! ") == "Oh my God!"


def test_fix_text_leaves_clean_text_alone():
    assert fix_text("Okay, fine.") == "Okay, fine."


def write_fake_meld(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    (root / "dev.csv").write_text(
        HEADER
        + "1,You must\u2019ve had your hands full.,The Interviewer,neutral,neutral,0,1,8,21,\"00:16:21,940\",\"00:16:23,442\"\n"
        + "2,\"Oh, great.\",Chandler,anger,negative,3,0,8,21,\"00:16:21,940\",\"00:16:23,442\"\n",
        encoding="utf-8",
    )
    return {
        "data": {
            "meld_root": str(root),
            "wav_root": str(tmp_path / "wav"),
            "splits": {"dev": {"csv": "dev.csv", "clips": "dev_clips"}},
        }
    }


def test_load_split_cleans_text_and_keeps_columns(tmp_path):
    df = load_split("dev", write_fake_meld(tmp_path))
    assert len(df) == 2
    assert df.loc[0, "Utterance"] == "You must've had your hands full."
    assert df.loc[1, "Utterance"] == "Oh, great."
    assert df.loc[1, "Speaker"] == "Chandler"
    assert df.loc[1, "Emotion"] == "anger"
    assert df.loc[1, "Dialogue_ID"] == 3


def test_load_split_attaches_wav_path(tmp_path):
    cfg = write_fake_meld(tmp_path)
    df = load_split("dev", cfg)
    assert Path(df.loc[0, "wav_path"]) == Path(cfg["data"]["wav_root"]) / "dev" / "dia0_utt1.wav"
    assert Path(df.loc[1, "wav_path"]) == Path(cfg["data"]["wav_root"]) / "dev" / "dia3_utt0.wav"
