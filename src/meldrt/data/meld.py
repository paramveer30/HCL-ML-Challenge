"""Loads the MELD CSVs, cleans the utterance text, and builds dialogue context."""
import re
from pathlib import Path

import pandas as pd

from meldrt.config import load_config

# Order matters: the three-character junk sequences go before the single characters.
TEXT_FIXES = {
    # curly quote, ellipsis, and long dash after UTF-8 was read as cp1252
    "\u00e2\u20ac\u2122": "'", "\u00e2\u20ac\u02dc": "'",
    "\u00e2\u20ac\u00a6": "...", "\u00e2\u20ac\u201d": "--",
    "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
    "\u2026": "...", "\u2014": "--", "\u2013": "-",
    "\x92": "'", "\x91": "'", "\x93": '"', "\x94": '"',      # cp1252 bytes read as latin-1
    "\x85": "...", "\x97": "--", "\x96": "-",
}


def fix_text(text):
    for bad, good in TEXT_FIXES.items():
        text = text.replace(bad, good)
    return re.sub(r"\s+", " ", text).strip()


def load_split(name, cfg):
    split = cfg["data"]["splits"][name]
    df = pd.read_csv(Path(cfg["data"]["meld_root"]) / split["csv"], encoding="utf-8")
    df["Utterance"] = df["Utterance"].map(fix_text)
    wav_dir = Path(cfg["data"]["wav_root"]) / name
    df["wav_path"] = [str(wav_dir / f"dia{d}_utt{u}.wav") for d, u in zip(df["Dialogue_ID"], df["Utterance_ID"])]
    return df


def build_context(df, i, k=2):
    """Previous k utterances from the same dialogue as row i, oldest first, as 'Speaker: text' lines."""
    row = df.iloc[i]
    same_dialogue = df[df["Dialogue_ID"] == row["Dialogue_ID"]]
    # only earlier lines: the live system never sees the future
    earlier = same_dialogue[same_dialogue["Utterance_ID"] < row["Utterance_ID"]]
    previous = earlier.sort_values("Utterance_ID").tail(k)
    return "\n".join(f"{speaker}: {text}" for speaker, text in zip(previous["Speaker"], previous["Utterance"]))


if __name__ == "__main__":
    # prints row counts and how many utterances have an extracted WAV
    cfg = load_config()
    for name in ["train", "dev", "test"]:
        df = load_split(name, cfg)
        have = sum(Path(p).exists() for p in df["wav_path"])
        print(f"{name}: {len(df)} utterances, {df['Dialogue_ID'].nunique()} dialogues, {len(df) - have} without a WAV")
