"""Builds the sample clips for the web page from the dev split of your MELD copy: a WAV and an mp4 per sample.

Which clips is set in configs/default.yaml under "samples". A file name starts with the true emotion of
each spoken line, so the page can compare it with what the robot says. A single line is copied as it is.
The lines of a dialogue are joined in order, with a short silence between them, so that the earlier
lines are the context for the later ones.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from meldrt.config import load_config
from meldrt.data.meld import load_split


def sample_name(emotions, clip):
    """The file name without its ending: the emotion of every line, then which clip it is."""
    return "_".join(emotions) + "_" + clip


def join_wavs(wavs, gap_seconds, sample_rate):
    """One array with the given arrays in order and gap_seconds of silence between them."""
    gap = np.zeros(int(gap_seconds * sample_rate), dtype=np.float32)
    parts = []
    for wav in wavs:
        parts += [wav, gap]
    return np.concatenate(parts[:-1])


def segment_cmd(src, dst, seconds):
    """ffmpeg call that turns one line's video into a piece exactly `seconds` long, in a format a browser plays.
    If the video is shorter than that (the silence after the line), its last picture is held."""
    return [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-i", str(src),
        "-an", "-vf", f"scale=640:360,fps=24,tpad=stop_mode=clone:stop_duration={seconds:.3f}",
        "-t", f"{seconds:.3f}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dst),
    ]


def concat_cmd(list_file, dst):
    """ffmpeg call that joins the pieces named in list_file without encoding them again."""
    return ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(dst)]


def main():
    cfg = load_config()
    s = cfg["samples"]
    sample_rate = cfg["audio"]["sample_rate"]
    clip_dir = Path(cfg["data"]["meld_root"]) / cfg["data"]["splits"]["dev"]["clips"]
    wav_dir = Path(cfg["data"]["wav_root"]) / "dev"
    out = Path(s["dir"])
    out.mkdir(parents=True, exist_ok=True)

    dev = load_split("dev", cfg)
    emotion_of = {f"dia{d}_utt{u}": e for d, u, e in zip(dev["Dialogue_ID"], dev["Utterance_ID"], dev["Emotion"])}

    for clip in s["lines"]:
        name = sample_name([emotion_of[clip]], clip)
        shutil.copy(wav_dir / f"{clip}.wav", out / f"{name}.wav")
        shutil.copy(clip_dir / f"{clip}.mp4", out / f"{name}.mp4")
        print(f"{name}: one line")

    for dialogue in s["dialogues"]:
        rows = dev[dev["Dialogue_ID"] == dialogue].sort_values("Utterance_ID")
        clips = [f"dia{dialogue}_utt{u}" for u in rows["Utterance_ID"]]
        name = sample_name(list(rows["Emotion"]), f"dia{dialogue}")
        wavs = [sf.read(wav_dir / f"{clip}.wav", dtype="float32")[0] for clip in clips]
        sf.write(out / f"{name}.wav", join_wavs(wavs, s["gap_seconds"], sample_rate), sample_rate)

        with tempfile.TemporaryDirectory() as tmp:
            pieces = []
            for i, (clip, wav) in enumerate(zip(clips, wavs)):
                # each piece lasts as long as the line's sound, plus the silence that follows it
                seconds = len(wav) / sample_rate + (s["gap_seconds"] if i < len(clips) - 1 else 0)
                piece = Path(tmp) / f"{i}.mp4"
                subprocess.run(segment_cmd(clip_dir / f"{clip}.mp4", piece, seconds), check=True)
                pieces.append(piece)
            list_file = Path(tmp) / "pieces.txt"
            list_file.write_text("".join(f"file '{piece.as_posix()}'\n" for piece in pieces), encoding="utf-8")
            subprocess.run(concat_cmd(list_file, out / f"{name}.mp4"), check=True)
        print(f"{name}: {len(clips)} lines of one dialogue")


if __name__ == "__main__":
    main()
