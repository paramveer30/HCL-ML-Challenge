"""Extracts 16 kHz mono WAVs from the MELD mp4 clips with ffmpeg. Safe to rerun."""
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from meldrt.config import load_config


def ffmpeg_cmd(src, dst, sample_rate, max_seconds):
    return [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-i", str(src),
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-t", str(max_seconds),
        "-f", "wav", str(dst),
    ]


def extract_one(src, dst, sample_rate, max_seconds):
    """Returns None on success, or the ffmpeg error text."""
    # write to a .part file first so a killed run never leaves a half-written WAV
    part = dst.with_suffix(".part")
    result = subprocess.run(ffmpeg_cmd(src, part, sample_rate, max_seconds), capture_output=True, text=True)
    if result.returncode != 0 or not part.exists() or part.stat().st_size <= 44:
        part.unlink(missing_ok=True)
        return result.stderr.strip() or "empty output"
    part.replace(dst)
    return None


def extract_split(name, cfg, workers):
    split = cfg["data"]["splits"][name]
    clip_dir = Path(cfg["data"]["meld_root"]) / split["clips"]
    out_dir = Path(cfg["data"]["wav_root"]) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = sorted(clip_dir.glob("dia*_utt*.mp4"))
    todo = [c for c in clips if not (out_dir / (c.stem + ".wav")).exists()]
    print(f"{name}: {len(clips)} clips, {len(clips) - len(todo)} already done")

    sr, max_s = cfg["audio"]["sample_rate"], cfg["audio"]["max_seconds"]
    with ThreadPoolExecutor(workers) as pool:
        errors = list(pool.map(lambda c: extract_one(c, out_dir / (c.stem + ".wav"), sr, max_s), todo))

    failed = [(c.name, e) for c, e in zip(todo, errors) if e is not None]
    with open(out_dir / "failed.txt", "w", encoding="utf-8") as f:
        for clip, err in failed:
            f.write(f"{clip}\t{err.splitlines()[-1]}\n")
    print(f"{name}: {len(todo) - len(failed)} extracted, {len(failed)} failed (see {out_dir / 'failed.txt'})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    cfg = load_config()
    for name in args.splits:
        extract_split(name, cfg, args.workers)
