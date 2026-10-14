"""Packs the trained models (and, separately, the sample clips) into zip files for a GitHub release.

A reviewer then runs the system without the dataset and without training: fetch_models.py downloads and unpacks them.
"""
import zipfile
from pathlib import Path

from meldrt.config import load_config


def trained_files(cfg):
    """Every file the live system loads that we trained or tuned ourselves, as paths relative to the repo root."""
    text = Path(cfg["text"]["checkpoint_dir"]) / "best"
    fusion = Path(cfg["fusion"]["run_dir"])
    return [
        text / "config.json",
        text / "model.safetensors",
        text / "tokenizer.json",
        text / "tokenizer_config.json",
        Path(cfg["audio_head"]["checkpoint"]),
        Path(cfg["vision_head"]["checkpoint"]),
        fusion / "alpha.json",
        fusion / "weights3.json",
        fusion / "calibration.json",
    ]


def sample_files(cfg, root="."):
    """The sample clips of the web page: every WAV and the MP4 next to it."""
    clips = Path(cfg["samples"]["dir"])
    return sorted(p.relative_to(root) for p in (Path(root) / clips).iterdir() if p.suffix in (".wav", ".mp4"))


def pack(files, zip_path, root="."):
    """Writes the files into zip_path under their paths relative to root. Raises if one is missing."""
    missing = [str(f) for f in files if not (Path(root) / f).exists()]
    if missing:
        raise FileNotFoundError("not found: " + ", ".join(missing))
    Path(zip_path).parent.mkdir(parents=True, exist_ok=True)
    # the weights barely compress, so they are stored as they are
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as z:
        for f in files:
            z.write(Path(root) / f, Path(f).as_posix())


def main():
    cfg = load_config()
    release = cfg["release"]
    for asset, files in [(release["models_asset"], trained_files(cfg)), (release["samples_asset"], sample_files(cfg))]:
        out = Path("dist") / asset
        pack(files, out)
        print(f"{out}: {len(files)} files, {out.stat().st_size / 2**20:.0f} MiB")
    print(f"\nUpload both to a release tagged {release['tag']}:")
    print(f"  gh release create {release['tag']} dist/{release['models_asset']} dist/{release['samples_asset']} "
          f"-R {release['repo']} --title \"Trained models\" --notes \"Unpacked by scripts/fetch_models.py\"")


if __name__ == "__main__":
    main()
