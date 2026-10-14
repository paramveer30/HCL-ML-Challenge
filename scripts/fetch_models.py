"""Downloads every model once, so the system can run with the network switched off.

Run this once after installing. It fetches the pretrained models, then the trained models and the sample clips
from the GitHub release (see the README). The reply model is fetched separately: ollama pull qwen2.5:1.5b
"""
import argparse
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

from faster_whisper.utils import download_model
from kokoro import KPipeline
from transformers import AutoFeatureExtractor, AutoModelForSequenceClassification, AutoTokenizer, ViTModel, WavLMModel

from meldrt.config import load_config
from scripts.package_models import trained_files


def download_asset(release, asset, dest):
    """Gets one release file: the plain link works for a public repository, the GitHub CLI for a private one."""
    url = f"https://github.com/{release['repo']}/releases/download/{release['tag']}/{asset}"
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except OSError:
        pass
    if shutil.which("gh"):
        done = subprocess.run(["gh", "release", "download", release["tag"], "-R", release["repo"], "-p", asset,
                               "-O", str(dest), "--clobber"])
        return done.returncode == 0
    return False


def unpack_release(cfg, asset, zip_path):
    """Unpacks one release zip into the repository, downloading it first unless a local copy is given."""
    release = cfg["release"]
    if zip_path is None:
        zip_path = Path("cache") / asset
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {asset} ...")
        if not download_asset(release, asset, zip_path):
            print(f"Could not download {asset}. Download it from the {release['tag']} release on "
                  f"https://github.com/{release['repo']}/releases and run:  "
                  f"python scripts/fetch_models.py --zip <path to {asset}>")
            return False
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(".")
        print(f"{asset}: {len(z.namelist())} files unpacked")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", nargs="*", default=[], help="release zips downloaded by hand, instead of downloading")
    args = parser.parse_args()
    cfg = load_config()

    print("Whisper:", download_model(cfg["asr"]["model"]))

    AutoFeatureExtractor.from_pretrained(cfg["audio"]["wavlm_model"])
    WavLMModel.from_pretrained(cfg["audio"]["wavlm_model"])
    print("WavLM:", cfg["audio"]["wavlm_model"])

    # the starting point for training the text model
    AutoTokenizer.from_pretrained(cfg["text"]["model"])
    AutoModelForSequenceClassification.from_pretrained(cfg["text"]["model"])
    print("RoBERTa:", cfg["text"]["model"])

    # the vision extension: a ViT for face features, a small face detector, and an expression classifier
    vision = cfg["vision"]
    ViTModel.from_pretrained(vision["model"], add_pooling_layer=False)
    for file, url in [(vision["detector_file"], vision["detector_url"]), (vision["expression_file"], vision["expression_url"])]:
        if not Path(file).exists():
            Path(file).parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, file)
    print("Face models:", vision["model"], vision["detector_file"], vision["expression_file"])

    # building the voice fetches Kokoro and spaCy's small English model; saying one word fetches the voice
    voice = cfg["voice"]
    tts = KPipeline(lang_code=voice["lang_code"], repo_id=voice["repo"], device="cpu")
    list(tts("Hello.", voice=voice["voice"]))
    print("Kokoro:", voice["repo"], voice["voice"])

    # the trained models and the page's sample clips, unless they are already here (for example after training)
    given = {Path(z).name: Path(z) for z in args.zip}
    release = cfg["release"]
    if all(Path(f).exists() for f in trained_files(cfg)) and release["models_asset"] not in given:
        print("Trained models: already here")
    else:
        unpack_release(cfg, release["models_asset"], given.get(release["models_asset"]))
    if any(Path(cfg["samples"]["dir"]).glob("*.wav")) and release["samples_asset"] not in given:
        print("Sample clips: already here")
    else:
        unpack_release(cfg, release["samples_asset"], given.get(release["samples_asset"]))

    print(f"\nNow fetch the reply model:  ollama pull {cfg['reply']['model']}")


if __name__ == "__main__":
    main()
