"""Downloads every pretrained model once, so the live scripts can run with the network switched off.

Run this once after installing. The reply model is fetched separately: ollama pull qwen2.5:1.5b
"""
import urllib.request
from pathlib import Path

from faster_whisper.utils import download_model
from kokoro import KPipeline
from transformers import AutoFeatureExtractor, AutoModelForSequenceClassification, AutoTokenizer, ViTModel, WavLMModel

from meldrt.config import load_config


def main():
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

    print(f"\nNow fetch the reply model:  ollama pull {cfg['reply']['model']}")


if __name__ == "__main__":
    main()
