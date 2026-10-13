"""Counts every learned parameter on the local inference path and fails if the total is over the cap.

Nothing here is a hardcoded number: PyTorch models are counted from the objects the pipeline loads,
Whisper from the CTranslate2 file faster-whisper loads, and the reply model from the tensor shapes
in the GGUF file Ollama loads.
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"

import json
import sys
from pathlib import Path

from faster_whisper.utils import download_model

from meldrt.config import load_config
from meldrt.eval.param_count import count_ct2, count_gguf, count_spacy, count_torch
from meldrt.pipeline.build import build_pipeline, build_speaker


def ollama_model_file(model):
    """Path of the GGUF file behind an Ollama model name such as 'qwen2.5:1.5b'."""
    root = Path(os.environ.get("OLLAMA_MODELS", Path.home() / ".ollama" / "models"))
    name, tag = model.split(":")
    with open(root / "manifests" / "registry.ollama.ai" / "library" / name / tag, encoding="utf-8") as f:
        layers = json.load(f)["layers"]
    digest = next(layer["digest"] for layer in layers if layer["mediaType"].endswith("image.model"))
    return root / "blobs" / digest.replace(":", "-")


def main():
    cfg = load_config()
    pipeline = build_pipeline(sample_rate=cfg["audio"]["sample_rate"])
    speaker = build_speaker(play=False)
    classifier = pipeline.classifier
    whisper_dir = download_model(cfg["asr"]["model"], local_files_only=True)

    rows = [
        ("Silero VAD", count_torch(pipeline.vad.model), "loaded model"),
        (f"Whisper {cfg['asr']['model']}", count_ct2(Path(whisper_dir) / "model.bin"), "CTranslate2 file"),
        ("RoBERTa-base with emotion head", count_torch(classifier.text_model), "loaded model"),
        ("WavLM-base-plus", count_torch(classifier.wavlm), "loaded model"),
        ("Audio head", count_torch(classifier.head), "loaded model"),
        (f"Reply model {cfg['reply']['model']}", count_gguf(ollama_model_file(cfg["reply"]["model"])), "GGUF tensor shapes"),
        ("Kokoro-82M", count_torch(speaker.tts.model), "loaded model"),
        ("Kokoro voice embedding", sum(v.numel() for v in speaker.tts.voices.values()), "loaded tensor"),
        ("spaCy model in Kokoro's text front end", count_spacy(speaker.tts.g2p.nlp), "loaded model"),
    ]
    total = sum(count for _, count, _ in rows)
    cap = cfg["eval"]["max_params"]

    print("\n| Component | Parameters | Counted from |")
    print("| --- | ---: | --- |")
    for name, count, source in rows:
        print(f"| {name} | {count:,} | {source} |")
    print(f"| **Total** | **{total:,}** | {100 * total / cap:.1f}% of the {cap:,} cap |")

    out_dir = Path(cfg["eval"]["bench_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump({"rows": [{"component": n, "parameters": c, "source": s} for n, c, s in rows],
                   "total": total, "cap": cap}, f, indent=2)

    if total > cap:
        sys.exit(f"over the cap: {total:,} > {cap:,}")


if __name__ == "__main__":
    main()
