"""Builds the whole live pipeline from the config: every model loaded and wired together."""
import json
from pathlib import Path

import torch

from meldrt.config import load_config
from meldrt.pipeline.asr import Asr
from meldrt.pipeline.classify import EmotionClassifier
from meldrt.pipeline.speak import Speaker, play_tone
from meldrt.pipeline.turn import Pipeline
from meldrt.pipeline.vad import Vad


def build_pipeline(sample_rate):
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vad = Vad(sample_rate, cfg["pipeline"]["vad_threshold"])
    a = cfg["asr"]
    asr = Asr(a["model"], device, a["compute_type"], a["beam_size"], a["max_no_speech_prob"])
    classifier = EmotionClassifier(cfg, device)
    # alpha was chosen on dev by train/tune_fusion.py
    with open(Path(cfg["fusion"]["run_dir"]) / "alpha.json", encoding="utf-8") as f:
        alpha = json.load(f)["alpha"]
    weights3 = None
    if cfg["pipeline"]["use_vision"]:
        # the three-model weights were chosen on dev by train/tune_fusion3.py
        with open(Path(cfg["fusion"]["run_dir"]) / "weights3.json", encoding="utf-8") as f:
            weights3 = json.load(f)["weights"]
    return Pipeline(vad, asr, classifier, cfg, alpha, sample_rate, weights3)


def build_speaker(play=True):
    return Speaker(load_config()["voice"], play=play)


def play_test_tone():
    play_tone(load_config()["voice"]["output_device"])
