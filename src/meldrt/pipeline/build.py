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
    # the temperature was fitted on dev by train/calibrate.py
    with open(Path(cfg["fusion"]["run_dir"]) / "calibration.json", encoding="utf-8") as f:
        temperature = json.load(f)["temperature"]
    weights3, temperature3 = None, temperature
    if cfg["pipeline"]["use_vision"]:
        weights3, temperature3 = live_fusion(cfg, temperature)
    return Pipeline(vad, asr, classifier, cfg, alpha, sample_rate, weights3, temperature, temperature3)


def live_fusion(cfg, temperature):
    """The (words, voice, face) weights and temperature for live turns that have a face."""
    f = cfg["fusion"]
    if f["live_weights"] is not None:
        # my choice for live use: the face gets a real say (dev scores of this mix in DECISIONS.md)
        return list(f["live_weights"]), f["live_temperature"]
    # the weights chosen on dev by train/tune_fusion3.py, the ones scored on test
    with open(Path(f["run_dir"]) / "weights3.json", encoding="utf-8") as fh:
        return json.load(fh)["weights"], temperature


def build_speaker(play=True):
    return Speaker(load_config()["voice"], play=play)


def play_test_tone():
    play_tone(load_config()["voice"]["output_device"])
