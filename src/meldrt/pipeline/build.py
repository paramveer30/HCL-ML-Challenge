"""Builds the whole live pipeline from the config: every model loaded and wired together."""
import json
from pathlib import Path

import torch

from meldrt.config import load_config
from meldrt.pipeline.asr import Asr
from meldrt.pipeline.classify import EmotionClassifier
from meldrt.pipeline.turn import Pipeline
from meldrt.pipeline.vad import Vad


def build_pipeline(sample_rate):
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vad = Vad(sample_rate, cfg["pipeline"]["vad_threshold"])
    asr = Asr(cfg["asr"]["model"], device, cfg["asr"]["compute_type"], cfg["asr"]["beam_size"])
    classifier = EmotionClassifier(cfg, device)
    # alpha was chosen on dev by train/tune_fusion.py
    with open(Path(cfg["fusion"]["run_dir"]) / "alpha.json", encoding="utf-8") as f:
        alpha = json.load(f)["alpha"]
    return Pipeline(vad, asr, classifier, cfg, alpha, sample_rate)
