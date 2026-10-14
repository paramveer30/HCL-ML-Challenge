"""Trains the audio head on cached WavLM features."""
import argparse

from meldrt.config import load_config
from meldrt.train.train_head import train_head

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config()
    train_head(cfg, cfg["audio_head"], cfg["data"]["feature_root"], args.seed, "audio")
