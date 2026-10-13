"""Loads the cached WavLM features for the rows of a split."""
from pathlib import Path

import numpy as np


def load_features(df, name, cfg):
    """Returns (feats, has_audio).

    has_audio: one bool per row of df, False where no feature file exists (clip too short or missing).
    feats: float32 array (number of True rows, 13, 768), in row order.
    """
    feature_dir = Path(cfg["data"]["feature_root"]) / name
    paths = [feature_dir / (Path(p).stem + ".npy") for p in df["wav_path"]]
    has_audio = np.array([p.exists() for p in paths])
    feats = np.stack([np.load(p) for p, ok in zip(paths, has_audio) if ok]).astype(np.float32)
    return feats, has_audio
