"""Runs frozen WavLM over every extracted WAV and caches 13 x 768 features per clip. Safe to rerun."""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, WavLMModel

from meldrt.config import load_config
from meldrt.models.audio import clip_features


def cache_split(name, cfg, model, extractor, device):
    wav_dir = Path(cfg["data"]["wav_root"]) / name
    out_dir = Path(cfg["data"]["feature_root"]) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    sr, min_s = cfg["audio"]["sample_rate"], cfg["audio"]["min_seconds"]

    wavs = sorted(wav_dir.glob("*.wav"))
    todo = [w for w in wavs if not (out_dir / (w.stem + ".npy")).exists()]
    print(f"{name}: {len(wavs)} wavs, {len(wavs) - len(todo)} already cached", flush=True)

    skipped = []
    for n, wav_path in enumerate(todo, 1):
        wav, file_sr = sf.read(wav_path, dtype="float32")
        if file_sr != sr or len(wav) < min_s * sr:
            skipped.append(wav_path.name)
            continue
        feats = clip_features(wav, model, extractor, sr, device)
        # save under a temporary name, then rename, so a killed run leaves no half-written file
        tmp = out_dir / (wav_path.stem + ".tmp.npy")
        np.save(tmp, feats)
        tmp.replace(out_dir / (wav_path.stem + ".npy"))
        if n % 500 == 0:
            print(f"{name}: {n}/{len(todo)}", flush=True)

    (out_dir / "skipped.txt").write_text("\n".join(skipped), encoding="utf-8")
    print(f"{name}: done, {len(skipped)} skipped (see {out_dir / 'skipped.txt'})", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    args = parser.parse_args()
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = AutoFeatureExtractor.from_pretrained(cfg["audio"]["wavlm_model"])
    model = WavLMModel.from_pretrained(cfg["audio"]["wavlm_model"]).to(device).eval()
    for name in args.splits:
        cache_split(name, cfg, model, extractor, device)
