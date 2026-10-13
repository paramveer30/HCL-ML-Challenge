from types import SimpleNamespace

import numpy as np
import torch

from meldrt.train.cache_audio import clip_features


def fake_extractor(wav, sampling_rate, return_tensors):
    return SimpleNamespace(input_values=torch.tensor(wav).unsqueeze(0))


def fake_model(input_values, output_hidden_states):
    # 13 layers, 5 time steps; layer i is filled with i at t=0 and i + 2 at the other steps
    layers = []
    for i in range(13):
        h = torch.full((1, 5, 768), float(i + 2))
        h[:, 0, :] = float(i)
        layers.append(h)
    return SimpleNamespace(hidden_states=tuple(layers))


def test_clip_features_averages_over_time_per_layer():
    wav = np.zeros(1600, dtype=np.float32)
    feats = clip_features(wav, fake_model, fake_extractor, 16000, "cpu")
    assert feats.shape == (13, 768)
    assert feats.dtype == np.float16
    for i in range(13):
        assert np.allclose(feats[i], i + 1.6, atol=1e-2)
