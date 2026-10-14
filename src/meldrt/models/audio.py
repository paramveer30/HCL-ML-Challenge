"""Audio features from frozen WavLM: every layer averaged over time."""
import numpy as np
import torch


def clip_features(wav, model, extractor, sample_rate, device):
    """wav: 1-D float32 array. Returns the time average of every WavLM layer, shape (13, 768), float16."""
    inputs = extractor(wav, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        out = model(inputs.input_values.to(device), output_hidden_states=True)
    layers = torch.stack(out.hidden_states)       # (13, 1, T, 768)
    return layers.mean(dim=2).squeeze(1).cpu().numpy().astype(np.float16)
