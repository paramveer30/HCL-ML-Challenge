"""Audio emotion head: a learned mix of the 13 WavLM layers, then a small MLP."""
import numpy as np
import torch
from torch import nn


def clip_features(wav, model, extractor, sample_rate, device):
    """wav: 1-D float32 array. Returns the time average of every WavLM layer, shape (13, 768), float16."""
    inputs = extractor(wav, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        out = model(inputs.input_values.to(device), output_hidden_states=True)
    layers = torch.stack(out.hidden_states)       # (13, 1, T, 768)
    return layers.mean(dim=2).squeeze(1).cpu().numpy().astype(np.float16)


class AudioHead(nn.Module):
    def __init__(self, n_layers=13, dim=768, hidden=256, n_classes=7, dropout=0.2):
        super().__init__()
        # one score per WavLM layer; zeros mean every layer starts with the same weight
        self.layer_scores = nn.Parameter(torch.zeros(n_layers))
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def layer_weights(self):
        """Softmax of the layer scores, shape (13,)."""
        return torch.softmax(self.layer_scores, dim=0)

    def forward(self, feats):
        """feats: (batch, 13, 768) float32. Returns logits, shape (batch, 7)."""
        weights = self.layer_weights().view(1, -1, 1)    # (1, 13, 1), lines up with the layer axis
        mixed = (weights * feats).sum(dim=1)             # (batch, 768)
        return self.mlp(mixed)
