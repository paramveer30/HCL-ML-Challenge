"""The emotion head used on top of a frozen encoder: a learned mix of its 13 layers, then a small MLP.

The audio model (on WavLM features) and the vision model (on face ViT features) are both this head.
"""
import torch
from torch import nn


class LayerMixHead(nn.Module):
    def __init__(self, n_layers=13, dim=768, hidden=256, n_classes=7, dropout=0.2):
        super().__init__()
        # one score per encoder layer; zeros mean every layer starts with the same weight
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
