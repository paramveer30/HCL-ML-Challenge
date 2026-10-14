"""Face features and facial expression from a frozen ViT trained for facial expressions."""
import numpy as np
import torch

from meldrt.labels import LABELS

# the ViT's own classifier names its classes like this; MELD calls the same seven emotions differently
EXPRESSION_TO_MELD = {"angry": "anger", "disgust": "disgust", "fear": "fear", "happy": "joy",
                      "neutral": "neutral", "sad": "sadness", "surprise": "surprise"}


def expression_order(id2label):
    """id2label: the ViT classifier's {index: name}. Returns, for each MELD label in order, the index of its class."""
    index_of = {EXPRESSION_TO_MELD[name]: int(index) for index, name in id2label.items()}
    return [index_of[label] for label in LABELS]


def face_outputs(crops, model, device):
    """crops: uint8 RGB face images, shape (n, 224, 224, 3), n >= 1. model: the ViT with its expression classifier.

    Returns (layers, expression):
      layers: the [CLS] token of every ViT layer for every face, shape (n, 13, 768), float32
      expression: the classifier's probabilities for every face in MELD label order, shape (n, 7)
    """
    pixels = torch.from_numpy(crops).to(device).permute(0, 3, 1, 2).float() / 255
    pixels = (pixels - 0.5) / 0.5                     # the scaling this ViT was trained with
    with torch.no_grad():
        out = model(pixel_values=pixels, output_hidden_states=True)
    hidden = torch.stack(out.hidden_states)           # (13, n, 197, 768): 196 patches plus [CLS] at position 0
    layers = hidden[:, :, 0].permute(1, 0, 2).cpu().numpy()
    probs = torch.softmax(out.logits, dim=1)[:, expression_order(model.config.id2label)]
    return layers, probs.cpu().numpy()


def mean_over_faces(layers):
    """layers: (n, 13, 768), one entry per face. Returns their average, shape (13, 768), float16 as cached."""
    return np.mean(layers, axis=0).astype(np.float16)
