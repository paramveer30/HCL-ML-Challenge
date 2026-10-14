"""The two face models: ViT features for the head trained on MELD, and a facial expression classifier for a live camera."""
import numpy as np
import torch

from meldrt.labels import LABELS

# the expression classifier (HSEmotion, EfficientNet-B0 trained on AffectNet) has 8 classes in this order.
# MELD has no contempt, so that class is dropped and the other seven are rescaled to sum to 1
EXPRESSION_CLASSES = ["anger", "contempt", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
EXPRESSION_ORDER = [EXPRESSION_CLASSES.index(label) for label in LABELS]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def face_layers(crops, model, device):
    """crops: uint8 RGB face images, shape (n, 224, 224, 3), n >= 1. model: the frozen face ViT.
    Returns the [CLS] token of every ViT layer for every face, shape (n, 13, 768), float32."""
    pixels = torch.from_numpy(crops).to(device).permute(0, 3, 1, 2).float() / 255
    pixels = (pixels - 0.5) / 0.5                     # the scaling this ViT was trained with
    with torch.no_grad():
        out = model(pixel_values=pixels, output_hidden_states=True)
    hidden = torch.stack(out.hidden_states)           # (13, n, 197, 768): 196 patches plus [CLS] at position 0
    return hidden[:, :, 0].permute(1, 0, 2).cpu().numpy()


def mean_over_faces(layers):
    """layers: (n, 13, 768), one entry per face. Returns their average, shape (13, 768), float16 as cached."""
    return np.mean(layers, axis=0).astype(np.float16)


def expression_probs(crops, session):
    """crops: uint8 RGB face images, shape (n, 224, 224, 3). session: the ONNX expression classifier.
    Returns its probabilities for every face over the 7 MELD labels in MELD order, shape (n, 7)."""
    pixels = ((crops.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD).transpose(0, 3, 1, 2)
    scores = session.run(None, {session.get_inputs()[0].name: pixels})[0][:, EXPRESSION_ORDER]
    exp = np.exp(scores - scores.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)
