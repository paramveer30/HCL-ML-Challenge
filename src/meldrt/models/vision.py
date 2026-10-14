"""Face features from a frozen ViT trained for facial expressions. The head on top is the same design as the audio head."""
import numpy as np
import torch


def frame_layers(crops, model, device):
    """crops: uint8 RGB face images, shape (n, 224, 224, 3), n >= 1.
    Returns the [CLS] token of every ViT layer for every face, shape (n, 13, 768), float32."""
    pixels = torch.from_numpy(crops).to(device).permute(0, 3, 1, 2).float() / 255
    pixels = (pixels - 0.5) / 0.5                     # the scaling this ViT was trained with
    with torch.no_grad():
        out = model(pixel_values=pixels, output_hidden_states=True)
    layers = torch.stack(out.hidden_states)           # (13, n, 197, 768): 196 patches plus [CLS] at position 0
    return layers[:, :, 0].permute(1, 0, 2).cpu().numpy()


def mean_over_faces(layers):
    """layers: (n, 13, 768), one entry per face. Returns their average, shape (13, 768), float16 as cached."""
    return np.mean(layers, axis=0).astype(np.float16)


def face_features(crops, model, device):
    """The features cached for one clip: every ViT layer's [CLS] token averaged over the clip's faces."""
    return mean_over_faces(frame_layers(crops, model, device))
