"""Predictions for the four rows of the ablation table: majority, text only, audio only, fused."""
import numpy as np

from meldrt.models.fusion import fill_missing_audio, late_fuse


def ablation_predictions(p_text, p_audio, has_audio, alpha, majority_class):
    """p_text, p_audio: (n, 7) probabilities; p_audio rows may be NaN where has_audio is False.

    Rows with no audio: 'audio only' predicts the majority class, and 'fused' uses text alone.
    """
    p_audio = fill_missing_audio(p_text, p_audio, has_audio)
    return {
        "majority": np.full(len(p_text), majority_class),
        "text only": p_text.argmax(axis=1),
        "audio only": np.where(has_audio, p_audio.argmax(axis=1), majority_class),
        "fused": late_fuse(p_text, p_audio, alpha).argmax(axis=1),
    }
