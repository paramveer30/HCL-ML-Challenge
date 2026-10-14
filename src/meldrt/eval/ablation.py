"""Predictions for the four rows of the ablation table: majority, text only, audio only, fused."""
import numpy as np

from meldrt.models.fusion import fill_missing_audio, fuse_three, late_fuse


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


def vision_predictions(p_text, p_audio, has_audio, p_vision, has_face, alpha, alpha_text_vision, weights, majority_class):
    """The three extra rows of the table once vision is added. p_vision rows may be NaN where has_face is False.

    Rows with no face: 'vision only' predicts the majority class, 'text + vision' uses text alone,
    and 'text + audio + vision' is the two-model fusion.
    """
    p_audio = fill_missing_audio(p_text, p_audio, has_audio)
    p_vision = fill_missing_audio(p_text, p_vision, has_face)
    return {
        "vision only": np.where(has_face, p_vision.argmax(axis=1), majority_class),
        "text + vision": late_fuse(p_text, p_vision, alpha_text_vision).argmax(axis=1),
        "text + audio + vision": fuse_three(p_text, p_audio, p_vision, has_face, weights, alpha).argmax(axis=1),
    }
