"""Late fusion of the text and audio probabilities, and the tone/text conflict flag."""
import numpy as np

from meldrt.eval.metrics import f1_summary


def softmax(logits):
    """Turns logits, shape (7,) or (n, 7), into probabilities that sum to 1 along the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)    # subtracting the max avoids overflow
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def late_fuse(p_text, p_audio, alpha):
    """Blend two probability arrays, shape (7,) or (n, 7). alpha is the weight on text."""
    return alpha * p_text + (1 - alpha) * p_audio


def fill_missing_audio(p_text, p_audio, has_audio):
    """Rows with no audio get the text probabilities, so their fused result is text alone for any alpha."""
    return np.where(has_audio[:, None], p_audio, p_text)


def tune_alpha(p_text, p_audio, y_true, step=0.05):
    """Try alpha = 0, step, ..., 1. Returns (best_alpha, best_weighted_f1). On a tie the smaller alpha wins."""
    best_alpha, best_f1 = 0.0, -1.0
    for i in range(round(1 / step) + 1):
        alpha = round(i * step, 6)
        predictions = late_fuse(p_text, p_audio, alpha).argmax(axis=1)
        f1 = f1_summary(y_true, predictions)["weighted_f1"]
        if f1 > best_f1:
            best_alpha, best_f1 = alpha, f1
    return best_alpha, best_f1


def tone_text_conflict(p_text, p_audio, min_conf=0.4):
    """For one line, shape (7,): True when the top labels differ and both top probabilities are >= min_conf."""
    labels_differ = p_text.argmax() != p_audio.argmax()
    both_sure = p_text.max() >= min_conf and p_audio.max() >= min_conf
    return bool(labels_differ and both_sure)
