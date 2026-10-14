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


def fuse_three(p_text, p_audio, p_vision, has_face, weights, alpha):
    """Blend three probability arrays, shape (n, 7). weights: (text, audio, vision), summing to 1.

    Rows with no face get the two-model blend with alpha instead, so without a camera
    the result is exactly the text + audio system.
    """
    w_text, w_audio, w_vision = weights
    three = w_text * p_text + w_audio * p_audio + w_vision * np.nan_to_num(p_vision)
    return np.where(has_face[:, None], three, late_fuse(p_text, p_audio, alpha))


def tune_three(p_text, p_audio, p_vision, has_face, alpha, y_true, step=0.05):
    """Try every (text, audio, vision) weight triple on a grid that sums to 1.

    Returns (best_weights, best_weighted_f1, grid). On a tie the triple with the smaller vision weight wins,
    then the smaller text weight: the search starts from vision 0.
    """
    n = round(1 / step)
    best_weights, best_f1, grid = None, -1.0, []
    for v in range(n + 1):
        for t in range(n + 1 - v):
            weights = (round(t * step, 6), round((n - v - t) * step, 6), round(v * step, 6))
            scores = f1_summary(y_true, fuse_three(p_text, p_audio, p_vision, has_face, weights, alpha).argmax(axis=1))
            grid.append({"weights": list(weights), **scores})
            if scores["weighted_f1"] > best_f1:
                best_weights, best_f1 = weights, scores["weighted_f1"]
    return best_weights, best_f1, grid


def tone_text_conflict(p_text, p_audio, min_conf=0.4):
    """For one line, shape (7,): True when the top labels differ and both top probabilities are >= min_conf."""
    labels_differ = p_text.argmax() != p_audio.argmax()
    both_sure = p_text.max() >= min_conf and p_audio.max() >= min_conf
    return bool(labels_differ and both_sure)
