import numpy as np
import pytest

from meldrt.models.fusion import fill_missing_audio, late_fuse, softmax, tone_text_conflict, tune_alpha


def probs(*pairs):
    """Builds one 7-way probability row from (class index, probability) pairs."""
    row = np.zeros(7)
    for index, p in pairs:
        row[index] = p
    return row


# --- late_fuse ---

def test_alpha_one_is_text_and_alpha_zero_is_audio():
    p_text, p_audio = probs((0, 0.7), (1, 0.3)), probs((0, 0.2), (1, 0.8))
    assert np.allclose(late_fuse(p_text, p_audio, 1.0), p_text)
    assert np.allclose(late_fuse(p_text, p_audio, 0.0), p_audio)


def test_fused_value_by_hand():
    p_text, p_audio = probs((0, 0.7), (1, 0.3)), probs((0, 0.2), (1, 0.8))
    fused = late_fuse(p_text, p_audio, 0.6)
    assert fused[0] == pytest.approx(0.6 * 0.7 + 0.4 * 0.2)
    assert fused[1] == pytest.approx(0.6 * 0.3 + 0.4 * 0.8)


def test_fused_rows_still_sum_to_one():
    rng = np.random.default_rng(0)
    p_text = rng.dirichlet(np.ones(7), size=5)
    p_audio = rng.dirichlet(np.ones(7), size=5)
    fused = late_fuse(p_text, p_audio, 0.35)
    assert fused.shape == (5, 7)
    assert np.allclose(fused.sum(axis=1), 1.0)


# --- tune_alpha ---

def test_tune_alpha_trusts_text_when_only_text_is_right():
    y = np.array([0, 1, 2, 3])
    p_text = np.stack([probs((c, 0.8), ((c + 1) % 7, 0.2)) for c in y])       # right
    p_audio = np.stack([probs((c, 0.2), ((c + 1) % 7, 0.8)) for c in y])      # wrong
    alpha, f1 = tune_alpha(p_text, p_audio, y)
    assert alpha >= 0.5       # at exactly 0.5 the two tie and argmax happens to pick the right one
    assert f1 == pytest.approx(1.0)


def test_tune_alpha_trusts_audio_when_only_audio_is_right():
    y = np.array([0, 1, 2, 3])
    p_text = np.stack([probs((c, 0.2), ((c + 1) % 7, 0.8)) for c in y])
    p_audio = np.stack([probs((c, 0.8), ((c + 1) % 7, 0.2)) for c in y])
    alpha, f1 = tune_alpha(p_text, p_audio, y)
    assert alpha < 0.5
    assert f1 == pytest.approx(1.0)


def test_tune_alpha_can_find_a_mix_that_beats_both():
    y = np.array([1, 0])
    # row 0: text is mildly wrong, audio strongly right. row 1: the other way round.
    p_text = np.stack([probs((0, 0.6), (1, 0.4)), probs((0, 0.9), (1, 0.1))])
    p_audio = np.stack([probs((0, 0.1), (1, 0.9)), probs((0, 0.4), (1, 0.6))])
    alpha, f1 = tune_alpha(p_text, p_audio, y)
    assert 0.0 < alpha < 1.0
    assert f1 == pytest.approx(1.0)


def test_tune_alpha_returns_a_grid_value():
    rng = np.random.default_rng(1)
    p_text = rng.dirichlet(np.ones(7), size=40)
    p_audio = rng.dirichlet(np.ones(7), size=40)
    y = rng.integers(0, 7, size=40)
    alpha, f1 = tune_alpha(p_text, p_audio, y, step=0.05)
    assert 0.0 <= alpha <= 1.0
    assert (alpha / 0.05) == pytest.approx(round(alpha / 0.05))
    assert 0.0 <= f1 <= 1.0


def test_tune_alpha_prefers_the_smaller_alpha_on_a_tie():
    # both models are right on every row, so every alpha scores 1.0
    y = np.array([0, 1])
    p = np.stack([probs((0, 0.9), (1, 0.1)), probs((0, 0.1), (1, 0.9))])
    alpha, f1 = tune_alpha(p, p, y)
    assert alpha == 0.0
    assert f1 == pytest.approx(1.0)


# --- softmax and fill_missing_audio ---

def test_softmax_rows_sum_to_one_and_keep_the_order():
    logits = np.array([[2.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0], [1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    p = softmax(logits)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert p[0].argmax() == 0 and p[0, 0] > p[0, 1] > p[0, 2] > p[0, 3]
    assert np.isfinite(p).all()                      # a huge logit must not overflow
    assert p[0, 0] / p[0, 1] == pytest.approx(np.e)   # one more logit point is a factor of e


def test_rows_without_audio_fuse_to_text_alone():
    p_text = np.stack([probs((0, 0.7), (1, 0.3)), probs((2, 0.6), (3, 0.4))])
    p_audio = np.stack([probs((1, 1.0)), np.full(7, np.nan)])
    has_audio = np.array([True, False])
    filled = fill_missing_audio(p_text, p_audio, has_audio)
    assert np.allclose(filled[0], p_audio[0])
    for alpha in [0.0, 0.3, 1.0]:
        assert np.allclose(late_fuse(p_text, filled, alpha)[1], p_text[1])


# --- tone_text_conflict ---

def test_conflict_when_labels_differ_and_both_are_confident():
    assert tone_text_conflict(probs((1, 0.5), (0, 0.5 - 1e-9)), probs((3, 0.6), (0, 0.4)), min_conf=0.4)


def test_no_conflict_when_labels_agree():
    assert not tone_text_conflict(probs((1, 0.9), (0, 0.1)), probs((1, 0.5), (0, 0.5 - 1e-9)), min_conf=0.4)


def test_no_conflict_when_one_side_is_unsure():
    p_text = probs((1, 0.9), (0, 0.1))
    p_audio = probs((3, 0.35), (0, 0.33), (2, 0.32))     # different label, but only 0.35 sure
    assert not tone_text_conflict(p_text, p_audio, min_conf=0.4)
    assert tone_text_conflict(p_text, p_audio, min_conf=0.3)
