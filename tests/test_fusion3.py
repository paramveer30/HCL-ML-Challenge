import numpy as np
import pytest

from meldrt.eval.ablation import vision_predictions
from meldrt.models.fusion import fuse_three, late_fuse, tune_three


def rows(*tops):
    """One probability row per entry: 0.8 on the given class, the rest shared out."""
    out = np.full((len(tops), 7), 0.2 / 6)
    for i, top in enumerate(tops):
        out[i, top] = 0.8
    return out


ALL = np.array([True, True, True])


def test_fuse_three_by_hand():
    p_text, p_audio, p_vision = rows(0), rows(1), rows(2)
    fused = fuse_three(p_text, p_audio, p_vision, np.array([True]), (0.5, 0.3, 0.2), alpha=0.45)
    assert fused[0, 0] == pytest.approx(0.5 * 0.8 + 0.5 * 0.2 / 6)
    assert fused[0, 2] == pytest.approx(0.2 * 0.8 + 0.8 * 0.2 / 6)
    assert fused.sum() == pytest.approx(1.0)


def test_rows_without_a_face_are_the_two_model_fusion():
    p_text, p_audio = rows(0, 0), rows(1, 1)
    p_vision = np.vstack([rows(2), np.full((1, 7), np.nan)])
    fused = fuse_three(p_text, p_audio, p_vision, np.array([True, False]), (0.2, 0.2, 0.6), alpha=0.45)
    assert fused[0].argmax() == 2
    assert np.allclose(fused[1], late_fuse(p_text[1], p_audio[1], 0.45))


def test_tune_three_trusts_vision_when_only_vision_is_right():
    y = np.array([0, 1, 2])
    weights, f1, grid = tune_three(rows(3, 3, 3), rows(4, 4, 4), rows(0, 1, 2), ALL, 0.45, y, step=0.25)
    assert f1 == pytest.approx(1.0)
    assert weights[2] >= 0.5
    assert len(grid) == 15                       # every triple of quarters that sums to 1
    assert all(sum(row["weights"]) == pytest.approx(1.0) for row in grid)


def test_tune_three_gives_vision_no_weight_when_it_adds_nothing():
    y = np.array([0, 1, 2])
    weights, f1, _ = tune_three(rows(0, 1, 2), rows(0, 1, 2), rows(5, 5, 5), ALL, 0.45, y, step=0.25)
    assert f1 == pytest.approx(1.0)
    assert weights[2] == 0.0


def test_the_three_extra_rows_and_the_no_face_rules():
    p_text, p_audio = rows(0, 0), rows(1, 1)
    p_vision = np.vstack([rows(2), np.full((1, 7), np.nan)])
    has_audio, has_face = np.array([True, True]), np.array([True, False])
    out = vision_predictions(p_text, p_audio, has_audio, p_vision, has_face, alpha=0.45,
                             alpha_text_vision=0.4, weights=(0.2, 0.2, 0.6), majority_class=6)
    assert out["vision only"].tolist() == [2, 6]            # no face: the majority class
    assert out["text + vision"].tolist() == [2, 0]          # no face: text alone
    assert out["text + audio + vision"].tolist() == [2, 1]  # no face: text + audio with alpha 0.45
