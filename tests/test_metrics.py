import pytest

from meldrt.eval.metrics import confusion, f1_summary, per_class_f1

# 6 examples: neutral (0) x3, joy (1) x2, anger (3) x1
Y_TRUE = [0, 0, 0, 1, 1, 3]
Y_PRED = [0, 0, 1, 1, 0, 0]
# by hand: neutral P=2/4 R=2/3 F1=4/7; joy P=1/2 R=1/2 F1=1/2; anger F1=0; other four classes F1=0
F1_NEUTRAL, F1_JOY = 4 / 7, 1 / 2


def test_per_class_f1_matches_hand_computation():
    scores = per_class_f1(Y_TRUE, Y_PRED)
    assert scores["neutral"] == pytest.approx(F1_NEUTRAL)
    assert scores["joy"] == pytest.approx(F1_JOY)
    assert scores["anger"] == 0.0
    assert len(scores) == 7


def test_weighted_f1_weights_by_true_count():
    expected = (3 * F1_NEUTRAL + 2 * F1_JOY + 1 * 0.0) / 6
    assert f1_summary(Y_TRUE, Y_PRED)["weighted_f1"] == pytest.approx(expected)


def test_macro_f1_averages_all_seven_classes():
    # classes that never appear still count as 0, so the denominator is 7
    expected = (F1_NEUTRAL + F1_JOY) / 7
    assert f1_summary(Y_TRUE, Y_PRED)["macro_f1"] == pytest.approx(expected)


def test_always_neutral_scores_low_macro_f1():
    y_true = [0] * 5 + [1, 2, 3, 4, 5]
    scores = f1_summary(y_true, [0] * 10)
    assert scores["macro_f1"] == pytest.approx((2 / 3) / 7)
    assert scores["weighted_f1"] == pytest.approx(0.5 * (2 / 3))


def test_confusion_rows_are_true_labels():
    m = confusion(Y_TRUE, Y_PRED)
    assert m.shape == (7, 7)
    assert m[0, 0] == 2 and m[0, 1] == 1   # neutral: 2 right, 1 called joy
    assert m[3, 0] == 1                    # the one anger was called neutral
    assert m.sum() == 6
