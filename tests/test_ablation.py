import numpy as np

from meldrt.eval.ablation import ablation_predictions


def probs(*pairs):
    row = np.zeros(7)
    for index, p in pairs:
        row[index] = p
    return row


def test_four_rows_and_the_no_audio_rules():
    p_text = np.stack([probs((1, 0.6), (3, 0.4)), probs((2, 0.9), (0, 0.1)), probs((4, 0.7), (0, 0.3))])
    p_audio = np.stack([probs((3, 0.9), (1, 0.1)), probs((2, 0.8), (0, 0.2)), np.full(7, np.nan)])
    has_audio = np.array([True, True, False])

    rows = ablation_predictions(p_text, p_audio, has_audio, alpha=0.5, majority_class=0)

    assert list(rows) == ["majority", "text only", "audio only", "fused"]
    assert rows["majority"].tolist() == [0, 0, 0]
    assert rows["text only"].tolist() == [1, 2, 4]
    assert rows["audio only"].tolist() == [3, 2, 0]     # last row has no audio: majority class
    # row 0: 0.5*0.4 + 0.5*0.9 = 0.65 for anger beats 0.35 for joy. last row: text alone.
    assert rows["fused"].tolist() == [3, 2, 4]


def test_alpha_one_makes_fused_equal_text():
    rng = np.random.default_rng(0)
    p_text = rng.dirichlet(np.ones(7), size=20)
    p_audio = rng.dirichlet(np.ones(7), size=20)
    rows = ablation_predictions(p_text, p_audio, np.ones(20, dtype=bool), alpha=1.0, majority_class=0)
    assert rows["fused"].tolist() == rows["text only"].tolist()
