import pytest
import torch

from meldrt.models.audio import AudioHead


def test_output_is_seven_logits_per_clip():
    head = AudioHead()
    feats = torch.randn(4, 13, 768)
    assert head(feats).shape == (4, 7)


def test_layer_weights_start_equal_and_sum_to_one():
    w = AudioHead().layer_weights()
    assert w.shape == (13,)
    assert w.sum().item() == pytest.approx(1.0)
    assert torch.allclose(w, torch.full((13,), 1 / 13))


def test_parameter_count():
    # 13 layer scores + (768 * 256 + 256) + (256 * 7 + 7)
    n = sum(p.numel() for p in AudioHead().parameters())
    assert n == 13 + (768 * 256 + 256) + (256 * 7 + 7)


def test_layers_are_mixed_before_the_mlp():
    # put all the weight on layer 5: the output must then depend on layer 5 only
    head = AudioHead().eval()
    with torch.no_grad():
        head.layer_scores.fill_(-1e4)
        head.layer_scores[5] = 0.0
    a = torch.randn(2, 13, 768)
    b = a.clone()
    b[:, 0] = torch.randn(2, 768)      # change a layer that has no weight
    assert torch.allclose(head(a), head(b), atol=1e-5)
    b[:, 5] = torch.randn(2, 768)      # change the layer that has all the weight
    assert not torch.allclose(head(a), head(b), atol=1e-5)


def test_equal_weights_mean_the_average_of_the_layers():
    # with equal weights, 13 copies of x mix back to x, and so does any reordering of the layers
    head = AudioHead().eval()
    feats = torch.randn(3, 13, 768)
    shuffled = feats[:, torch.randperm(13)]
    assert torch.allclose(head(feats), head(shuffled), atol=1e-4)


def test_gradient_reaches_the_layer_scores():
    head = AudioHead()
    loss = head(torch.randn(8, 13, 768)).sum()
    loss.backward()
    assert head.layer_scores.grad is not None
    assert head.layer_scores.grad.abs().sum() > 0


def test_dropout_is_off_in_eval_mode():
    head = AudioHead().eval()
    feats = torch.randn(2, 13, 768)
    assert torch.equal(head(feats), head(feats))
