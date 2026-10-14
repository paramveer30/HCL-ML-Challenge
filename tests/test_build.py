import json

from meldrt.pipeline.build import live_fusion


def test_live_weights_from_the_config_win_with_their_own_temperature(tmp_path):
    cfg = {"fusion": {"live_weights": [0.4, 0.25, 0.35], "live_temperature": 0.6, "run_dir": str(tmp_path)}}
    assert live_fusion(cfg, 0.7) == ([0.4, 0.25, 0.35], 0.6)


def test_without_live_weights_the_dev_tuned_ones_are_used(tmp_path):
    (tmp_path / "weights3.json").write_text(json.dumps({"weights": [0.45, 0.55, 0.0]}))
    cfg = {"fusion": {"live_weights": None, "live_temperature": 0.6, "run_dir": str(tmp_path)}}
    assert live_fusion(cfg, 0.7) == ([0.45, 0.55, 0.0], 0.7)
