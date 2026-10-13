from meldrt.config import load_config


def test_default_config_has_the_three_splits():
    cfg = load_config()
    assert set(cfg["data"]["splits"]) == {"train", "dev", "test"}
    assert cfg["audio"]["sample_rate"] == 16000


def test_load_config_reads_a_given_file(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("audio:\n  sample_rate: 8000\n", encoding="utf-8")
    assert load_config(p)["audio"]["sample_rate"] == 8000
