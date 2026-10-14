import zipfile

from meldrt.config import load_config
from scripts.package_models import pack, sample_files, trained_files


def test_the_models_zip_holds_everything_the_live_system_loads():
    cfg = load_config()
    names = {f.as_posix() for f in trained_files(cfg)}
    assert cfg["audio_head"]["checkpoint"] in names and cfg["vision_head"]["checkpoint"] in names
    assert {"runs/fusion/alpha.json", "runs/fusion/weights3.json", "runs/fusion/calibration.json"} <= names
    assert "checkpoints/text/best/model.safetensors" in names


def test_pack_keeps_paths_relative_to_the_repo(tmp_path):
    (tmp_path / "checkpoints" / "audio").mkdir(parents=True)
    (tmp_path / "checkpoints" / "audio" / "best.pt").write_bytes(b"weights")
    out = tmp_path / "dist" / "models.zip"
    pack(["checkpoints/audio/best.pt"], out, root=tmp_path)
    with zipfile.ZipFile(out) as z:
        assert z.namelist() == ["checkpoints/audio/best.pt"] and z.read("checkpoints/audio/best.pt") == b"weights"


def test_pack_refuses_a_missing_file(tmp_path):
    try:
        pack(["checkpoints/text/best/model.safetensors"], tmp_path / "models.zip", root=tmp_path)
    except FileNotFoundError as e:
        assert "model.safetensors" in str(e)
    else:
        raise AssertionError("a missing file must not give a half-empty zip")


def test_samples_are_the_wav_and_mp4_files(tmp_path):
    clips = tmp_path / "ui" / "clips"
    clips.mkdir(parents=True)
    for name in ["joy_dia1_utt1.wav", "joy_dia1_utt1.mp4", "notes.txt"]:
        (clips / name).write_bytes(b"x")
    found = [f.as_posix() for f in sample_files({"samples": {"dir": "ui/clips"}}, root=tmp_path)]
    assert found == ["ui/clips/joy_dia1_utt1.mp4", "ui/clips/joy_dia1_utt1.wav"]
