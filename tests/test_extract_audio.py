from meldrt.data.extract_audio import ffmpeg_cmd


def test_ffmpeg_cmd_asks_for_16k_mono_capped_wav():
    cmd = ffmpeg_cmd("in.mp4", "out.part", 16000, 15)
    assert cmd[cmd.index("-ar") + 1] == "16000"
    assert cmd[cmd.index("-ac") + 1] == "1"
    assert cmd[cmd.index("-t") + 1] == "15"
    assert cmd[cmd.index("-i") + 1] == "in.mp4"
    assert cmd[-1] == "out.part"
