import numpy as np

from scripts.make_samples import concat_cmd, join_wavs, sample_name, segment_cmd
from ui.pipeline import sample_labels


def test_a_sample_name_starts_with_the_emotion_of_every_line():
    assert sample_name(["fear"], "dia40_utt8") == "fear_dia40_utt8"
    assert sample_name(["neutral", "joy", "surprise"], "dia3") == "neutral_joy_surprise_dia3"


def test_the_page_reads_the_labels_back_from_the_name():
    assert sample_labels(sample_name(["neutral", "joy", "surprise"], "dia3") + ".wav") == ["neutral", "joy", "surprise"]


def test_joined_lines_have_silence_between_them_but_not_after_the_last():
    a, b = np.ones(16000, dtype=np.float32), np.ones(8000, dtype=np.float32)
    joined = join_wavs([a, b], gap_seconds=1.5, sample_rate=16000)
    assert len(joined) == 16000 + 24000 + 8000
    assert joined[16000:40000].max() == 0.0 and joined[-1] == 1.0


def test_one_line_alone_is_unchanged():
    a = np.ones(100, dtype=np.float32)
    assert np.array_equal(join_wavs([a], 1.5, 16000), a)


def test_a_video_piece_is_cut_or_held_to_the_length_of_its_sound():
    cmd = segment_cmd("line.mp4", "piece.mp4", 4.25)
    assert cmd[0] == "ffmpeg" and "line.mp4" in cmd and cmd[-1] == "piece.mp4"
    assert cmd[cmd.index("-t") + 1] == "4.250"
    assert "libx264" in cmd and "-an" in cmd          # a format browsers play, and no sound track


def test_pieces_are_joined_without_encoding_again():
    cmd = concat_cmd("pieces.txt", "out.mp4")
    assert cmd[cmd.index("-i") + 1] == "pieces.txt" and cmd[cmd.index("-c") + 1] == "copy"
