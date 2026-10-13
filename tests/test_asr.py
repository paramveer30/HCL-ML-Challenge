from types import SimpleNamespace

import numpy as np

from meldrt.pipeline import asr

WAV = np.zeros(16000, dtype=np.float32)


class FakeWhisper:
    """Returns prepared segments instead of listening."""

    segments = []

    def __init__(self, model_name, device, compute_type):
        self.loaded = (model_name, device, compute_type)

    def transcribe(self, wav, language, beam_size):
        return iter(self.segments), None


def segment(text, no_speech_prob):
    return SimpleNamespace(text=text, no_speech_prob=no_speech_prob)


def make(monkeypatch, segments):
    monkeypatch.setattr(asr, "WhisperModel", FakeWhisper)
    FakeWhisper.segments = segments
    return asr.Asr("small.en", "cpu", "int8_float32", beam_size=1, max_no_speech_prob=0.75)


def test_segments_are_joined_into_one_line(monkeypatch):
    model = make(monkeypatch, [segment(" I got the job!", 0.02), segment(" Can you believe it?", 0.05)])
    assert model.transcribe(WAV) == "I got the job! Can you believe it?"
    assert model.model.loaded == ("small.en", "cpu", "int8_float32")


def test_a_word_made_up_from_silence_is_dropped(monkeypatch):
    assert make(monkeypatch, [segment(" you", 0.86)]).transcribe(WAV) == ""


def test_speech_with_one_unsure_segment_is_kept(monkeypatch):
    model = make(monkeypatch, [segment(" Hi Ross!", 0.59), segment(" you", 0.9)])
    assert model.transcribe(WAV) == "Hi Ross! you"


def test_no_segments_means_no_words(monkeypatch):
    assert make(monkeypatch, []).transcribe(WAV) == ""
