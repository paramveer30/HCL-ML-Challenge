import numpy as np
import pytest

from meldrt.labels import LABELS
from meldrt.pipeline.turn import Pipeline
from meldrt.schema import TurnState, View

CFG = {
    "audio": {"sample_rate": 16000},
    "data": {"context_k": 2},
    "fusion": {"conflict_min_conf": 0.4},
    "reply": {"low_conf": 0.45},
    "pipeline": {
        "frame_samples": 512, "start_ms": 250, "end_silence_ms": 600, "max_turn_s": 15,
        "preroll_ms": 200, "history_turns": 3, "speaker_name": "Speaker",
    },
}
FRAME = np.zeros(512, dtype=np.float32)


def probs(*pairs):
    row = np.zeros(7)
    for index, p in pairs:
        row[index] = p
    return row


class FakeVad:
    """Answers from a prepared list of flags instead of listening."""

    def __init__(self, flags):
        self.flags = list(flags)
        self.resets = 0

    def is_speech(self, frame):
        return self.flags.pop(0)

    def reset(self):
        self.resets += 1


class FakeAsr:
    def __init__(self, texts):
        self.texts = list(texts)
        self.lengths = []

    def transcribe(self, wav):
        self.lengths.append(len(wav))
        return self.texts.pop(0)


class FakeClassifier:
    def __init__(self, p_text, p_audio):
        self.p_text, self.p_audio = p_text, p_audio
        self.calls = []

    def text_probs(self, context, target):
        self.calls.append((context, target))
        return self.p_text

    def audio_probs(self, wav):
        return self.p_audio


ONE_TURN = [False] * 5 + [True] * 20 + [False] * 19      # 5 quiet, 20 speech, 19 quiet frames
JOY, ANGER = probs((1, 0.6), (0, 0.4)), probs((3, 0.7), (0, 0.3))


def run(pipeline, n_frames):
    return [pipeline.push_frame(FRAME) for _ in range(n_frames)]


def make(flags, texts, p_text=JOY, p_audio=ANGER, alpha=0.5):
    asr, classifier = FakeAsr(texts), FakeClassifier(p_text, p_audio)
    return Pipeline(FakeVad(flags), asr, classifier, CFG, alpha=alpha, sample_rate=16000), asr, classifier


def test_one_state_comes_out_on_the_frame_that_ends_the_turn():
    pipeline, asr, _ = make(ONE_TURN, ["it was great"])
    results = run(pipeline, len(ONE_TURN))
    assert all(r is None for r in results[:-1])
    state = results[-1]
    assert isinstance(state, TurnState)
    assert state.turn_id == 1
    assert state.transcript == "it was great"


def test_turn_audio_keeps_the_lead_in_and_drops_the_closing_silence():
    pipeline, asr, _ = make(ONE_TURN, ["it was great"])
    state = run(pipeline, len(ONE_TURN))[-1]
    # 5 quiet frames of lead-in + 20 speech frames; the 19 closing quiet frames are dropped
    assert asr.lengths == [25 * 512]
    assert state.audio_seconds == pytest.approx(25 * 512 / 16000)
    assert state.latency_ms["endpoint"] == pytest.approx(19 * 32)
    assert set(state.latency_ms) == {"endpoint", "asr", "audio", "text", "state"}


def test_fusion_and_views():
    pipeline, _, _ = make(ONE_TURN, ["it was great"], alpha=0.5)
    state = run(pipeline, len(ONE_TURN))[-1]
    assert state.text_view == View("joy", pytest.approx(0.6))
    assert state.audio_view == View("anger", pytest.approx(0.7))
    # fused: neutral 0.35, joy 0.30, anger 0.35 -> the first of the tied labels, neutral
    assert state.probs["joy"] == pytest.approx(0.30)
    assert state.probs["anger"] == pytest.approx(0.35)
    assert state.label == "neutral"
    assert state.tone_text_conflict is True          # labels differ, 0.6 and 0.7 are both at least 0.4
    assert list(state.probs) == LABELS


def test_no_conflict_when_one_model_is_unsure():
    unsure_audio = probs((3, 0.35), (0, 0.33), (2, 0.32))
    pipeline, _, _ = make(ONE_TURN, ["it was great"], p_audio=unsure_audio)
    assert run(pipeline, len(ONE_TURN))[-1].tone_text_conflict is False


def test_too_short_for_audio_falls_back_to_text():
    pipeline, _, _ = make(ONE_TURN, ["hi"], p_audio=None)
    state = run(pipeline, len(ONE_TURN))[-1]
    assert state.audio_view is None
    assert state.label == "joy" and state.confidence == pytest.approx(0.6)
    assert state.tone_text_conflict is False
    assert "audio" not in state.to_event()["views"]


def test_previous_turns_become_context():
    pipeline, _, classifier = make(ONE_TURN * 4, ["one", "two", "three", "four"])
    states = [s for s in run(pipeline, len(ONE_TURN) * 4) if s]
    assert [s.turn_id for s in states] == [1, 2, 3, 4]
    contexts = [context for context, _ in classifier.calls]
    assert contexts[0] == ""
    assert contexts[1] == "Speaker: one"
    assert contexts[3] == "Speaker: two\nSpeaker: three"      # only the last 2 turns
    assert classifier.calls[3][1] == "Speaker: four"
    assert [t["transcript"] for t in pipeline.history] == ["two", "three", "four"]   # history keeps 3


def test_reply_streams_tokens_then_done_and_is_remembered(monkeypatch):
    prompts = []

    def fake_stream(prompt, reply_cfg):
        prompts.append(prompt)
        yield from ["Hi ", "there", "!"]

    monkeypatch.setattr("meldrt.pipeline.turn.stream_reply", fake_stream)
    pipeline, _, _ = make(ONE_TURN * 2, ["hello", "how are you"])
    first, second = [s for s in run(pipeline, len(ONE_TURN) * 2) if s]

    events = list(pipeline.reply(second))
    assert [e["type"] for e in events] == ["token", "token", "token", "done"]
    assert [e["text"] for e in events[:3]] == ["Hi ", "there", "!"]
    done = events[-1]
    assert done["reply"] == "Hi there!" and done["turn_id"] == 2
    assert done["latency_ms"]["first_token"] <= done["latency_ms"]["reply"]
    assert pipeline.history[-1] == {"transcript": "how are you", "label": "neutral", "reply": "Hi there!"}
    # the prompt holds the earlier turn as history and the current turn as what was just said
    assert "Person: hello" in prompts[0]
    assert 'The person just said: "how are you"' in prompts[0]


def test_reset_forgets_the_conversation():
    pipeline, _, classifier = make(ONE_TURN * 2, ["one", "two"])
    run(pipeline, len(ONE_TURN))
    pipeline.reset()
    assert len(pipeline.history) == 0
    run(pipeline, len(ONE_TURN))
    assert classifier.calls[1][0] == ""          # the second turn sees no context from before the reset


def test_empty_transcript_gives_no_state():
    pipeline, _, _ = make(ONE_TURN * 2, ["", "real words"])
    states = [s for s in run(pipeline, len(ONE_TURN) * 2) if s]
    assert len(states) == 1
    assert states[0].turn_id == 1 and states[0].transcript == "real words"


def test_vad_is_reset_after_each_turn():
    pipeline, _, _ = make(ONE_TURN, ["hello"])
    run(pipeline, len(ONE_TURN))
    assert pipeline.vad.resets == 1


def test_wrong_frame_size_is_rejected():
    pipeline, _, _ = make(ONE_TURN, ["hello"])
    with pytest.raises(ValueError):
        pipeline.push_frame(np.zeros(480, dtype=np.float32))


def test_wrong_sample_rate_is_rejected():
    with pytest.raises(ValueError):
        Pipeline(FakeVad([]), FakeAsr([]), FakeClassifier(JOY, ANGER), CFG, alpha=0.5, sample_rate=44100)


def test_state_event_has_the_documented_shape():
    pipeline, _, _ = make(ONE_TURN, ["it was great"])
    event = run(pipeline, len(ONE_TURN))[-1].to_event()
    assert event["type"] == "state"
    assert set(event) == {"type", "turn_id", "transcript", "emotion", "views", "tone_text_conflict", "audio_seconds", "latency_ms"}
    assert set(event["emotion"]) == {"label", "confidence", "probs"}
    assert event["views"]["text"] == {"label": "joy", "confidence": 0.6}
    assert event["views"]["audio"] == {"label": "anger", "confidence": 0.7}
