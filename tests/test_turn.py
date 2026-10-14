import time

import numpy as np
import pytest

from meldrt.labels import LABELS
from meldrt.pipeline.turn import Pipeline
from meldrt.schema import TurnState

CFG = {
    "audio": {"sample_rate": 16000},
    "data": {"context_k": 2},
    "fusion": {"conflict_min_conf": 0.4},
    "vision": {"frames_per_clip": 8},
    "reply": {"low_conf": 0.5},
    "pipeline": {
        "frame_samples": 512, "start_ms": 250, "end_silence_ms": 600, "max_turn_s": 15,
        "preroll_ms": 200, "history_turns": 3, "speaker_name": "Speaker", "input_device": None,
        "warm_during_silence": False, "max_faces": 64,
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
        self.p_vision = None
        self.calls = []
        self.audio_lengths = []

    def text_probs(self, context, target):
        self.calls.append((context, target))
        return self.p_text

    def audio_probs(self, wav):
        self.audio_lengths.append(len(wav))
        return self.p_audio

    def look(self, frame, from_camera):
        # a frame that is all zeros has no face in it
        return "a face" if frame.any() else None

    def vision_probs(self, faces, from_camera):
        self.faces_used, self.from_camera = len(faces), from_camera
        return self.p_vision


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
    assert (state.text_view.label, state.text_view.confidence) == ("joy", pytest.approx(0.6))
    assert (state.audio_view.label, state.audio_view.confidence) == ("anger", pytest.approx(0.7))
    assert state.audio_view.probs["anger"] == pytest.approx(0.7) and len(state.audio_view.probs) == 7
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
    # the prompt holds the earlier turn as chat history and the current turn last
    messages = prompts[0]
    assert {"role": "user", "content": "hello"} in messages
    assert messages[-1]["role"] == "user" and messages[-1]["content"].endswith("how are you")


def test_a_reply_that_is_cut_short_is_still_remembered(monkeypatch):
    monkeypatch.setattr("meldrt.pipeline.turn.stream_reply", lambda prompt, reply_cfg: iter(["Hi ", "there", "!"]))
    pipeline, _, _ = make(ONE_TURN, ["hello"])
    state = run(pipeline, len(ONE_TURN))[-1]
    reply = pipeline.reply(state)
    next(reply)
    next(reply)
    reply.close()                                 # the listener stops: the person cut in
    assert pipeline.history[-1]["reply"] == "Hi there"


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
    assert set(event) == {"type", "turn_id", "transcript", "emotion", "views", "tone_text_conflict", "face_text_conflict",
                          "faces_seen", "audio_seconds", "latency_ms"}
    assert set(event["emotion"]) == {"label", "confidence", "probs"}
    assert event["views"]["text"] == {"label": "joy", "confidence": 0.6}
    assert event["views"]["audio"] == {"label": "anger", "confidence": 0.7}


def test_the_audio_model_is_kept_busy_only_while_waiting_for_the_end_of_a_turn():
    cfg = {**CFG, "pipeline": {**CFG["pipeline"], "warm_during_silence": True}}
    classifier = FakeClassifier(JOY, ANGER)
    pipeline = Pipeline(FakeVad(ONE_TURN), FakeAsr(["hello"]), classifier, cfg, alpha=0.5, sample_rate=16000)
    states = run(pipeline, len(ONE_TURN))
    # 18 quiet frames inside the turn each get half a second of silence; the 19th ends the turn
    assert classifier.audio_lengths[:18] == [8000] * 18
    # then the real turn: 20 speech frames plus the lead-in kept from before it
    assert len(classifier.audio_lengths) == 19 and classifier.audio_lengths[18] > 8000
    assert states[-1].audio_view.label == "anger"


def test_no_extra_work_when_warming_is_off():
    pipeline, _, classifier = make(ONE_TURN, ["hello"])
    run(pipeline, len(ONE_TURN))
    assert len(classifier.audio_lengths) == 1


# --- vision ---

SADNESS = probs((4, 0.8), (0, 0.2))
FACE = np.ones((120, 160, 3), dtype=np.uint8)
NO_FACE = np.zeros((120, 160, 3), dtype=np.uint8)


def make_with_vision(weights3=(0.5, 0.25, 0.25), p_text=JOY, p_audio=ANGER, p_vision=SADNESS):
    classifier = FakeClassifier(p_text, p_audio)
    classifier.p_vision = p_vision
    pipeline = Pipeline(FakeVad(ONE_TURN), FakeAsr(["hello"]), classifier, CFG, alpha=0.5, sample_rate=16000, weights3=weights3)
    return pipeline, classifier


def test_without_vision_images_are_ignored_and_the_state_is_unchanged():
    pipeline, _, _ = make(ONE_TURN, ["hello"])
    pipeline.push_image(FACE)
    state = run(pipeline, len(ONE_TURN))[-1]
    assert state.vision_view is None and state.faces_seen == 0 and not state.face_text_conflict
    assert state.probs["joy"] == pytest.approx(0.5 * 0.6)          # text and audio only, alpha 0.5


def test_faces_seen_during_the_turn_join_the_fusion():
    pipeline, classifier = make_with_vision()
    for _ in range(3):
        pipeline.push_image(FACE)
    pipeline.push_image(NO_FACE)
    state = run(pipeline, len(ONE_TURN))[-1]
    assert classifier.faces_used == 3 and state.faces_seen == 3
    assert (state.vision_view.label, state.vision_view.confidence) == ("sadness", pytest.approx(0.8))
    # by hand: 0.5 text + 0.25 audio + 0.25 vision
    assert state.probs["joy"] == pytest.approx(0.5 * 0.6)
    assert state.probs["anger"] == pytest.approx(0.25 * 0.7)
    assert state.probs["sadness"] == pytest.approx(0.25 * 0.8)
    assert state.probs["neutral"] == pytest.approx(0.5 * 0.4 + 0.25 * 0.3 + 0.25 * 0.2)
    assert sum(state.probs.values()) == pytest.approx(1.0)
    assert "vision" in state.latency_ms


def test_a_face_from_before_the_turn_is_not_used():
    pipeline, _ = make_with_vision()
    pipeline.faces.append((time.perf_counter() - 60, "an old face", True))
    state = run(pipeline, len(ONE_TURN))[-1]
    assert state.vision_view is None and state.faces_seen == 0
    assert state.probs["joy"] == pytest.approx(0.5 * 0.6)          # falls back to text and audio


def test_face_and_words_conflict_is_flagged_on_its_own():
    pipeline, _ = make_with_vision(p_audio=JOY)
    pipeline.push_image(FACE)
    state = run(pipeline, len(ONE_TURN))[-1]
    assert state.face_text_conflict and not state.tone_text_conflict


def test_voice_and_face_agreeing_against_the_words_flags_both_even_below_the_threshold():
    # happy words, and a voice and a face that both lean angry without being sure (under 0.4 each)
    unsure_anger = probs((3, 0.35), (0, 0.3), (4, 0.2), (1, 0.15))
    pipeline, _ = make_with_vision(p_text=JOY, p_audio=unsure_anger, p_vision=unsure_anger)
    pipeline.push_image(FACE)
    state = run(pipeline, len(ONE_TURN))[-1]
    assert state.tone_text_conflict and state.face_text_conflict
    # the fused label still follows the weights: 0.5 words, 0.25 voice, 0.25 face
    assert state.label == "joy"


def test_too_short_for_audio_uses_text_in_its_place_next_to_the_face():
    pipeline, _ = make_with_vision(p_audio=None)
    pipeline.push_image(FACE)
    state = run(pipeline, len(ONE_TURN))[-1]
    assert state.audio_view is None
    assert state.probs["joy"] == pytest.approx(0.75 * 0.6)
    assert state.probs["sadness"] == pytest.approx(0.25 * 0.8)


def test_the_state_event_carries_the_face_view():
    pipeline, _ = make_with_vision()
    pipeline.push_image(FACE)
    event = run(pipeline, len(ONE_TURN))[-1].to_event()
    assert event["views"]["vision"] == {"label": "sadness", "confidence": 0.8}
    assert event["face_text_conflict"] is True and event["faces_seen"] == 1


def test_the_classifier_is_told_whether_the_faces_came_from_a_camera():
    pipeline, classifier = make_with_vision()
    pipeline.push_image(FACE)
    run(pipeline, len(ONE_TURN))
    assert classifier.from_camera is False        # a clip's video
    pipeline, classifier = make_with_vision()
    pipeline.push_image(FACE, from_camera=True)
    run(pipeline, len(ONE_TURN))
    assert classifier.from_camera is True


def test_faces_are_used_for_one_turn_only():
    classifier = FakeClassifier(JOY, ANGER)
    classifier.p_vision = SADNESS
    pipeline = Pipeline(FakeVad(ONE_TURN * 2), FakeAsr(["hello", "again"]), classifier, CFG, alpha=0.5, sample_rate=16000,
                        weights3=(0.5, 0.25, 0.25))
    pipeline.push_image(FACE)
    first, second = [s for s in run(pipeline, len(ONE_TURN) * 2) if s]
    assert first.faces_seen == 1 and second.faces_seen == 0


def test_pictures_from_two_sources_in_one_turn_are_not_mixed():
    pipeline, classifier = make_with_vision()
    pipeline.push_image(FACE)                        # a clip's picture
    pipeline.push_image(FACE, from_camera=True)      # and a camera's, for example from a second browser tab
    pipeline.push_image(FACE, from_camera=True)
    state = run(pipeline, len(ONE_TURN))[-1]
    assert classifier.from_camera is True and classifier.faces_used == 2 and state.faces_seen == 2


def test_push_image_hands_back_what_it_saw():
    pipeline, _ = make_with_vision()
    assert pipeline.push_image(FACE) == "a face"
    assert pipeline.push_image(NO_FACE) is None


def test_the_temperature_changes_the_confidence_but_not_the_label():
    plain, _, _ = make(ONE_TURN, ["hello"])
    sharp = Pipeline(FakeVad(ONE_TURN), FakeAsr(["hello"]), FakeClassifier(JOY, ANGER), CFG, alpha=0.5,
                     sample_rate=16000, temperature=0.5)
    before, after = run(plain, len(ONE_TURN))[-1], run(sharp, len(ONE_TURN))[-1]
    # fused is neutral 0.35, anger 0.35, joy 0.30; at temperature 0.5 every value is squared and rescaled
    squares = {"neutral": 0.35 ** 2, "anger": 0.35 ** 2, "joy": 0.30 ** 2}
    assert after.label == before.label
    assert after.probs["joy"] == pytest.approx(squares["joy"] / sum(squares.values()))
    assert after.confidence > before.confidence


def test_keep_warm_runs_the_audio_model_on_half_a_second_of_silence():
    pipeline, _, classifier = make(ONE_TURN, ["hello"])
    pipeline.keep_warm()
    assert classifier.audio_lengths == [8000]


def test_clearing_audio_also_forgets_the_faces():
    pipeline, _ = make_with_vision()
    pipeline.push_image(FACE)
    pipeline.clear_audio()
    assert len(pipeline.faces) == 0
