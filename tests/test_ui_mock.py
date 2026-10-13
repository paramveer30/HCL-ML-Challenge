import asyncio

from ui.pipeline_adapter import LABELS, MOCK_SCENARIOS, TIMING_STAGES, MockPipeline


async def no_sleep(seconds):
    return None


async def chunks(n):
    for _ in range(n):
        yield b"\0" * 1024


async def collect(events):
    return [event async for event in events]


def run_clip(name, monkeypatch):
    monkeypatch.setattr("ui.pipeline_adapter.asyncio.sleep", no_sleep)
    return asyncio.run(collect(MockPipeline(seed=0).run_clip(name)))


def test_label_order_is_fixed():
    assert LABELS == ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]


def test_a_mock_turn_sends_every_kind_of_event_in_a_sensible_order(monkeypatch):
    events = run_clip("good_news.wav", monkeypatch)
    kinds = [event["type"] for event in events]
    for kind in ["status", "turn_start", "transcript", "emotion", "reply_delta", "reply_done", "speaking", "timing"]:
        assert kind in kinds
    assert kinds.index("transcript") < kinds.index("emotion") < kinds.index("reply_delta") < kinds.index("reply_done")
    assert [e["state"] for e in events if e["type"] == "status"] == ["hearing", "thinking", "speaking", "idle"]
    assert [e["on"] for e in events if e["type"] == "speaking"] == [True, False]
    assert [e["stage"] for e in events if e["type"] == "timing"] == TIMING_STAGES


def test_emotion_event_has_four_full_distributions(monkeypatch):
    emotion = next(e for e in run_clip("good_news.wav", monkeypatch) if e["type"] == "emotion")
    for group in ["fused", "text", "audio", "vision"]:
        assert list(emotion[group]) == LABELS
        assert abs(sum(emotion[group].values()) - 1.0) < 0.02
    assert emotion["top"] == "joy"
    assert emotion["confidence"] == emotion["fused"]["joy"]
    assert emotion["disagree"] is False


def test_some_scenarios_disagree(monkeypatch):
    emotion = next(e for e in run_clip("its_fine.wav", monkeypatch) if e["type"] == "emotion")
    assert emotion["disagree"] is True


def test_the_streamed_reply_adds_up_to_the_whole_reply(monkeypatch):
    events = run_clip("long_day.wav", monkeypatch)
    reply = "".join(e["text"] for e in events if e["type"] == "reply_delta")
    assert reply == MOCK_SCENARIOS["long_day.wav"][3]


def test_a_microphone_turn_reports_hearing_once(monkeypatch):
    monkeypatch.setattr("ui.pipeline_adapter.asyncio.sleep", no_sleep)
    events = asyncio.run(collect(MockPipeline(seed=1).run_turn(chunks(5))))
    assert [e["state"] for e in events if e["type"] == "status"].count("hearing") == 1
    assert any(e["type"] == "transcript" for e in events)


def test_mock_lists_its_sample_clips():
    assert MockPipeline().list_clips() == list(MOCK_SCENARIOS)
