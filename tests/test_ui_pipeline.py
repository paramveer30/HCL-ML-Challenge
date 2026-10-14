import asyncio
import time
from types import SimpleNamespace

import cv2
import numpy as np
import soundfile as sf

from meldrt.data.video import read_timed_frames
from meldrt.schema import TurnState, View
from ui.pipeline import LABELS, TIMING_STAGES, PagePipeline, sample_labels

P_TEXT = np.array([0.1, 0.6, 0.1, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
P_AUDIO = np.array([0.2, 0.1, 0.1, 0.1, 0.4, 0.05, 0.05], dtype=np.float32)
JOY_FACE = np.array([0.05, 0.8, 0.03, 0.03, 0.03, 0.03, 0.03], dtype=np.float32)


def as_dict(probs):
    return {name: float(p) for name, p in zip(LABELS, probs)}


class FakePipeline:
    """Stands in for the live pipeline: every turn says the same thing, no models."""

    frame_samples = 512

    def __init__(self, short=False):
        self.short = short          # True: the turn is too short for the audio model
        self.turn_id = 0
        self.turn_ended_at = 0.0
        self.frames = 0
        self.resets = 0
        self.cleared = 0
        self.images = []
        self.from_camera = None
        self.warmed = 0
        self.slow_reply = False     # True: pause between pieces, like a real model writing

    def process_turn(self, wav, endpoint_ms):
        self.turn_id += 1
        fused = 0.45 * P_TEXT + 0.55 * P_AUDIO
        return TurnState(
            turn_id=self.turn_id,
            transcript="I got the job!",
            label=LABELS[int(fused.argmax())],
            confidence=float(fused.max()),
            probs=as_dict(fused),
            text_view=View("joy", 0.6, as_dict(P_TEXT)),
            audio_view=None if self.short else View("sadness", 0.4, as_dict(P_AUDIO)),
            tone_text_conflict=False,
            audio_seconds=len(wav) / 16000,
            latency_ms={"endpoint": endpoint_ms, "asr": 300.0, "audio": 40.0, "text": 20.0, "state": 320.0},
        )

    def push_frame(self, frame):
        self.frames += 1
        return self.process_turn(frame, 600.0) if self.frames == 3 else None

    def reply(self, state):
        for piece in ["That is ", "great ", "news."]:
            if self.slow_reply:
                time.sleep(0.2)
            yield {"type": "token", "turn_id": state.turn_id, "text": piece}
        yield {"type": "done", "turn_id": state.turn_id, "reply": "That is great news.", "latency_ms": {}}

    def push_image(self, frame, from_camera=False):
        self.images.append(frame)
        self.from_camera = from_camera
        # a face in the middle of any picture that is not black
        if not frame.any():
            return None
        return SimpleNamespace(box=(40, 30, 80, 60), points=[(60, 50), (100, 50), (80, 60), (65, 75), (95, 75)],
                               expression=JOY_FACE)

    def read_video(self, path):
        return read_timed_frames(path, 8, 15)

    def keep_warm(self):
        self.warmed += 1

    def clear_audio(self):
        self.cleared += 1

    def reset(self):
        self.resets += 1


def page_with_fakes(tmp_path, short=False):
    page = PagePipeline(clips_dir=tmp_path, voice=False)
    page.pipeline = FakePipeline(short)
    return page


async def audio(seconds):
    for _ in range(int(seconds * 16000 / 512)):
        yield b"\0" * 1024


async def collect(events):
    return [event async for event in events]


# --- a turn from the talk button ---

def test_a_turn_sends_its_events_in_a_sensible_order(tmp_path):
    events = asyncio.run(collect(page_with_fakes(tmp_path).run_turn(audio(1.0))))
    kinds = [event["type"] for event in events]
    assert kinds.index("transcript") < kinds.index("emotion") < kinds.index("reply_delta") < kinds.index("reply_done")
    assert [e["state"] for e in events if e["type"] == "status"] == ["hearing", "thinking", "thinking", "speaking", "idle"]
    assert "".join(e["text"] for e in events if e["type"] == "reply_delta") == "That is great news."
    # no voice here, so there is no first_audio timing
    assert [e["stage"] for e in events if e["type"] == "timing"] == TIMING_STAGES[:3]


def test_the_emotion_event_carries_each_models_own_probabilities(tmp_path):
    events = asyncio.run(collect(page_with_fakes(tmp_path).run_turn(audio(1.0))))
    emotion = next(e for e in events if e["type"] == "emotion")
    for name in ["text", "audio", "fused"]:
        assert list(emotion[name]) == LABELS
    assert abs(emotion["text"]["joy"] - 0.6) < 1e-6
    assert abs(emotion["audio"]["sadness"] - 0.4) < 1e-6
    assert emotion["top"] == "joy" and abs(emotion["confidence"] - emotion["fused"]["joy"]) < 1e-6


def test_a_turn_too_short_for_the_audio_model_shows_zeros_for_it(tmp_path):
    events = asyncio.run(collect(page_with_fakes(tmp_path, short=True).run_turn(audio(1.0))))
    emotion = next(e for e in events if e["type"] == "emotion")
    assert set(emotion["audio"].values()) == {0.0}


def test_a_turn_without_a_face_shows_zeros_for_the_face_model(tmp_path):
    events = asyncio.run(collect(page_with_fakes(tmp_path).run_turn(audio(1.0))))
    emotion = next(e for e in events if e["type"] == "emotion")
    assert list(emotion["vision"]) == LABELS and set(emotion["vision"].values()) == {0.0}
    assert emotion["faces"] == 0 and emotion["face_disagree"] is False


def test_a_tap_on_the_button_is_not_a_turn(tmp_path):
    events = asyncio.run(collect(page_with_fakes(tmp_path).run_turn(audio(0.1))))
    assert [e["type"] for e in events] == ["status", "status", "notice", "status"]


def test_the_gpu_is_kept_busy_while_the_button_is_held(tmp_path):
    page = page_with_fakes(tmp_path)
    asyncio.run(collect(page.run_turn(audio(1.0))))
    assert page.pipeline.warmed >= 1


# --- sample clips ---

def replay(tmp_path, monkeypatch, name):
    monkeypatch.setattr("ui.pipeline.time.sleep", lambda seconds: None)
    sf.write(tmp_path / name, np.zeros(16000, dtype=np.float32), 16000)
    page = page_with_fakes(tmp_path)
    return page, asyncio.run(collect(page.run_clip(name)))


def test_a_clip_goes_through_the_pipeline_frame_by_frame(tmp_path, monkeypatch):
    page, events = replay(tmp_path, monkeypatch, "hello.wav")
    assert page.list_clips() == ["hello.wav"]
    assert page.pipeline.frames == 62          # 1 s of clip + 1 s of added silence, in 512-sample frames
    assert [e["text"] for e in events if e["type"] == "transcript"] == ["I got the job!"]
    assert events[-1] == {"type": "status", "state": "idle"}


def test_each_sample_starts_a_new_conversation(tmp_path, monkeypatch):
    page, _ = replay(tmp_path, monkeypatch, "hello.wav")
    assert page.pipeline.resets == 1
    assert page.pipeline.cleared == 1          # and leaves no audio or faces behind for the next turn


def test_sample_labels_come_from_the_start_of_the_file_name():
    assert sample_labels("fear_dia10_utt2.wav") == ["fear"]
    assert sample_labels("neutral_joy_dia3.wav") == ["neutral", "joy"]
    assert sample_labels("hello.wav") == []


def test_a_labelled_sample_says_what_was_expected_and_what_came_out(tmp_path, monkeypatch):
    _, events = replay(tmp_path, monkeypatch, "fear_dia10_utt2.wav")
    notices = [e["text"] for e in events if e["type"] == "notice"]
    assert notices == ["sample label: fear. the robot said: joy."]


def test_labels_are_not_compared_when_the_turn_count_differs(tmp_path, monkeypatch):
    _, events = replay(tmp_path, monkeypatch, "neutral_joy_dia3.wav")      # two labels, the fake hears one turn
    notices = [e["text"] for e in events if e["type"] == "notice"]
    assert notices == ["this sample has 2 labelled lines but the robot heard 1 turns, so they are not compared."]


def test_an_unlabelled_sample_adds_no_label_line(tmp_path, monkeypatch):
    _, events = replay(tmp_path, monkeypatch, "hello.wav")
    assert [e for e in events if e["type"] == "notice"] == []


def test_a_clip_with_the_wrong_sample_rate_is_refused(tmp_path):
    sf.write(tmp_path / "wrong.wav", np.zeros(8000, dtype=np.float32), 8000)
    page = page_with_fakes(tmp_path)
    events = asyncio.run(collect(page.run_clip("wrong.wav")))
    assert page.pipeline.frames == 0
    assert [e["type"] for e in events] == ["notice", "status"]


def test_a_sample_with_a_video_feeds_its_frames_in_as_it_plays(tmp_path, monkeypatch):
    monkeypatch.setattr("ui.pipeline.time.sleep", lambda seconds: None)
    sf.write(tmp_path / "joy_clip.wav", np.zeros(16000, dtype=np.float32), 16000)
    writer = cv2.VideoWriter(str(tmp_path / "joy_clip.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 24, (160, 120))
    for _ in range(24):
        writer.write(np.full((120, 160, 3), 128, dtype=np.uint8))
    writer.release()
    page = page_with_fakes(tmp_path)
    assert page.video_of("joy_clip.wav") == tmp_path / "joy_clip.mp4"
    asyncio.run(collect(page.run_clip("joy_clip.wav")))
    assert len(page.pipeline.images) == 8          # 8 frames spread over the one-second video
    assert page.pipeline.from_camera is False      # a clip's pictures, not a live camera


def test_a_sample_without_a_video_has_none(tmp_path):
    assert page_with_fakes(tmp_path).video_of("hello.wav") is None


def test_clear_forgets_the_conversation(tmp_path):
    page = page_with_fakes(tmp_path)
    page.reset()
    assert page.pipeline.resets == 1


# --- interrupting ---

def test_an_interrupted_reply_stops_after_the_piece_it_was_on(tmp_path):
    page = page_with_fakes(tmp_path)
    page.pipeline.slow_reply = True

    async def talk_over_it():
        events = []
        async for event in page.run_turn(audio(1.0)):
            events.append(event)
            if event["type"] == "reply_delta":
                page.interrupt()                    # the person starts talking as the first words appear
        return events

    events = asyncio.run(talk_over_it())
    said = "".join(e["text"] for e in events if e["type"] == "reply_delta")
    assert said != "That is great news." and said.startswith("That is ")
    assert events[-1] == {"type": "status", "state": "idle"}


def test_the_next_turn_after_an_interruption_runs_normally(tmp_path):
    page = page_with_fakes(tmp_path)
    page.interrupt()
    events = asyncio.run(collect(page.run_turn(audio(1.0))))
    assert "".join(e["text"] for e in events if e["type"] == "reply_delta") == "That is great news."


# --- camera ---

def jpeg(value=128, width=160, height=120):
    ok, data = cv2.imencode(".jpg", np.full((height, width, 3), value, dtype=np.uint8))
    return data.tobytes()


def test_a_camera_image_reaches_the_pipeline_and_the_face_comes_back_as_shares_of_the_picture(tmp_path):
    page = page_with_fakes(tmp_path)
    face = page.look(jpeg())
    assert page.pipeline.images[0].shape == (120, 160, 3) and page.pipeline.from_camera is True
    assert face["type"] == "face" and face["found"] is True
    assert face["box"] == [40 / 160, 30 / 120, 80 / 160, 60 / 120]
    assert face["points"][0] == [60 / 160, 50 / 120] and len(face["points"]) == 5
    assert face["label"] == "joy" and abs(face["confidence"] - 0.8) < 1e-6


def test_a_picture_without_a_face_says_so(tmp_path):
    assert page_with_fakes(tmp_path).look(jpeg(value=0)) == {"type": "face", "found": False}


def test_broken_image_data_is_dropped(tmp_path):
    page = page_with_fakes(tmp_path)
    assert page.look(b"not a picture") is None
    assert page.pipeline.images == []


def test_camera_images_before_the_models_are_loaded_are_ignored(tmp_path):
    assert PagePipeline(clips_dir=tmp_path, voice=False).look(jpeg()) is None


# --- loading ---

def test_models_that_cannot_be_loaded_are_reported_instead_of_crashing(tmp_path):
    page = PagePipeline(clips_dir=tmp_path, voice=False)

    def broken_load():
        raise FileNotFoundError("checkpoints/text/best")
        yield

    page.load = broken_load
    events = asyncio.run(collect(page.prepare()))
    assert [e["type"] for e in events] == ["notice", "notice", "status"]
    assert "checkpoints/text/best" in events[0]["text"]
    assert events[-1] == {"type": "status", "state": "unavailable"}


def test_models_that_are_already_loaded_are_ready_at_once(tmp_path):
    events = asyncio.run(collect(page_with_fakes(tmp_path).prepare()))
    assert events == [{"type": "status", "state": "idle"}]
