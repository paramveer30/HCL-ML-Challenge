"""The one interface the web UI talks to, with a mock that fakes a full turn.

Events are plain dicts, sent to the browser as JSON:
  {"type": "status", "state": "idle|listening|hearing|thinking|speaking"}
  {"type": "turn_start", "turn_id": int}
  {"type": "transcript", "turn_id": int, "text": str}
  {"type": "emotion", "turn_id": int, "fused": {...}, "text": {...}, "audio": {...}, "vision": {...},
   "top": str, "confidence": float, "disagree": bool, "face_disagree": bool, "faces": int}
  {"type": "reply_delta", "turn_id": int, "text": str}
  {"type": "reply_done", "turn_id": int}
  {"type": "speaking", "on": bool}
  {"type": "timing", "turn_id": int, "stage": str, "ms": int}
  {"type": "notice", "text": str}
"""
import asyncio
import os
import random
import threading
import time
from pathlib import Path

import cv2
import numpy as np

LABELS = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]
TIMING_STAGES = ["transcript", "emotion", "first_word", "first_audio"]
SAMPLE_RATE = 16000


class PipelineAdapter:
    """What the UI needs from a pipeline. The real one and the mock both follow this."""

    def list_clips(self):
        """Names of sample clips that can be replayed."""
        raise NotImplementedError

    async def run_turn(self, audio_chunks):
        """audio_chunks: an async iterator of bytes, each 512 samples of 16 kHz mono 16-bit PCM.
        Yields events for one turn, in the format at the top of this file."""
        raise NotImplementedError
        yield

    async def run_clip(self, name):
        """Replays one sample clip by name. Yields the same events as run_turn."""
        raise NotImplementedError
        yield

    def reset(self):
        """Forgets the conversation so far."""
        raise NotImplementedError


def probs(**given):
    """A full 7-way distribution from a few named values; the rest is shared out evenly."""
    rest = (1.0 - sum(given.values())) / (len(LABELS) - len(given))
    return {label: round(given.get(label, rest), 3) for label in LABELS}


# name -> (what was said, text model view, audio model view, reply)
MOCK_SCENARIOS = {
    "good_news.wav": (
        "I got the job!",
        probs(joy=0.82, surprise=0.09), probs(joy=0.64, surprise=0.17),
        "That's fantastic news! You must be so happy.",
    ),
    "its_fine.wav": (
        "It's fine.",
        probs(neutral=0.5, joy=0.3), probs(sadness=0.66, neutral=0.2),
        "You say it's fine, but you sound a little down. Is everything okay?",
    ),
    "another_meeting.wav": (
        "Oh great, another meeting.",
        probs(joy=0.52, neutral=0.24), probs(anger=0.57, neutral=0.2),
        "You sound annoyed, even though your words are cheerful. Rough day?",
    ),
    "what_was_that.wav": (
        "What was that noise?",
        probs(surprise=0.55, fear=0.25), probs(fear=0.52, surprise=0.3),
        "That startled you, didn't it? I'm right here with you.",
    ),
    "long_day.wav": (
        "I had a really long day.",
        probs(sadness=0.56, neutral=0.27), probs(sadness=0.5, neutral=0.34),
        "That sounds tiring. Do you want to tell me about it?",
    ),
    "hello.wav": (
        "Hello there.",
        probs(neutral=0.78, joy=0.14), probs(neutral=0.66, joy=0.2),
        "Hello! It's good to hear your voice.",
    ),
}
MOCK_ALPHA = 0.45          # weight on the text model, as in the real system
MOCK_CONFLICT_MIN = 0.5


def top_label(distribution):
    return max(distribution, key=distribution.get)


class MockPipeline(PipelineAdapter):
    """Fakes realistic events with small delays, so the UI can be built and shown without any model."""

    def __init__(self, seed=None):
        self.turn_id = 0
        self.random = random.Random(seed)

    def list_clips(self):
        return list(MOCK_SCENARIOS)

    def reset(self):
        pass

    async def run_turn(self, audio_chunks):
        heard = False
        async for _ in audio_chunks:
            if not heard:
                heard = True
                yield {"type": "status", "state": "hearing"}
        async for event in self.play(self.random.choice(list(MOCK_SCENARIOS))):
            yield event

    async def run_clip(self, name):
        yield {"type": "status", "state": "hearing"}
        await asyncio.sleep(0.6)
        async for event in self.play(name if name in MOCK_SCENARIOS else "hello.wav"):
            yield event

    async def play(self, name):
        transcript, p_text, p_audio, reply = MOCK_SCENARIOS[name]
        self.turn_id += 1
        turn = self.turn_id
        jitter = lambda ms: int(ms * self.random.uniform(0.85, 1.2))

        yield {"type": "turn_start", "turn_id": turn}
        yield {"type": "status", "state": "thinking"}

        asr_ms = jitter(420)
        await asyncio.sleep(asr_ms / 1000)
        yield {"type": "transcript", "turn_id": turn, "text": transcript}
        yield {"type": "timing", "turn_id": turn, "stage": "transcript", "ms": asr_ms}

        emotion_ms = jitter(60)
        await asyncio.sleep(emotion_ms / 1000)
        fused = {k: round(MOCK_ALPHA * p_text[k] + (1 - MOCK_ALPHA) * p_audio[k], 3) for k in LABELS}
        disagree = (top_label(p_text) != top_label(p_audio)
                    and max(p_text.values()) >= MOCK_CONFLICT_MIN and max(p_audio.values()) >= MOCK_CONFLICT_MIN)
        # the mock face simply agrees with the tone of voice
        yield {"type": "emotion", "turn_id": turn, "fused": fused, "text": p_text, "audio": p_audio, "vision": p_audio,
               "top": top_label(fused), "confidence": fused[top_label(fused)], "disagree": disagree,
               "face_disagree": disagree, "faces": 6}
        yield {"type": "timing", "turn_id": turn, "stage": "emotion", "ms": emotion_ms}

        first_word_ms = jitter(180)
        await asyncio.sleep(first_word_ms / 1000)
        yield {"type": "timing", "turn_id": turn, "stage": "first_word", "ms": first_word_ms}
        yield {"type": "status", "state": "speaking"}
        words = reply.split(" ")
        for i, word in enumerate(words):
            yield {"type": "reply_delta", "turn_id": turn, "text": word + ("" if i == len(words) - 1 else " ")}
            if i == 3:
                # the voice starts once the first few words exist
                yield {"type": "speaking", "on": True}
                yield {"type": "timing", "turn_id": turn, "stage": "first_audio",
                       "ms": asr_ms + emotion_ms + first_word_ms + jitter(240)}
            await asyncio.sleep(0.09)
        yield {"type": "reply_done", "turn_id": turn}
        await asyncio.sleep(0.9)
        yield {"type": "speaking", "on": False}
        yield {"type": "status", "state": "idle"}


class ProbabilityRecorder:
    """Sits in front of the emotion classifier and remembers the last probabilities each model gave.

    The pipeline's state only keeps each model's top label, and the page shows all 7 numbers.
    """

    def __init__(self, classifier):
        self.classifier = classifier
        self.last_text = None
        self.last_audio = None
        self.last_vision = None

    def text_probs(self, context, target):
        self.last_text = self.classifier.text_probs(context, target)
        return self.last_text

    def audio_probs(self, wav):
        self.last_audio = self.classifier.audio_probs(wav)
        return self.last_audio

    def face_layers(self, frame):
        return self.classifier.face_layers(frame)

    def vision_probs(self, layers):
        self.last_vision = self.classifier.vision_probs(layers)
        return self.last_vision


def sample_labels(name):
    """The true emotions a sample's file name starts with, one per spoken line:
    "fear_dia10_utt2.wav" gives ["fear"], "neutral_joy_dia3.wav" gives ["neutral", "joy"]."""
    labels = []
    for part in Path(name).stem.split("_"):
        if part not in LABELS:
            break
        labels.append(part)
    return labels


async def in_thread(make_events):
    """Runs a blocking generator in a worker thread and hands its events to the async side one by one."""
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()
    finished = object()

    def work():
        try:
            for event in make_events():
                loop.call_soon_threadsafe(queue.put_nowait, event)
            loop.call_soon_threadsafe(queue.put_nowait, finished)
        except Exception as error:
            loop.call_soon_threadsafe(queue.put_nowait, error)

    threading.Thread(target=work, daemon=True).start()
    while True:
        event = await queue.get()
        if event is finished:
            return
        if isinstance(event, Exception):
            raise event
        yield event


class RealPipeline(PipelineAdapter):
    """The real models behind the same interface. They are loaded the first time they are needed.

    A turn from the talk button is the audio between press and release. A replayed clip is pushed
    through the voice detector and the endpointer in 32 ms frames at real-time pace, like a microphone.
    The reply is spoken through this computer's speakers, the same way scripts/live.py does it.
    """

    def __init__(self, clips_dir="ui/clips", voice=True):
        self.clips_dir = Path(clips_dir)
        self.voice = voice
        self.pipeline = None
        self.speaker = None
        self.recorder = None
        self.lock = threading.Lock()        # one turn at a time, even with two browser tabs open

    def list_clips(self):
        return sorted(p.name for p in self.clips_dir.glob("*.wav"))

    def reset(self):
        if self.pipeline is not None:
            self.pipeline.reset()

    def add_frame(self, jpeg):
        """jpeg: one camera image as JPEG bytes. The face in it, if any, is kept for the turn being spoken."""
        if self.pipeline is None:
            return
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            self.pipeline.push_image(frame)

    def video_of(self, name):
        """The video file that belongs to a sample: same name, .mp4. None if there is none."""
        path = self.clips_dir / (Path(name).stem + ".mp4")
        return path if path.exists() else None

    async def prepare(self):
        """Loads the models ahead of the first turn."""
        async for event in in_thread(self.prepare_events):
            yield event

    async def run_turn(self, audio_chunks):
        chunks = []
        async for chunk in audio_chunks:
            if not chunks:
                yield {"type": "status", "state": "hearing"}
            chunks.append(chunk)
        wav = np.frombuffer(b"".join(chunks), dtype=np.int16).astype(np.float32) / 32768
        async for event in in_thread(lambda: self.turn_events(wav)):
            yield event

    async def run_clip(self, name):
        async for event in in_thread(lambda: self.clip_events(name)):
            yield event

    # ---- everything below runs in a worker thread ----

    def load(self):
        """Loads every model once. Yields notices so the page can say what is going on."""
        if self.pipeline is not None:
            return
        yield {"type": "status", "state": "thinking"}
        yield {"type": "notice", "text": "loading the models. this takes a moment the first time."}
        os.environ["HF_HUB_OFFLINE"] = "1"          # the live path never touches the network
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from meldrt.pipeline.build import build_pipeline, build_speaker

        pipeline = build_pipeline(sample_rate=SAMPLE_RATE)
        self.recorder = ProbabilityRecorder(pipeline.classifier)
        pipeline.classifier = self.recorder
        pipeline.warm_up()
        self.speaker = build_speaker() if self.voice else None
        self.pipeline = pipeline
        yield {"type": "notice", "text": "models ready."}

    def prepare_events(self):
        with self.lock:
            yield from self.load()
            yield {"type": "status", "state": "idle"}

    def turn_events(self, wav):
        """One push-to-talk turn: the audio between press and release is the turn."""
        with self.lock:
            yield from self.load()
            yield {"type": "status", "state": "thinking"}
            # under a quarter of a second is a tap on the button, not a sentence
            state = self.pipeline.process_turn(wav, endpoint_ms=0.0) if len(wav) >= SAMPLE_RATE // 4 else None
            if state is None:
                yield {"type": "notice", "text": "I did not catch any words. hold the button while you speak."}
            else:
                yield from self.answer(state)
            yield {"type": "status", "state": "idle"}

    def clip_events(self, name):
        """One sample clip, fed in 32 ms frames at the speed a microphone would deliver them."""
        import soundfile as sf

        with self.lock:
            yield from self.load()
            wav, sample_rate = sf.read(self.clips_dir / Path(name).name, dtype="float32")
            if wav.ndim != 1 or sample_rate != SAMPLE_RATE:
                yield {"type": "notice", "text": "this clip is not 16 kHz mono audio."}
                yield {"type": "status", "state": "idle"}
                return
            # one second of silence after the clip, so the endpointer can hear that the turn is over
            wav = np.concatenate([wav, np.zeros(SAMPLE_RATE, dtype=np.float32)])
            # a sample is its own conversation: the sample played before it must not become its context
            self.pipeline.reset()
            yield {"type": "status", "state": "hearing"}
            self.play_out_loud(wav)
            # if the sample has a video, its frames go in at the moment they belong, like a camera
            video = self.video_of(name)
            pictures = self.pipeline.read_video(video) if video is not None else []
            n = self.pipeline.frame_samples
            said = []
            for start in range(0, len(wav) - n + 1, n):
                frame_start = time.perf_counter()
                while pictures and pictures[0][0] <= start / SAMPLE_RATE:
                    self.pipeline.push_image(pictures.pop(0)[1])
                state = self.pipeline.push_frame(wav[start : start + n])
                if state is not None:
                    said.append(state.label)
                    yield from self.answer(state)
                    # the robot's voice cut the clip off; carry on from where the pipeline is
                    self.play_out_loud(wav[start + n :])
                time.sleep(max(0.0, n / SAMPLE_RATE - (time.perf_counter() - frame_start)))
            if not said:
                yield {"type": "notice", "text": "no speech found in this clip."}
            expected = sample_labels(name)
            if expected and len(expected) == len(said):
                yield {"type": "notice", "text": f"sample label: {', '.join(expected)}. the robot said: {', '.join(said)}."}
            elif expected and said:
                yield {"type": "notice", "text": f"this sample has {len(expected)} labelled lines but the robot heard "
                                                 f"{len(said)} turns, so they are not compared."}
            yield {"type": "status", "state": "idle"}

    def play_out_loud(self, wav):
        """Plays clip audio through the speakers without waiting, so the person watching hears what the robot hears."""
        if self.speaker is not None and len(wav):
            import sounddevice as sd

            # no speakers plugged in must not stop the turn; the robot's own voice handles this the same way
            try:
                sd.play(wav, SAMPLE_RATE, device=self.speaker.output_device)
            except Exception as error:
                print(f"[page] could not play the clip out loud: {error}", flush=True)

    def answer(self, state):
        """Turns one finished state into page events, then streams and speaks the reply."""
        turn = state.turn_id
        p_text = {label: float(p) for label, p in zip(LABELS, self.recorder.last_text)}
        if state.audio_view is None:
            p_audio = {label: 0.0 for label in LABELS}      # the turn was too short for the audio model
        else:
            p_audio = {label: float(p) for label, p in zip(LABELS, self.recorder.last_audio)}
        state_ms = state.latency_ms["state"]

        yield {"type": "turn_start", "turn_id": turn}
        yield {"type": "status", "state": "thinking"}
        yield {"type": "transcript", "turn_id": turn, "text": state.transcript}
        yield {"type": "timing", "turn_id": turn, "stage": "transcript", "ms": round(state.latency_ms["asr"])}
        if state.vision_view is None:
            p_vision = {label: 0.0 for label in LABELS}     # no face was seen during the turn
        else:
            p_vision = {label: float(p) for label, p in zip(LABELS, self.recorder.last_vision)}
        yield {"type": "emotion", "turn_id": turn, "fused": state.probs, "text": p_text, "audio": p_audio, "vision": p_vision,
               "top": state.label, "confidence": state.confidence, "disagree": state.tone_text_conflict,
               "face_disagree": state.face_text_conflict, "faces": state.faces_seen}
        # the audio model runs next to speech recognition, so this is the time added after the transcript
        yield {"type": "timing", "turn_id": turn, "stage": "emotion",
               "ms": round(max(0.0, state_ms - state.latency_ms["asr"]))}

        events = self.pipeline.reply(state)
        if self.speaker is not None:
            from meldrt.pipeline.speak import speak_while_streaming

            events = speak_while_streaming(events, self.speaker)
        first = True
        for event in events:
            if event["type"] != "token":
                continue
            if first:
                first = False
                first_word_ms = (time.perf_counter() - self.pipeline.turn_ended_at) * 1000 - state_ms
                yield {"type": "timing", "turn_id": turn, "stage": "first_word", "ms": round(max(0.0, first_word_ms))}
                yield {"type": "status", "state": "speaking"}
                yield {"type": "speaking", "on": True}
            yield {"type": "reply_delta", "turn_id": turn, "text": event["text"]}
        yield {"type": "reply_done", "turn_id": turn}

        if self.speaker is not None:
            first_audio_at = self.speaker.wait()
            if first_audio_at is not None:
                yield {"type": "timing", "turn_id": turn, "stage": "first_audio",
                       "ms": round((first_audio_at - self.pipeline.turn_ended_at) * 1000)}
        yield {"type": "speaking", "on": False}
