"""The real pipeline, driven by the web page. The models are loaded when the first page connects.

A turn from the talk button is the audio between press and release. A sample clip is pushed through
the voice detector and the endpointer in 32 ms frames at real-time pace, like a microphone, with the
pictures of its video (if it has one) going in at the moments they belong. The reply is spoken through
this computer's speakers, the same way scripts/live.py does it.

Events are plain dicts, sent to the browser as JSON:
  {"type": "status", "state": "idle|loading|unavailable|listening|hearing|thinking|speaking"}
  {"type": "turn_start", "turn_id": int}
  {"type": "transcript", "turn_id": int, "text": str}
  {"type": "emotion", "turn_id": int, "fused": {...}, "text": {...}, "audio": {...}, "vision": {...},
   "top": str, "confidence": float, "disagree": bool, "face_disagree": bool, "faces": int}
  {"type": "reply_delta", "turn_id": int, "text": str}
  {"type": "reply_done", "turn_id": int}
  {"type": "timing", "turn_id": int, "stage": str, "ms": int}
  {"type": "face", "found": bool, "box": [x, y, w, h], "points": [[x, y] * 5], "label": str, "confidence": float}
  {"type": "notice", "text": str}
"""
import asyncio
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

LABELS = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]
TIMING_STAGES = ["transcript", "emotion", "first_word", "first_audio"]
SAMPLE_RATE = 16000


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


NO_FACE = {label: 0.0 for label in LABELS}
WARM_EVERY_CHUNKS = 4       # while the talk button is held, the GPU gets a little work every 4 chunks (128 ms)


class PagePipeline:
    def __init__(self, clips_dir="ui/clips", voice=True):
        self.clips_dir = Path(clips_dir)
        self.voice = voice
        self.pipeline = None
        self.speaker = None
        self.lock = threading.Lock()            # one turn at a time, even with two browser tabs open
        self.interrupted = threading.Event()    # set when the person starts talking over the robot
        self.warming = False

    def list_clips(self):
        return sorted(p.name for p in self.clips_dir.glob("*.wav"))

    def video_of(self, name):
        """The video file that belongs to a sample: same name, .mp4. None if there is none."""
        path = self.clips_dir / (Path(name).stem + ".mp4")
        return path if path.exists() else None

    def reset(self):
        if self.pipeline is not None:
            self.pipeline.reset()

    def interrupt(self):
        self.interrupted.set()
        if self.speaker is not None:
            self.speaker.stop()

    def look(self, jpeg):
        """jpeg: one camera image as JPEG bytes. The face in it, if any, is kept for the turn being spoken.
        Returns a face event for the page: where the face is (as shares of the picture) and what it shows."""
        if self.pipeline is None:
            return None
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return None
        face = self.pipeline.push_image(frame, from_camera=True)
        if face is None:
            return {"type": "face", "found": False}
        height, width = frame.shape[:2]
        x, y, w, h = face.box
        top = int(face.expression.argmax())
        return {"type": "face", "found": True, "box": [x / width, y / height, w / width, h / height],
                "points": [[px / width, py / height] for px, py in face.points],
                "label": LABELS[top], "confidence": float(face.expression[top])}

    async def prepare(self):
        """Loads the models ahead of the first turn."""
        async for event in in_thread(self.prepare_events):
            yield event

    async def run_turn(self, audio_chunks):
        loop = asyncio.get_running_loop()
        chunks = []
        async for chunk in audio_chunks:
            if not chunks:
                yield {"type": "status", "state": "hearing"}
            chunks.append(chunk)
            if len(chunks) % WARM_EVERY_CHUNKS == 0 and self.pipeline is not None and not self.warming:
                loop.run_in_executor(None, self.keep_warm)
        wav = np.frombuffer(b"".join(chunks), dtype=np.int16).astype(np.float32) / 32768
        async for event in in_thread(lambda: self.turn_events(wav)):
            yield event

    async def run_clip(self, name):
        async for event in in_thread(lambda: self.clip_events(name)):
            yield event

    # ---- everything below runs in a worker thread ----

    def keep_warm(self):
        # the GPU slows down while the person is still talking; a little work keeps it fast for the end of the turn
        self.warming = True
        try:
            self.pipeline.keep_warm()
        finally:
            self.warming = False

    def load(self):
        """Loads every model once. Yields what is going on, so the page can show it."""
        if self.pipeline is not None:
            return
        yield {"type": "status", "state": "loading"}
        os.environ["HF_HUB_OFFLINE"] = "1"          # the live path never touches the network
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from meldrt.pipeline.build import build_pipeline, build_speaker

        pipeline = build_pipeline(sample_rate=SAMPLE_RATE)
        pipeline.warm_up()
        self.speaker = build_speaker() if self.voice else None
        self.pipeline = pipeline

    def prepare_events(self):
        with self.lock:
            try:
                yield from self.load()
            except Exception as error:
                # most often: the trained checkpoints are not there yet, or Ollama is not running
                yield {"type": "notice", "text": f"the models could not be loaded: {error}"}
                yield {"type": "notice", "text": "the README lists what has to exist: 'Reproduce the models' and 'Setup'."}
                yield {"type": "status", "state": "unavailable"}
                return
            yield {"type": "status", "state": "idle"}

    def turn_events(self, wav):
        """One push-to-talk turn: the audio between press and release is the turn."""
        with self.lock:
            self.interrupted.clear()
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
            self.interrupted.clear()
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
            # if the sample has a video, its pictures go in at the moment they belong, like a camera
            video = self.video_of(name)
            pictures = self.pipeline.read_video(video) if video is not None else []
            n = self.pipeline.frame_samples
            said = []
            for start in range(0, len(wav) - n + 1, n):
                if self.interrupted.is_set():
                    break
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
            if not said and not self.interrupted.is_set():
                yield {"type": "notice", "text": "no speech found in this clip."}
            expected = sample_labels(name)
            if expected and len(expected) == len(said):
                yield {"type": "notice", "text": f"sample label: {', '.join(expected)}. the robot said: {', '.join(said)}."}
            elif expected and said and not self.interrupted.is_set():
                yield {"type": "notice", "text": f"this sample has {len(expected)} labelled lines but the robot heard "
                                                 f"{len(said)} turns, so they are not compared."}
            # pictures from the end of the clip must not count as faces of whatever the person says next
            self.pipeline.clear_audio()
            yield {"type": "status", "state": "idle"}

    def play_out_loud(self, wav):
        """Plays clip audio through the speakers without waiting, so the person watching hears what the robot hears."""
        if self.speaker is not None and len(wav) and not self.interrupted.is_set():
            import sounddevice as sd

            # no speakers plugged in must not stop the turn; the robot's own voice handles this the same way
            try:
                sd.play(wav, SAMPLE_RATE, device=self.speaker.output_device)
            except Exception as error:
                print(f"[page] could not play the clip out loud: {error}", flush=True)

    def answer(self, state):
        """Turns one finished state into page events, then streams and speaks the reply."""
        turn = state.turn_id
        state_ms = state.latency_ms["state"]

        yield {"type": "turn_start", "turn_id": turn}
        yield {"type": "status", "state": "thinking"}
        yield {"type": "transcript", "turn_id": turn, "text": state.transcript}
        yield {"type": "timing", "turn_id": turn, "stage": "transcript", "ms": round(state.latency_ms["asr"])}
        yield {"type": "emotion", "turn_id": turn, "fused": state.probs, "text": state.text_view.probs,
               # a turn too short for the audio model, or one with no face in view, shows zeros in that column
               "audio": state.audio_view.probs if state.audio_view else NO_FACE,
               "vision": state.vision_view.probs if state.vision_view else NO_FACE,
               "top": state.label, "confidence": state.confidence, "disagree": state.tone_text_conflict,
               "face_disagree": state.face_text_conflict, "faces": state.faces_seen}
        # the audio model runs before speech recognition, so this is the time added after the transcript
        yield {"type": "timing", "turn_id": turn, "stage": "emotion",
               "ms": round(max(0.0, state_ms - state.latency_ms["asr"]))}

        events = self.pipeline.reply(state)
        if self.speaker is not None:
            from meldrt.pipeline.speak import speak_while_streaming

            events = speak_while_streaming(events, self.speaker)
        first = True
        for event in events:
            if self.interrupted.is_set():
                break
            if event["type"] != "token":
                continue
            if first:
                first = False
                first_word_ms = (time.perf_counter() - self.pipeline.turn_ended_at) * 1000 - state_ms
                yield {"type": "timing", "turn_id": turn, "stage": "first_word", "ms": round(max(0.0, first_word_ms))}
                yield {"type": "status", "state": "speaking"}
            yield {"type": "reply_delta", "turn_id": turn, "text": event["text"]}
        events.close()
        yield {"type": "reply_done", "turn_id": turn}

        if self.speaker is not None and not self.interrupted.is_set():
            first_audio_at = self.speaker.wait()
            if first_audio_at is not None:
                yield {"type": "timing", "turn_id": turn, "stage": "first_audio",
                       "ms": round((first_audio_at - self.pipeline.turn_ended_at) * 1000)}
