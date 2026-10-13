"""The live turn loop: 32 ms frames go in, one TurnState comes out for each finished turn."""
import math
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from meldrt.labels import LABELS
from meldrt.models.fusion import late_fuse, tone_text_conflict
from meldrt.models.text import format_target
from meldrt.pipeline.endpointer import Endpointer
from meldrt.pipeline.respond import build_prompt, stream_reply
from meldrt.schema import TurnState, View


def timed(fn, *args):
    """Runs fn and returns (its result, how long it took in ms)."""
    start = time.perf_counter()
    result = fn(*args)
    return result, (time.perf_counter() - start) * 1000


def view_of(probs):
    return View(label=LABELS[int(probs.argmax())], confidence=float(probs.max()))


class Pipeline:
    def __init__(self, vad, asr, classifier, cfg, alpha, sample_rate):
        p = cfg["pipeline"]
        if sample_rate != cfg["audio"]["sample_rate"]:
            raise ValueError(f"expected {cfg['audio']['sample_rate']} Hz audio, got {sample_rate}")
        self.vad, self.asr, self.classifier, self.alpha = vad, asr, classifier, alpha
        self.sample_rate = sample_rate
        self.frame_samples = p["frame_samples"]
        self.frame_ms = 1000 * self.frame_samples / sample_rate
        self.endpoint_settings = (p["start_ms"], p["end_silence_ms"], p["max_turn_s"])
        self.endpointer = Endpointer(self.frame_ms, *self.endpoint_settings)
        self.min_conf = cfg["fusion"]["conflict_min_conf"]
        self.context_k = cfg["data"]["context_k"]
        self.speaker = p["speaker_name"]
        self.input_device = p["input_device"]
        self.reply_cfg = cfg["reply"]

        # frames from just before a turn is detected: the speech frames that triggered it, plus a little more
        preroll = self.endpointer.start_frames - 1 + math.ceil(p["preroll_ms"] / self.frame_ms)
        self.recent = deque(maxlen=preroll)
        self.turn = []
        self.history = deque(maxlen=p["history_turns"])     # dicts with transcript, label, reply
        self.turn_id = 0
        self.turn_ended_at = 0.0     # clock reading when the latest turn was decided to be over
        self.pool = ThreadPoolExecutor(2)

    def push_frame(self, frame):
        """frame: float32 array of exactly frame_samples samples. Returns a TurnState when a turn just ended."""
        if frame.shape != (self.frame_samples,):
            raise ValueError(f"expected a frame of {self.frame_samples} samples, got shape {frame.shape}")
        event = self.endpointer.push(self.vad.is_speech(frame))
        if event == "start":
            self.turn = list(self.recent) + [frame]
            self.recent.clear()
        elif event == "end":
            self.turn.append(frame)
            return self.finish_turn()
        elif self.endpointer.in_turn:
            self.turn.append(frame)
        else:
            self.recent.append(frame)
        return None

    def finish_turn(self):
        # drop the silence that told us the turn was over
        trailing = self.endpointer.silence_run
        frames = self.turn[: len(self.turn) - trailing]
        self.turn = []
        self.vad.reset()
        return self.process_turn(np.concatenate(frames), endpoint_ms=trailing * self.frame_ms)

    def process_turn(self, wav, endpoint_ms):
        """One finished turn of audio to a TurnState. Returns None if no words were recognized."""
        start = self.turn_ended_at = time.perf_counter()
        # speech recognition and the audio model start together; the audio model does not need the words
        asr_job = self.pool.submit(timed, self.asr.transcribe, wav)
        audio_job = self.pool.submit(timed, self.classifier.audio_probs, wav)
        transcript, asr_ms = asr_job.result()
        p_audio, audio_ms = audio_job.result()
        if not transcript:
            return None

        previous = list(self.history)[-self.context_k :] if self.context_k else []
        context = "\n".join(format_target(self.speaker, turn["transcript"]) for turn in previous)
        p_text, text_ms = timed(self.classifier.text_probs, context, format_target(self.speaker, transcript))

        if p_audio is None:
            fused, audio_view, conflict = p_text, None, False
        else:
            fused = late_fuse(p_text, p_audio, self.alpha)
            audio_view = view_of(p_audio)
            conflict = tone_text_conflict(p_text, p_audio, self.min_conf)

        self.turn_id += 1
        state = TurnState(
            turn_id=self.turn_id,
            transcript=transcript,
            label=LABELS[int(fused.argmax())],
            confidence=float(fused.max()),
            probs={name: float(p) for name, p in zip(LABELS, fused)},
            text_view=view_of(p_text),
            audio_view=audio_view,
            tone_text_conflict=conflict,
            audio_seconds=len(wav) / self.sample_rate,
            latency_ms={"endpoint": endpoint_ms, "asr": asr_ms, "audio": audio_ms, "text": text_ms},
        )
        state.latency_ms["state"] = (time.perf_counter() - start) * 1000
        self.history.append({"transcript": transcript, "label": state.label, "reply": ""})
        return state

    def reply(self, state):
        """Streams the robot's reply to a state. Yields one token event per piece, then one done event."""
        messages = build_prompt(state, list(self.history)[:-1], self.reply_cfg["low_conf"])
        pieces, first_token_ms = [], None
        for piece in stream_reply(messages, self.reply_cfg):
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - self.turn_ended_at) * 1000
            pieces.append(piece)
            yield {"type": "token", "turn_id": state.turn_id, "text": piece}
        reply_ms = (time.perf_counter() - self.turn_ended_at) * 1000
        text = "".join(pieces).strip().strip('"')     # the small model sometimes wraps its reply in quotes
        self.history[-1]["reply"] = text
        latency = {"first_token": round(first_token_ms or reply_ms), "reply": round(reply_ms)}
        yield {"type": "done", "turn_id": state.turn_id, "reply": text, "latency_ms": latency}

    def clear_audio(self):
        """Throws away any audio heard so far, but keeps the conversation. Used after the robot has spoken."""
        self.recent.clear()
        self.turn = []
        self.endpointer = Endpointer(self.frame_ms, *self.endpoint_settings)
        self.vad.reset()

    def reset(self):
        """Forgets the conversation so far as well. Used between unrelated clips."""
        self.history.clear()
        self.clear_audio()

    def warm_up(self):
        """Runs every model once so the first real turn is not slowed by loading."""
        silence = np.zeros(self.sample_rate, dtype=np.float32)
        self.asr.transcribe(silence)
        self.classifier.audio_probs(silence)
        self.classifier.text_probs("", format_target(self.speaker, "hello"))
        for _ in stream_reply([{"role": "user", "content": "Say hello."}], self.reply_cfg):
            pass
