"""The live turn loop: 32 ms frames go in, one TurnState comes out for each finished turn."""
import math
import time
from collections import deque

import numpy as np

from meldrt.data.video import read_timed_frames
from meldrt.labels import LABELS
from meldrt.models.fusion import late_fuse, sharpen, tone_text_conflict, two_against_words
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
    return View(label=LABELS[int(probs.argmax())], confidence=float(probs.max()),
                probs={name: float(p) for name, p in zip(LABELS, probs)})


class Pipeline:
    def __init__(self, vad, asr, classifier, cfg, alpha, sample_rate, weights3=None, temperature=1.0):
        p = cfg["pipeline"]
        if sample_rate != cfg["audio"]["sample_rate"]:
            raise ValueError(f"expected {cfg['audio']['sample_rate']} Hz audio, got {sample_rate}")
        self.vad, self.asr, self.classifier, self.alpha = vad, asr, classifier, alpha
        self.temperature = temperature      # makes the fused confidence match how often it is right (train/calibrate.py)
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
        self.warm_during_silence = p["warm_during_silence"]
        self.warm_wav = np.zeros(sample_rate // 2, dtype=np.float32)
        # vision is on when the three-model weights (text, audio, vision) are given
        self.weights3 = weights3
        self.frames_per_clip = cfg["vision"]["frames_per_clip"]
        self.max_turn_s = p["max_turn_s"]
        self.faces = deque(maxlen=p["max_faces"])     # (clock reading, Face, from a camera?) for recent pictures with a face

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
            if self.warm_during_silence and self.endpointer.silence_run > 0:
                # the turn may be about to end
                self.keep_warm()
        else:
            self.recent.append(frame)
        return None

    def keep_warm(self):
        """The GPU slows down while it sits idle during the speech. Half a second of silence through
        the audio model, with the answer thrown away, keeps it at full speed for the end of the turn."""
        self.classifier.audio_probs(self.warm_wav)

    def push_image(self, frame, from_camera=False):
        """frame: one BGR image, from a live camera or from a clip's video. If it shows a face, the face
        is remembered with the time it arrived, so the turn being spoken can use it. Returns the Face, or None."""
        if self.weights3 is None:
            return None
        face = self.classifier.look(frame, from_camera)
        if face is not None:
            self.faces.append((time.perf_counter(), face, from_camera))
        return face

    def read_video(self, path):
        """The pictures of a clip's video the face model should see: (seconds into the clip, BGR image) pairs,
        spread evenly over it, the same number as when the features were cached for training."""
        return read_timed_frames(path, self.frames_per_clip, self.max_turn_s)

    def finish_turn(self):
        # drop the silence that showed the turn was over
        trailing = self.endpointer.silence_run
        frames = self.turn[: len(self.turn) - trailing]
        self.turn = []
        self.vad.reset()
        return self.process_turn(np.concatenate(frames), endpoint_ms=trailing * self.frame_ms)

    def process_turn(self, wav, endpoint_ms):
        """One finished turn of audio to a TurnState. Returns None if no words were recognized."""
        start = self.turn_ended_at = time.perf_counter()
        # one after the other: on a single GPU the two models are slower side by side (see DECISIONS.md)
        p_audio, audio_ms = timed(self.classifier.audio_probs, wav)
        transcript, asr_ms = timed(self.asr.transcribe, wav)
        if not transcript:
            return None

        previous = list(self.history)[-self.context_k :] if self.context_k else []
        context = "\n".join(format_target(self.speaker, turn["transcript"]) for turn in previous)
        p_text, text_ms = timed(self.classifier.text_probs, context, format_target(self.speaker, transcript))

        # the faces seen while this turn was being spoken: from its first sample to now
        turn_started = start - endpoint_ms / 1000 - len(wav) / self.sample_rate
        seen = [(face, from_camera) for seen_at, face, from_camera in self.faces if seen_at >= turn_started]
        self.faces.clear()      # a face belongs to one turn only
        # one turn has one source of pictures; if both kinds arrived, the camera's are the ones kept
        from_camera = any(camera for _, camera in seen)
        faces = [face for face, camera in seen if camera == from_camera]
        p_vision, vision_ms = timed(self.classifier.vision_probs, faces, from_camera) if faces else (None, 0.0)

        if p_audio is None:
            fused, audio_view, conflict = p_text, None, False
        else:
            fused = late_fuse(p_text, p_audio, self.alpha)
            audio_view = view_of(p_audio)
            conflict = tone_text_conflict(p_text, p_audio, self.min_conf)
        vision_view, face_conflict = None, False
        if p_vision is not None:
            # with a face all three are blended; a turn too short for the audio model uses text in its place
            w_text, w_audio, w_vision = self.weights3
            fused = w_text * p_text + w_audio * (p_text if p_audio is None else p_audio) + w_vision * p_vision
            vision_view = view_of(p_vision)
            face_conflict = tone_text_conflict(p_text, p_vision, self.min_conf)
            if p_audio is not None and two_against_words(p_text, p_audio, p_vision):
                # voice and face agree with each other and not with the words. The fused label still follows
                # the weights chosen on dev, but the mismatch is flagged even if neither of the two is very sure
                conflict = face_conflict = True

        fused = sharpen(fused, self.temperature)

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
            vision_view=vision_view,
            face_text_conflict=face_conflict,
            faces_seen=len(faces),
        )
        if faces:
            state.latency_ms["vision"] = vision_ms
        state.latency_ms["state"] = (time.perf_counter() - start) * 1000
        self.history.append({"transcript": transcript, "label": state.label, "reply": ""})
        return state

    def reply(self, state):
        """Streams the robot's reply to a state. Yields one token event per piece, then one done event."""
        messages = build_prompt(state, list(self.history)[:-1], self.reply_cfg["low_conf"])
        pieces, first_token_ms = [], None
        try:
            for piece in stream_reply(messages, self.reply_cfg):
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - self.turn_ended_at) * 1000
                pieces.append(piece)
                yield {"type": "token", "turn_id": state.turn_id, "text": piece}
        finally:
            # also when the listener stops early because the person cut in: remember what was said so far
            text = "".join(pieces).strip().strip('"')     # the small model sometimes wraps its reply in quotes
            self.history[-1]["reply"] = text
        reply_ms = (time.perf_counter() - self.turn_ended_at) * 1000
        latency = {"first_token": round(first_token_ms or reply_ms), "reply": round(reply_ms)}
        yield {"type": "done", "turn_id": state.turn_id, "reply": text, "latency_ms": latency}

    def clear_audio(self):
        """Throws away any audio heard so far, but keeps the conversation. Used after the robot has spoken."""
        self.recent.clear()
        self.turn = []
        self.faces.clear()
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
