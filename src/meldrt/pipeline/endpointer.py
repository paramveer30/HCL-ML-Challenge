"""Turns a stream of speech / no-speech flags (one per audio frame) into turn start and end events."""
import math


class Endpointer:
    def __init__(self, frame_ms=32, start_ms=250, end_silence_ms=600, max_turn_s=15):
        # round up so we never start or end early; round down so a turn never exceeds the cap
        self.start_frames = math.ceil(start_ms / frame_ms)          # 8 frames = 256 ms
        self.end_frames = math.ceil(end_silence_ms / frame_ms)      # 19 frames = 608 ms
        self.max_frames = int(max_turn_s * 1000 // frame_ms)        # 468 frames = 14.976 s
        self.in_turn = False
        self.speech_run = 0      # speech frames in a row while waiting for a turn
        self.silence_run = 0     # silent frames in a row inside a turn
        self.turn_frames = 0     # length of the turn so far, counting the frames that started it

    def push(self, is_speech):
        """Feed the flag for one frame. Returns "start", "end", or None."""
        if not self.in_turn:
            self.speech_run = self.speech_run + 1 if is_speech else 0
            if self.speech_run == self.start_frames:
                self.in_turn = True
                self.turn_frames = self.speech_run
                self.silence_run = 0
                self.speech_run = 0
                return "start"
            return None

        self.turn_frames += 1
        self.silence_run = 0 if is_speech else self.silence_run + 1
        if self.silence_run == self.end_frames or self.turn_frames == self.max_frames:
            self.in_turn = False
            return "end"
        return None
