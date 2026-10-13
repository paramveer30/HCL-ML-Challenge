"""Speaks the robot's reply with Kokoro, one sentence at a time, without holding up the text stream."""
import queue
import re
import threading
import time

import sounddevice as sd
import torch
from kokoro import KPipeline

KOKORO_SAMPLE_RATE = 24000


class SentenceChunker:
    """Collects streamed text and hands back each sentence as soon as it is complete."""

    def __init__(self):
        self.buffer = ""

    def push(self, piece):
        """Adds a piece of text. Returns the sentences that are now finished (often none)."""
        self.buffer += piece
        # a sentence is finished when . ! or ? is followed by a space: the next sentence has begun
        parts = re.split(r"(?<=[.!?])\s+", self.buffer)
        self.buffer = parts[-1]
        return [part.strip() for part in parts[:-1] if part.strip()]

    def flush(self):
        """Returns whatever is left when the reply ends, or None."""
        rest, self.buffer = self.buffer.strip(), ""
        return rest or None


class Speaker:
    def __init__(self, voice_cfg, play=True):
        self.play = play                 # False: prepare the sound but stay silent (used for timing runs)
        device = voice_cfg["device"] if torch.cuda.is_available() else "cpu"
        self.tts = KPipeline(lang_code=voice_cfg["lang_code"], repo_id=voice_cfg["repo"], device=device)
        self.voice = voice_cfg["voice"]
        self.speed = voice_cfg["speed"]
        self.mute_tail_s = voice_cfg["mute_tail_ms"] / 1000
        self.output_device = voice_cfg["output_device"]
        # synthesize once without playing, so the first real sentence is not slowed by loading
        list(self.tts("Hello.", voice=self.voice, speed=self.speed))
        self.first_audio_at = None       # clock reading when this reply's first sound started playing
        self.sentences = queue.Queue()   # text waiting to be turned into sound
        self.sounds = queue.Queue()      # sound waiting to be played
        # two workers, so the next sentence is synthesized while the current one is still playing
        threading.Thread(target=self.synthesize_loop, daemon=True).start()
        threading.Thread(target=self.play_loop, daemon=True).start()

    def start_reply(self):
        self.first_audio_at = None

    def say(self, sentence):
        """Queues one sentence and returns at once."""
        self.sentences.put(sentence)

    def synthesize_loop(self):
        while True:
            sentence = self.sentences.get()
            chunks = [result.audio for result in self.tts(sentence, voice=self.voice, speed=self.speed)]
            if chunks:
                self.sounds.put(torch.cat(chunks).cpu().numpy())
            self.sentences.task_done()

    def play_loop(self):
        while True:
            audio = self.sounds.get()
            if self.first_audio_at is None:
                self.first_audio_at = time.perf_counter()
            if self.play:
                sd.play(audio, KOKORO_SAMPLE_RATE, device=self.output_device)
                sd.wait()
            self.sounds.task_done()

    def wait(self):
        """Blocks until everything queued so far has been spoken, plus a short tail for the room to go quiet."""
        self.sentences.join()
        self.sounds.join()
        time.sleep(self.mute_tail_s)


def speak_while_streaming(events, speaker):
    """Passes reply events through unchanged, and sends each finished sentence to the speaker right away."""
    chunker = SentenceChunker()
    speaker.start_reply()
    for event in events:
        if event["type"] == "token":
            for sentence in chunker.push(event["text"]):
                speaker.say(sentence)
        else:
            rest = chunker.flush()
            if rest:
                speaker.say(rest)
        yield event
