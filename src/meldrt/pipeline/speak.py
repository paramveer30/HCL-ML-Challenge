"""Speaks the robot's reply with Kokoro, one sentence at a time, without holding up the text stream."""
import queue
import re
import threading
import time

import numpy as np
import sounddevice as sd
import torch
from kokoro import KPipeline

KOKORO_SAMPLE_RATE = 24000


def play_tone(output_device, seconds=0.4, frequency=440):
    """A short beep on the speakers the voice will use."""
    t = np.arange(int(seconds * KOKORO_SAMPLE_RATE)) / KOKORO_SAMPLE_RATE
    sd.play((0.2 * np.sin(2 * np.pi * frequency * t)).astype(np.float32), KOKORO_SAMPLE_RATE, device=output_device)
    sd.wait()


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
        self.stopped = False             # True from stop() until the next reply starts
        self.sentences = queue.Queue()   # text waiting to be turned into sound
        self.sounds = queue.Queue()      # sound waiting to be played
        # two workers, so the next sentence is synthesized while the current one is still playing
        threading.Thread(target=self.synthesize_loop, daemon=True).start()
        threading.Thread(target=self.play_loop, daemon=True).start()

    def start_reply(self):
        self.first_audio_at = None
        self.stopped = False

    def say(self, sentence):
        """Queues one sentence and returns at once."""
        if not self.stopped:
            self.sentences.put(sentence)

    def stop(self):
        """Stops talking at once: drops the sentences still waiting and cuts the sound that is playing."""
        self.stopped = True
        for waiting in (self.sentences, self.sounds):
            while True:
                try:
                    waiting.get_nowait()
                except queue.Empty:
                    break
                waiting.task_done()
        if self.play:
            sd.stop()

    def synthesize_loop(self):
        while True:
            sentence = self.sentences.get()
            # if a sentence fails (for example the GPU runs out of memory) it is skipped; without the
            # finally, wait() would block forever and the robot would stop listening
            try:
                chunks = [result.audio for result in self.tts(sentence, voice=self.voice, speed=self.speed)]
                if chunks and not self.stopped:
                    self.sounds.put(torch.cat(chunks).cpu().numpy())
            except Exception as error:
                print(f"[voice] could not speak a sentence: {error}", flush=True)
            finally:
                self.sentences.task_done()

    def play_loop(self):
        while True:
            audio = self.sounds.get()
            try:
                if self.first_audio_at is None:
                    self.first_audio_at = time.perf_counter()
                if self.play and not self.stopped:
                    sd.play(audio, KOKORO_SAMPLE_RATE, device=self.output_device)
                    sd.wait()
            except Exception as error:
                print(f"[voice] could not play sound: {error}", flush=True)
            finally:
                self.sounds.task_done()

    def wait(self):
        """Blocks until everything queued so far has been spoken, plus a short tail for the room to go quiet.

        Returns the clock reading when this reply's first sound started, or None if nothing was spoken.
        """
        self.sentences.join()
        self.sounds.join()
        time.sleep(self.mute_tail_s)
        return self.first_audio_at


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
