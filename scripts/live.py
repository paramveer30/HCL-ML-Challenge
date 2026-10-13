"""Listens to the microphone and answers each spoken turn. Stop with Ctrl+C.

Prints the state for each turn as JSON, then the reply as it streams, then the timings.
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"      # the live path never touches the network

import json
import queue
import sys

import sounddevice as sd

from meldrt.pipeline.build import build_pipeline

SAMPLE_RATE = 16000


def main():
    # the chat model can write characters a Windows console can't show; print "?" instead of crashing
    sys.stdout.reconfigure(errors="replace")

    print("loading models...", flush=True)
    pipeline = build_pipeline(sample_rate=SAMPLE_RATE)
    pipeline.warm_up()

    # the sound card calls on_audio from its own thread; we only hand the frame over to the main loop
    frames = queue.Queue()

    def on_audio(indata, frame_count, time_info, status):
        frames.put(indata[:, 0].copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=pipeline.frame_samples, callback=on_audio
    )
    with stream:
        print("listening. Say something, then pause. Ctrl+C to stop.", flush=True)
        try:
            while True:
                state = pipeline.push_frame(frames.get())
                if state is None:
                    continue
                print(json.dumps(state.to_event(), indent=2))
                print("reply: ", end="", flush=True)
                # frames that arrive while the reply streams wait in the queue for the next turn
                for event in pipeline.reply(state):
                    if event["type"] == "token":
                        print(event["text"], end="", flush=True)
                    else:
                        print(f"\nreply timings (ms after the turn ended): {event['latency_ms']}", flush=True)
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
