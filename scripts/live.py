"""Listens to the microphone and answers each spoken turn, in text and out loud. Stop with Ctrl+C.

Prints the state for each turn as JSON, then the reply as it streams, then the timings.
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"      # the live path never touches the network

import argparse
import json
import queue
import sys

import sounddevice as sd

from meldrt.pipeline.build import build_pipeline, build_speaker
from meldrt.pipeline.speak import speak_while_streaming

SAMPLE_RATE = 16000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-voice", action="store_true", help="print the reply without speaking it")
    args = parser.parse_args()

    # the chat model can write characters a Windows console can't show; print "?" instead of crashing
    sys.stdout.reconfigure(errors="replace")

    print("loading models...", flush=True)
    pipeline = build_pipeline(sample_rate=SAMPLE_RATE)
    pipeline.warm_up()
    speaker = None if args.no_voice else build_speaker()

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

                events = pipeline.reply(state)
                if speaker is not None:
                    events = speak_while_streaming(events, speaker)
                print("reply: ", end="", flush=True)
                for event in events:
                    if event["type"] == "token":
                        print(event["text"], end="", flush=True)
                    else:
                        print(f"\nreply timings (ms after the turn ended): {event['latency_ms']}", flush=True)

                if speaker is not None:
                    # the mic also hears the robot. Wait until it has finished, then throw away what was
                    # recorded meanwhile, so the robot never answers itself. This also means no barge-in.
                    speaker.wait()
                    first_audio_ms = (speaker.first_audio_at - pipeline.turn_ended_at) * 1000
                    print(f"first spoken audio: {first_audio_ms:.0f} ms after the turn ended", flush=True)
                    while not frames.empty():
                        frames.get_nowait()
                    pipeline.clear_audio()
                # without a voice, frames that arrived while the reply streamed stay queued for the next turn
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
