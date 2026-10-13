"""Listens to the microphone and answers each spoken turn, in text and out loud. Stop with Ctrl+C.

For each turn it prints what it heard, the emotion, the reply as it streams, and the timings.
Add --json to print the full state record instead of the short lines.
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"      # the live path never touches the network
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import argparse
import queue
import sys
import warnings

import sounddevice as sd

from meldrt.pipeline.build import build_pipeline, build_speaker, play_test_tone
from meldrt.pipeline.display import show_reply, show_state, show_timing
from meldrt.pipeline.speak import speak_while_streaming

SAMPLE_RATE = 16000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-voice", action="store_true", help="print the reply without speaking it")
    parser.add_argument("--json", action="store_true", help="print the full state record for each turn")
    args = parser.parse_args()

    # the chat model can write characters a Windows console can't show; print "?" instead of crashing
    sys.stdout.reconfigure(errors="replace")
    warnings.filterwarnings("ignore")       # library deprecation notices would bury the conversation

    if not args.no_voice:
        print("Audio check... ", end="", flush=True)
        play_test_tone()
        print("you should have heard a tone :)", flush=True)
    print("Loading models... takes a sec", flush=True)
    pipeline = build_pipeline(sample_rate=SAMPLE_RATE)
    pipeline.warm_up()
    speaker = None if args.no_voice else build_speaker()

    # the sound card calls on_audio from its own thread; we only hand the frame over to the main loop
    frames = queue.Queue()

    def on_audio(indata, frame_count, time_info, status):
        frames.put(indata[:, 0].copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=pipeline.frame_samples,
        device=pipeline.input_device, callback=on_audio,
    )
    with stream:
        print("\nListening. Say something, then pause. Ctrl+C to stop.", flush=True)
        try:
            while True:
                state = pipeline.push_frame(frames.get())
                if state is None:
                    continue
                show_state(state, as_json=args.json)

                events = pipeline.reply(state)
                if speaker is not None:
                    events = speak_while_streaming(events, speaker)
                done = show_reply(events)

                first_audio_ms = None
                if speaker is not None:
                    # the mic also hears the robot. Wait until it has finished, then throw away what was
                    # recorded meanwhile, so the robot never answers itself. This also means no barge-in.
                    first_audio_at = speaker.wait()
                    if first_audio_at is not None:
                        first_audio_ms = (first_audio_at - pipeline.turn_ended_at) * 1000
                    while not frames.empty():
                        frames.get_nowait()
                    pipeline.clear_audio()
                # without a voice, frames that arrived while the reply streamed stay queued for the next turn
                show_timing(state, done, first_audio_ms)
                print("\nListening.", flush=True)
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
