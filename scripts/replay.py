"""Replays WAV files through the live pipeline in 32 ms frames, at the speed a microphone would deliver them.

Prints the state for each turn as JSON, then the reply as it streams, then the timings. The reply is also spoken.
Example: python scripts/replay.py D:/data/MELD/wav/dev/dia0_utt1.wav --label surprise
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"      # the live path never touches the network

import argparse
import json
import sys
import time

import numpy as np
import soundfile as sf

from meldrt.pipeline.build import build_pipeline, build_speaker
from meldrt.pipeline.speak import speak_while_streaming


def handle_turn(pipeline, speaker, state, with_reply):
    print(json.dumps(state.to_event(), indent=2))
    if not with_reply:
        return
    events = pipeline.reply(state)
    if speaker is not None:
        events = speak_while_streaming(events, speaker)
    print("reply: ", end="", flush=True)
    for event in events:
        if event["type"] == "token":
            print(event["text"], end="", flush=True)
        else:
            print(f"\nreply timings (ms after the turn ended): {event['latency_ms']}")
    if speaker is not None:
        speaker.wait()
        first_audio_ms = (speaker.first_audio_at - pipeline.turn_ended_at) * 1000
        print(f"first spoken audio: {first_audio_ms:.0f} ms after the turn ended")


def replay(pipeline, speaker, path, real_time, with_reply):
    wav, sample_rate = sf.read(path, dtype="float32")
    if wav.ndim != 1 or sample_rate != pipeline.sample_rate:
        raise ValueError(f"{path}: expected mono {pipeline.sample_rate} Hz audio")
    # one second of silence after the clip, so the pipeline can hear that the turn is over
    wav = np.concatenate([wav, np.zeros(pipeline.sample_rate, dtype=np.float32)])

    n = pipeline.frame_samples
    frame_seconds = n / sample_rate
    turns = 0
    for start in range(0, len(wav) - n + 1, n):
        frame_start = time.perf_counter()
        state = pipeline.push_frame(wav[start : start + n])
        if state is not None:
            turns += 1
            handle_turn(pipeline, speaker, state, with_reply)
        if real_time:
            time.sleep(max(0.0, frame_seconds - (time.perf_counter() - frame_start)))
    if turns == 0:
        print("no turn detected in this clip")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wavs", nargs="+", help="16 kHz mono WAV files")
    parser.add_argument("--label", help="the true emotion, printed next to the result (one WAV only)")
    parser.add_argument("--fast", action="store_true", help="push frames as fast as possible instead of in real time")
    parser.add_argument("--no-reply", action="store_true", help="print the state only")
    parser.add_argument("--no-voice", action="store_true", help="print the reply without speaking it")
    args = parser.parse_args()

    # the chat model can write characters a Windows console can't show; print "?" instead of crashing
    sys.stdout.reconfigure(errors="replace")

    print("loading models...", flush=True)
    pipeline = build_pipeline(sample_rate=16000)
    pipeline.warm_up()
    speaker = None if args.no_voice or args.no_reply else build_speaker()

    for path in args.wavs:
        print(f"\n=== {path}" + (f"   (true label: {args.label})" if args.label else ""))
        pipeline.reset()        # each file is its own conversation
        replay(pipeline, speaker, path, real_time=not args.fast, with_reply=not args.no_reply)


if __name__ == "__main__":
    main()
