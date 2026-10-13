"""Replays WAV files through the live pipeline in 32 ms frames, at the speed a microphone would deliver them.

For each turn it prints what it heard, the emotion, the reply as it streams, and the timings. The reply is also spoken.
Add --json to print the full state record instead of the short lines.
Example: python scripts/replay.py D:/data/MELD/wav/dev/dia0_utt1.wav --label surprise --json
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"      # the live path never touches the network
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import argparse
import sys
import time
import warnings

import numpy as np
import soundfile as sf

from meldrt.pipeline.build import build_pipeline, build_speaker
from meldrt.pipeline.display import show_reply, show_state, show_timing
from meldrt.pipeline.speak import speak_while_streaming


def handle_turn(pipeline, speaker, state, with_reply, as_json):
    show_state(state, as_json)
    if not with_reply:
        return
    events = pipeline.reply(state)
    if speaker is not None:
        events = speak_while_streaming(events, speaker)
    done = show_reply(events)
    first_audio_ms = None
    if speaker is not None:
        speaker.wait()
        first_audio_ms = (speaker.first_audio_at - pipeline.turn_ended_at) * 1000
    show_timing(state, done, first_audio_ms)


def replay(pipeline, speaker, path, real_time, with_reply, as_json):
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
            handle_turn(pipeline, speaker, state, with_reply, as_json)
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
    parser.add_argument("--json", action="store_true", help="print the full state record for each turn")
    args = parser.parse_args()

    # the chat model can write characters a Windows console can't show; print "?" instead of crashing
    sys.stdout.reconfigure(errors="replace")
    warnings.filterwarnings("ignore")       # library deprecation notices would bury the output

    print("loading models, this takes a few seconds...", flush=True)
    pipeline = build_pipeline(sample_rate=16000)
    pipeline.warm_up()
    speaker = None if args.no_voice or args.no_reply else build_speaker()

    for path in args.wavs:
        print(f"\n=== {path}" + (f"   (true label: {args.label})" if args.label else ""))
        pipeline.reset()        # each file is its own conversation
        replay(pipeline, speaker, path, real_time=not args.fast, with_reply=not args.no_reply, as_json=args.json)


if __name__ == "__main__":
    main()
