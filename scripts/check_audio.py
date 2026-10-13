"""Checks the speakers and the microphone the live script will use.

Run this if you cannot hear the robot or it does not hear you. It plays a tone, records 6 seconds,
and reports how loud the recording was and how much of it the voice detector heard as speech.
"""
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from meldrt.config import load_config
from meldrt.pipeline.vad import Vad

TONE_RATE = 24000      # the rate the voice plays at
MIC_RATE = 16000


def main():
    cfg = load_config()
    mic = cfg["pipeline"]["input_device"]
    out = cfg["voice"]["output_device"]

    print(sd.query_devices())
    print("\n'>' marks the default microphone and '<' the default speakers.")
    print(f"configured microphone: {mic or 'system default'}")
    print(f"configured speakers:   {out or 'system default'}")

    print("\n1. Playing a one second tone. You should hear it.")
    t = np.arange(TONE_RATE) / TONE_RATE
    sd.play((0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), TONE_RATE, device=out)
    sd.wait()

    print("2. Recording 6 seconds. Keep talking at normal volume until it stops.")
    recording = sd.rec(6 * MIC_RATE, samplerate=MIC_RATE, channels=1, dtype="float32", device=mic)
    sd.wait()
    wav = recording[:, 0]

    n = cfg["pipeline"]["frame_samples"]
    vad = Vad(MIC_RATE, cfg["pipeline"]["vad_threshold"])
    flags = [vad.is_speech(wav[i : i + n]) for i in range(0, len(wav) - n + 1, n)]
    longest = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    start_frames = int(np.ceil(cfg["pipeline"]["start_ms"] / (1000 * n / MIC_RATE)))
    peak = float(np.abs(wav).max())
    print(f"   loudest sample: {peak:.3f} (1.0 is the maximum)")
    print(f"   heard as speech: {sum(flags)} of {len(flags)} frames ({sum(flags) * n / MIC_RATE:.1f} s)")
    print(f"   longest unbroken stretch of speech: {longest} frames (a turn needs {start_frames} in a row to start)")

    out_path = Path(cfg["eval"]["bench_dir"]) / "mic_check.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, wav, MIC_RATE)
    print(f"   saved the recording to {out_path}, so you can listen to what the robot hears")
    if peak < 0.02:
        print("   This is very quiet. Move closer or raise the microphone level in the system sound settings.")
    elif longest < start_frames:
        print("   Sound came in, but the voice detector never heard enough speech in a row to start a turn.")
        print("   The robot will not react to this microphone as it is. Try another microphone.")
    else:
        print("   The microphone looks fine.")

    print("\nIf you did not hear the tone, set voice.output_device in configs/default.yaml to part of the")
    print("name of the device you listen on (from the list above), or change the system default output.")


if __name__ == "__main__":
    main()
