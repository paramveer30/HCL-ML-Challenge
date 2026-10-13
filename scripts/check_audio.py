"""Checks the speakers and the microphone the live script will use.

Run this if you cannot hear the robot or it does not hear you. It plays a tone, records 4 seconds,
and reports how loud the recording was and how much of it the voice detector heard as speech.
"""
import numpy as np
import sounddevice as sd

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

    print("2. Recording 4 seconds. Say a sentence at normal volume now.")
    recording = sd.rec(4 * MIC_RATE, samplerate=MIC_RATE, channels=1, dtype="float32", device=mic)
    sd.wait()
    wav = recording[:, 0]

    n = cfg["pipeline"]["frame_samples"]
    vad = Vad(MIC_RATE, cfg["pipeline"]["vad_threshold"])
    flags = [vad.is_speech(wav[i : i + n]) for i in range(0, len(wav) - n + 1, n)]
    peak = float(np.abs(wav).max())
    print(f"   loudest sample: {peak:.3f} (1.0 is the maximum)")
    print(f"   heard as speech: {sum(flags)} of {len(flags)} frames ({sum(flags) * n / MIC_RATE:.1f} s)")
    if peak < 0.02:
        print("   This is very quiet. Move closer or raise the microphone level in the system sound settings.")
    elif sum(flags) == 0:
        print("   Sound came in but none of it was heard as speech.")
    else:
        print("   The microphone looks fine.")

    print("\nIf you did not hear the tone, set voice.output_device in configs/default.yaml to part of the")
    print("name of the device you listen on (from the list above), or change the system default output.")


if __name__ == "__main__":
    main()
