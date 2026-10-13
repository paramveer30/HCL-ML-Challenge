"""Times every stage of the live pipeline over replayed turns, and records peak memory.

Clips are pushed in 32 ms frames at real-time pace, exactly like replay.py. Only WAV files are read.
The voice prepares each sentence but stays silent; "first_audio" is when the first sound is ready
to be handed to the sound card. With vision on, each clip's video pictures are pushed in at the moment
they belong, and "image" is the time one picture takes (face detection plus the ViT).
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
import json
import platform
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf
import torch

from meldrt.config import load_config
from meldrt.pipeline.build import build_pipeline, build_speaker
from meldrt.pipeline.speak import speak_while_streaming

STAGES = ["endpoint", "asr", "audio", "text", "vision", "state", "first_token", "reply", "first_audio"]


def gpu_used_mib():
    """Memory in use on the whole card, all programs together."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True
    )
    return int(out.stdout.strip().splitlines()[0])


def ram_used_mib():
    """RAM held by this process plus Ollama's processes."""
    total = psutil.Process().memory_info().rss
    for process in psutil.process_iter(["name", "memory_info"]):
        if (process.info["name"] or "").lower().startswith("ollama") and process.info["memory_info"]:
            total += process.info["memory_info"].rss
    return total / 2**20


def unload_reply_model(reply_cfg):
    """Asks Ollama to drop the model, so the GPU baseline is measured without it."""
    body = json.dumps({"model": reply_cfg["model"], "keep_alive": 0}).encode("utf-8")
    request = urllib.request.Request(reply_cfg["url"] + "/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
    urllib.request.urlopen(request).read()
    time.sleep(3)


def main():
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-dir", default=str(Path(cfg["data"]["wav_root"]) / "dev"))
    parser.add_argument("--video-dir", default=str(Path(cfg["data"]["meld_root"]) / cfg["data"]["splits"]["dev"]["clips"]))
    parser.add_argument("--turns", type=int, default=50)
    parser.add_argument("--no-voice", action="store_true")
    args = parser.parse_args()

    unload_reply_model(cfg["reply"])
    gpu_baseline = gpu_used_mib()

    # memory is read between turns only: reading it while a turn is being processed slowed the turn down
    peaks = {"gpu": gpu_baseline, "ram": 0.0}

    def read_memory():
        peaks["gpu"] = max(peaks["gpu"], gpu_used_mib())
        peaks["ram"] = max(peaks["ram"], ram_used_mib())

    pipeline = build_pipeline(sample_rate=cfg["audio"]["sample_rate"])
    pipeline.warm_up()
    speaker = None if args.no_voice else build_speaker(play=False)

    n = pipeline.frame_samples
    frame_seconds = n / pipeline.sample_rate
    turns = []
    image_ms = []        # one entry per video picture pushed in
    for path in sorted(Path(args.wav_dir).glob("*.wav")):
        if len(turns) >= args.turns:
            break
        wav, _ = sf.read(path, dtype="float32")
        if len(wav) < pipeline.sample_rate:          # skip clips under one second
            continue
        wav = np.concatenate([wav, np.zeros(pipeline.sample_rate, dtype=np.float32)])
        pipeline.reset()
        video = Path(args.video_dir) / (path.stem + ".mp4")
        pictures = pipeline.read_video(video) if pipeline.weights3 is not None and video.exists() else []
        for start in range(0, len(wav) - n + 1, n):
            frame_start = time.perf_counter()
            while pictures and pictures[0][0] <= start / pipeline.sample_rate:
                image_start = time.perf_counter()
                pipeline.push_image(pictures.pop(0)[1])
                image_ms.append((time.perf_counter() - image_start) * 1000)
            state = pipeline.push_frame(wav[start : start + n])
            if state is not None:
                events = pipeline.reply(state)
                if speaker is not None:
                    events = speak_while_streaming(events, speaker)
                final = list(events)[-1]
                row = {**state.latency_ms, **final["latency_ms"], "audio_seconds": state.audio_seconds,
                       "faces_seen": state.faces_seen}
                if speaker is not None:
                    row["first_audio"] = (speaker.wait() - pipeline.turn_ended_at) * 1000
                turns.append(row)
                read_memory()
                print(f"turn {len(turns):2d}  {path.name}  " + "  ".join(f"{k} {row[k]:.0f}" for k in STAGES if k in row),
                      flush=True)
            time.sleep(max(0.0, frame_seconds - (time.perf_counter() - frame_start)))

    stages = [s for s in STAGES if any(s in t for t in turns)]
    summary = {s: {"p50": float(np.percentile([t[s] for t in turns if s in t], 50)),
                   "p95": float(np.percentile([t[s] for t in turns if s in t], 95))} for s in stages}
    if image_ms:
        summary["image"] = {"p50": float(np.percentile(image_ms, 50)), "p95": float(np.percentile(image_ms, 95))}

    print(f"\n{len(turns)} turns, clips of {min(t['audio_seconds'] for t in turns):.1f} to "
          f"{max(t['audio_seconds'] for t in turns):.1f} s")
    print("\n| Stage (ms after the turn ended, except endpoint) | p50 | p95 |")
    print("| --- | ---: | ---: |")
    for s in stages:
        print(f"| {s} | {summary[s]['p50']:.0f} | {summary[s]['p95']:.0f} |")
    if image_ms:
        with_face = sum(t["faces_seen"] > 0 for t in turns)
        print(f"\none video picture (face detection + ViT): p50 {summary['image']['p50']:.0f} ms, "
              f"p95 {summary['image']['p95']:.0f} ms over {len(image_ms)} pictures; {with_face} of {len(turns)} turns had a face")

    hardware = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "cpu": platform.processor(),
        "cpu_threads": psutil.cpu_count(),
        "ram_total_mib": round(psutil.virtual_memory().total / 2**20),
    }
    memory = {
        "gpu_baseline_mib": gpu_baseline,
        "gpu_peak_mib": peaks["gpu"],
        "gpu_used_by_the_system_mib": peaks["gpu"] - gpu_baseline,
        "ram_peak_mib": round(peaks["ram"]),
        "torch_peak_mib": round(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else 0,
    }
    print(f"\nhardware: {hardware}")
    print(f"memory: {memory}")

    out_dir = Path(cfg["eval"]["bench_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "latency.json", "w", encoding="utf-8") as f:
        json.dump({"turns": len(turns), "voice": speaker is not None, "summary": summary,
                   "hardware": hardware, "memory": memory, "per_turn": turns}, f, indent=2)


if __name__ == "__main__":
    main()
