"""Finds the largest face in a few frames of every MELD clip and caches frozen ViT features, 13 x 768 per clip.

Safe to rerun. Works from the mp4 files only: no label file is opened, so the test clips can be cached too.
Reading and face detection run in worker processes on the CPU; the ViT runs here on the GPU.
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import ViTModel

from meldrt.config import load_config
from meldrt.data.video import FaceFinder, face_crops, read_frames
from meldrt.models.vision import face_layers, mean_over_faces

finder = None       # one face detector per worker process


def start_worker(detector_file, min_score, min_face_px, detect_width):
    global finder
    cv2.setNumThreads(1)        # the workers are the parallelism; OpenCV's own threads would fight them
    finder = FaceFinder(detector_file, min_score, min_face_px, detect_width)


def clip_faces(job):
    path, k, max_seconds, margin, size = job
    frames = read_frames(path, k, max_seconds)
    return Path(path).stem, face_crops(frames, finder, margin, size)


def cache_split(name, cfg, model, device, pool, batch_clips=48):
    v = cfg["vision"]
    clip_dir = Path(cfg["data"]["meld_root"]) / cfg["data"]["splits"][name]["clips"]
    out_dir = Path(v["feature_root"]) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = sorted(clip_dir.glob("dia*_utt*.mp4"))
    no_face_file = out_dir / "no_face.txt"
    known_no_face = set(no_face_file.read_text(encoding="utf-8").split()) if no_face_file.exists() else set()
    todo = [c for c in clips if not (out_dir / (c.stem + ".npy")).exists() and c.stem not in known_no_face]
    print(f"{name}: {len(clips)} clips, {len(clips) - len(todo)} already done", flush=True)

    jobs = [(str(c), v["frames_per_clip"], cfg["audio"]["max_seconds"], v["face_margin"], v["image_size"]) for c in todo]
    # one batch of clips at a time. While the ViT works on a batch, the workers already find the faces
    # of the next one. Handing the workers the whole list at once piled face crops up in memory faster
    # than the GPU could use them.
    batches = [jobs[i : i + batch_clips] for i in range(0, len(jobs), batch_clips)]
    waiting = pool.map_async(clip_faces, batches[0]) if batches else None
    for n, batch in enumerate(batches):
        start = n * batch_clips
        results = waiting.get()
        waiting = pool.map_async(clip_faces, batches[n + 1]) if n + 1 < len(batches) else None
        found = [(stem, crops) for stem, crops in results if len(crops)]
        known_no_face.update(Path(job[0]).stem for job in batch)
        known_no_face.difference_update(stem for stem, _ in found)
        if found:
            faces = np.concatenate([crops for _, crops in found])
            layers = np.concatenate([face_layers(faces[i : i + 64], model, device) for i in range(0, len(faces), 64)])
            first = 0
            for stem, crops in found:
                # save under a temporary name, then rename, so a killed run leaves no half-written file
                tmp = out_dir / (stem + ".tmp.npy")
                np.save(tmp, mean_over_faces(layers[first : first + len(crops)]))
                tmp.replace(out_dir / (stem + ".npy"))
                first += len(crops)
        no_face_file.write_text(" ".join(sorted(known_no_face)), encoding="utf-8")
        if n % 10 == 9:
            print(f"{name}: {start + len(batch)}/{len(jobs)}", flush=True)

    stats = {"clips": len(clips), "clips_with_a_face": len(list(out_dir.glob("dia*_utt*.npy"))),
             "clips_without_a_face": len(known_no_face), "frames_per_clip": v["frames_per_clip"]}
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"{name}: done, {stats}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    cfg = load_config()
    v = cfg["vision"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # the checkpoint also holds a 7-way expression classifier; only the encoder under it is used
    model = ViTModel.from_pretrained(v["model"], add_pooling_layer=False).to(device).eval()
    with Pool(args.workers, start_worker, (v["detector_file"], v["min_face_score"], v["min_face_px"], v["detect_width"])) as pool:
        for name in args.splits:
            cache_split(name, cfg, model, device, pool)
