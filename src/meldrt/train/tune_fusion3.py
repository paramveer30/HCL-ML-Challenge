"""Picks the three-model fusion weights on dev from the saved dev logits of the text, audio, and vision models.

The two-model weight alpha (tune_fusion.py) is not changed here: rows with no face keep using it.
"""
import json
from pathlib import Path

import numpy as np

from meldrt.config import load_config
from meldrt.data.meld import load_split
from meldrt.eval.metrics import f1_summary
from meldrt.labels import LABEL2ID
from meldrt.models.fusion import fill_missing_audio, fuse_three, softmax, tune_alpha, tune_three


def main():
    cfg = load_config()
    step = cfg["fusion"]["alpha_step"]
    run_dir = Path(cfg["fusion"]["run_dir"])

    dev_df = load_split("dev", cfg)
    y = np.array([LABEL2ID[e] for e in dev_df["Emotion"]])
    text_logits = np.load(Path(cfg["text"]["run_dir"]) / "dev_logits.npy")
    audio_logits = np.load(Path(cfg["audio_head"]["run_dir"]) / "dev_logits.npy")
    vision_logits = np.load(Path(cfg["vision_head"]["run_dir"]) / "dev_logits.npy")
    with open(run_dir / "alpha.json", encoding="utf-8") as f:
        alpha = json.load(f)["alpha"]

    has_audio = ~np.isnan(audio_logits[:, 0])
    has_face = ~np.isnan(vision_logits[:, 0])
    p_text = softmax(text_logits)
    p_audio = fill_missing_audio(p_text, softmax(audio_logits), has_audio)
    p_vision = fill_missing_audio(p_text, softmax(vision_logits), has_face)

    # text + vision, one weight, the same search as text + audio
    alpha_vision, text_vision_f1 = tune_alpha(p_text, p_vision, y, step)
    print(f"text + vision: best weight on text {alpha_vision:.2f}, dev weighted F1 {text_vision_f1:.4f}")

    weights, best_f1, grid = tune_three(p_text, p_audio, p_vision, has_face, alpha, y, step)
    two_model = f1_summary(y, fuse_three(p_text, p_audio, p_vision, has_face, (alpha, 1 - alpha, 0.0), alpha).argmax(axis=1))
    print(f"text + audio (alpha {alpha}): dev weighted F1 {two_model['weighted_f1']:.4f}")
    print(f"text + audio + vision: best weights (text, audio, vision) {weights}, dev weighted F1 {best_f1:.4f}")
    for row in sorted(grid, key=lambda r: -r["weighted_f1"])[:8]:
        print(f"  {row['weights']}  weighted {row['weighted_f1']:.4f}  macro {row['macro_f1']:.4f}")

    with open(run_dir / "weights3.json", "w", encoding="utf-8") as f:
        json.dump({"weights": list(weights), "dev_weighted_f1": best_f1, "alpha_text_vision": alpha_vision,
                   "text_vision_dev_weighted_f1": text_vision_f1, "rows_with_a_face": int(has_face.sum()), "grid": grid}, f, indent=2)


if __name__ == "__main__":
    main()
