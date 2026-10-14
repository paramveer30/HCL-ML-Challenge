"""Picks the fusion weight alpha on dev from the saved dev logits of the two models."""
import json
from pathlib import Path

import numpy as np

from meldrt.config import load_config
from meldrt.data.meld import load_split
from meldrt.eval.metrics import f1_summary
from meldrt.labels import LABEL2ID
from meldrt.models.fusion import fill_missing_audio, late_fuse, softmax, tune_alpha


def main():
    cfg = load_config()
    step = cfg["fusion"]["alpha_step"]

    dev_df = load_split("dev", cfg)
    y = np.array([LABEL2ID[e] for e in dev_df["Emotion"]])
    text_logits = np.load(Path(cfg["text"]["run_dir"]) / "dev_logits.npy")
    audio_logits = np.load(Path(cfg["audio_head"]["run_dir"]) / "dev_logits.npy")

    has_audio = ~np.isnan(audio_logits[:, 0])
    p_text = softmax(text_logits)
    p_audio = fill_missing_audio(p_text, softmax(audio_logits), has_audio)

    alpha, best_f1 = tune_alpha(p_text, p_audio, y, step)

    # the whole curve, to show how flat it is around the best alpha
    grid = []
    for i in range(round(1 / step) + 1):
        a = round(i * step, 6)
        scores = f1_summary(y, late_fuse(p_text, p_audio, a).argmax(axis=1))
        grid.append({"alpha": a, **scores})
        print(f"alpha {a:.2f}  dev weighted F1 {scores['weighted_f1']:.4f}  macro F1 {scores['macro_f1']:.4f}")
    print(f"best alpha {alpha:.2f}, dev weighted F1 {best_f1:.4f}")

    run_dir = Path(cfg["fusion"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "alpha.json", "w", encoding="utf-8") as f:
        json.dump({"alpha": alpha, "dev_weighted_f1": best_f1, "grid": grid}, f, indent=2)


if __name__ == "__main__":
    main()
