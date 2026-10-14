"""Fits one temperature on dev so the fused confidence matches how often the fused label is right.

Averaging a sharp text distribution with a flat audio one gives a fused confidence that is too low:
the label is right more often than the number says. The temperature fixes the number and never
changes the label, so no F1 score depends on it.
"""
import json
from pathlib import Path

import numpy as np

from meldrt.config import load_config
from meldrt.data.meld import load_split
from meldrt.labels import LABEL2ID
from meldrt.models.fusion import fill_missing_audio, fit_temperature, late_fuse, sharpen, softmax

BINS = [(0.0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]


def reliability(probs, y_true):
    """For each confidence bin: how many rows fall in it and how often their top label is right."""
    confidence, right = probs.max(axis=1), probs.argmax(axis=1) == y_true
    rows = []
    for low, high in BINS:
        inside = (confidence >= low) & (confidence < high)
        if inside.any():
            rows.append({"from": low, "to": min(high, 1.0), "rows": int(inside.sum()),
                         "mean_confidence": float(confidence[inside].mean()), "accuracy": float(right[inside].mean())})
    # the usual one-number summary: the gap between confidence and accuracy, weighted by bin size
    gap = sum(row["rows"] * abs(row["mean_confidence"] - row["accuracy"]) for row in rows) / len(y_true)
    return {"accuracy": float(right.mean()), "mean_confidence": float(confidence.mean()), "calibration_error": float(gap), "bins": rows}


def show(title, table):
    print(f"{title}: accuracy {table['accuracy']:.3f}, mean confidence {table['mean_confidence']:.3f}, "
          f"calibration error {table['calibration_error']:.3f}")
    for row in table["bins"]:
        print(f"  confidence {row['from']:.1f} to {row['to']:.1f}: right {row['accuracy']:.2f} of the time ({row['rows']} rows)")


def main():
    cfg = load_config()
    run_dir = Path(cfg["fusion"]["run_dir"])

    dev_df = load_split("dev", cfg)
    y = np.array([LABEL2ID[e] for e in dev_df["Emotion"]])
    p_text = softmax(np.load(Path(cfg["text"]["run_dir"]) / "dev_logits.npy"))
    audio_logits = np.load(Path(cfg["audio_head"]["run_dir"]) / "dev_logits.npy")
    has_audio = ~np.isnan(audio_logits[:, 0])
    with open(run_dir / "alpha.json", encoding="utf-8") as f:
        alpha = json.load(f)["alpha"]
    fused = late_fuse(p_text, fill_missing_audio(p_text, softmax(audio_logits), has_audio), alpha)

    temperature, before, after = fit_temperature(fused, y)
    print(f"temperature {temperature:.2f}: average negative log probability {before:.3f} -> {after:.3f}")
    table_before, table_after = reliability(fused, y), reliability(sharpen(fused, temperature), y)
    show("before", table_before)
    show("after", table_after)

    # is the temperature stable? fit on the dialogues with an even number, check on the odd ones, and the other way
    even = dev_df["Dialogue_ID"].to_numpy() % 2 == 0
    halves = []
    for name, fit_on in [("even", even), ("odd", ~even)]:
        t, _, _ = fit_temperature(fused[fit_on], y[fit_on])
        held_out = reliability(sharpen(fused[~fit_on], t), y[~fit_on])
        halves.append({"fitted_on": name, "temperature": t, "held_out_accuracy": held_out["accuracy"],
                       "held_out_mean_confidence": held_out["mean_confidence"]})
        print(f"fitted on {name} dialogues: temperature {t:.2f}; on the other half accuracy {held_out['accuracy']:.3f}, "
              f"mean confidence {held_out['mean_confidence']:.3f}")

    with open(run_dir / "calibration.json", "w", encoding="utf-8") as f:
        json.dump({"temperature": temperature, "before": table_before, "after": table_after, "halves": halves}, f, indent=2)


if __name__ == "__main__":
    main()
