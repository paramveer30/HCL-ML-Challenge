"""The only script that reads the test split. Scores majority, text only, audio only, and fused.

Everything it uses was fixed on dev beforehand: both checkpoints and alpha.
Run with --split dev to check the script itself without touching test.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from meldrt.config import load_config
from meldrt.data.features import load_features
from meldrt.data.meld import load_split
from meldrt.eval.ablation import ablation_predictions
from meldrt.eval.metrics import confusion, f1_summary, per_class_f1
from meldrt.labels import LABEL2ID, LABELS
from meldrt.models.audio import AudioHead
from meldrt.models.fusion import softmax
from meldrt.models.text import predict_logits
from meldrt.train.train_audio import predict
from meldrt.train.train_text import make_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    args = parser.parse_args()

    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = load_split(args.split, cfg)
    y = np.array([LABEL2ID[e] for e in df["Emotion"]])

    # majority class comes from train, never from the split being scored
    train_labels = [LABEL2ID[e] for e in load_split("train", cfg)["Emotion"]]
    majority_class = int(np.bincount(train_labels).argmax())

    # text model
    best_dir = Path(cfg["text"]["checkpoint_dir"]) / "best"
    tokenizer = AutoTokenizer.from_pretrained(best_dir)
    text_model = AutoModelForSequenceClassification.from_pretrained(best_dir).to(device)
    contexts, targets, _ = make_examples(df, cfg["data"]["context_k"])
    p_text = softmax(predict_logits(text_model, tokenizer, contexts, targets, cfg["text"]["max_tokens"]))

    # audio head on the cached features
    a = cfg["audio_head"]
    feats, has_audio = load_features(df, args.split, cfg)
    head = AudioHead(hidden=a["hidden"], dropout=a["dropout"]).to(device)
    head.load_state_dict(torch.load(a["checkpoint"], map_location=device))
    p_audio = np.full((len(df), len(LABELS)), np.nan, dtype=np.float32)
    p_audio[has_audio] = softmax(predict(head, torch.from_numpy(feats).to(device)))

    with open(Path(cfg["fusion"]["run_dir"]) / "alpha.json", encoding="utf-8") as f:
        alpha = json.load(f)["alpha"]

    rows = ablation_predictions(p_text, p_audio, has_audio, alpha, majority_class)

    print(f"\nsplit: {args.split}   rows: {len(df)}   rows with audio: {int(has_audio.sum())}   alpha: {alpha}")
    print(f"majority class (from train): {LABELS[majority_class]}\n")
    print("| Model | Weighted F1 | Macro F1 |")
    print("| --- | --- | --- |")
    results = {}
    for name, predictions in rows.items():
        scores = f1_summary(y, predictions)
        results[name] = {**scores, "per_class": per_class_f1(y, predictions)}
        print(f"| {name} | {scores['weighted_f1']:.4f} | {scores['macro_f1']:.4f} |")

    print("\n| Emotion | n | Text only | Audio only | Fused |")
    print("| --- | --- | --- | --- | --- |")
    counts = np.bincount(y, minlength=len(LABELS))
    for i, label in enumerate(LABELS):
        f1s = [results[name]["per_class"][label] for name in ["text only", "audio only", "fused"]]
        print(f"| {label} | {counts[i]} | {f1s[0]:.3f} | {f1s[1]:.3f} | {f1s[2]:.3f} |")

    matrix = confusion(y, rows["fused"])
    print("\nconfusion matrix for fused (rows: true, columns: predicted)")
    print(" " * 10 + " ".join(f"{label[:7]:>8s}" for label in LABELS))
    for label, row in zip(LABELS, matrix):
        print(f"{label:>9s} " + " ".join(f"{n:8d}" for n in row))

    run_dir = Path(cfg["eval"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "split": args.split,
        "rows": len(df),
        "rows_with_audio": int(has_audio.sum()),
        "alpha": alpha,
        "results": results,
        "fused_confusion": matrix.tolist(),
    }
    with open(run_dir / f"{args.split}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
