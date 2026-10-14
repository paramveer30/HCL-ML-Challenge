"""Trains the vision head on cached face features. Every choice uses dev; test is never read here.

The head and the training loop are the same as for audio (train_audio.py): a learned mix of the
13 layers of a frozen encoder, then a small MLP. Only the features differ.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from meldrt.config import load_config
from meldrt.data.features import load_features
from meldrt.data.meld import load_split
from meldrt.eval.metrics import f1_summary
from meldrt.labels import LABEL2ID, LABELS
from meldrt.models.audio import AudioHead
from meldrt.train.class_weights import class_weights
from meldrt.train.train_audio import predict
from meldrt.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config()
    h = cfg["vision_head"]
    root = cfg["vision"]["feature_root"]
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_df = load_split("train", cfg)
    dev_df = load_split("dev", cfg)
    train_feats, train_has = load_features(train_df, "train", cfg, root)
    dev_feats, dev_has = load_features(dev_df, "dev", cfg, root)
    # rows where no face was found are left out here; fusion falls back to the other models for them
    train_labels = np.array([LABEL2ID[e] for e in train_df["Emotion"]])[train_has]
    dev_labels = np.array([LABEL2ID[e] for e in dev_df["Emotion"]])[dev_has]
    print(f"train: {len(train_labels)} of {len(train_df)} rows have a face; dev: {len(dev_labels)} of {len(dev_df)}")

    train_x = torch.from_numpy(train_feats).to(device)
    train_y = torch.from_numpy(train_labels).to(device)
    dev_x = torch.from_numpy(dev_feats).to(device)

    counts = np.bincount(train_labels, minlength=len(LABELS))
    weights = class_weights(counts.tolist(), h["class_weighting"]).to(device)

    head = AudioHead(hidden=h["hidden"], dropout=h["dropout"]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=h["lr"])

    best_f1, best_state, best_logits, per_epoch = -1.0, None, None, []
    for epoch in range(1, h["epochs"] + 1):
        head.train()
        order = torch.randperm(len(train_x), device=device)
        for start in range(0, len(order), h["batch_size"]):
            batch = order[start : start + h["batch_size"]]
            loss = torch.nn.functional.cross_entropy(head(train_x[batch]), train_y[batch], weight=weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        dev_logits = predict(head, dev_x)
        scores = f1_summary(dev_labels, dev_logits.argmax(axis=1))
        per_epoch.append({"epoch": epoch, **scores})
        print(f"epoch {epoch:2d}  dev weighted F1 {scores['weighted_f1']:.4f}  macro F1 {scores['macro_f1']:.4f}")
        if scores["weighted_f1"] > best_f1:
            best_f1 = scores["weighted_f1"]
            best_state = copy.deepcopy(head.state_dict())
            best_logits = dev_logits

    checkpoint = Path(h["checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint)

    # one row per dev utterance, NaN where there is no face, so it lines up with the other models' logits
    all_logits = np.full((len(dev_df), len(LABELS)), np.nan, dtype=np.float32)
    all_logits[dev_has] = best_logits
    run_dir = Path(h["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "dev_logits.npy", all_logits)

    head.load_state_dict(best_state)
    summary = {
        "seed": args.seed,
        "class_weighting": h["class_weighting"],
        "train_rows_with_a_face": int(train_has.sum()),
        "dev_rows_with_a_face": int(dev_has.sum()),
        "best_epoch": max(per_epoch, key=lambda e: e["weighted_f1"])["epoch"],
        "best_dev_rows_with_a_face": f1_summary(dev_labels, best_logits.argmax(axis=1)),
        "layer_weights": [round(w, 4) for w in head.layer_weights().tolist()],
        "per_epoch": per_epoch,
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_epoch"}, indent=2))


if __name__ == "__main__":
    main()
