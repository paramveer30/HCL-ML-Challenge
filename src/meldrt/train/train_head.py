"""Trains an emotion head on cached features of a frozen encoder: WavLM for audio, the face ViT for vision.

The two heads are the same model trained the same way; only the features differ.
Every choice uses dev; test is never read here.
"""
import copy
import json
from pathlib import Path

import numpy as np
import torch

from meldrt.data.features import load_features
from meldrt.data.meld import load_split
from meldrt.eval.metrics import f1_summary
from meldrt.labels import LABEL2ID, LABELS
from meldrt.models.head import LayerMixHead
from meldrt.train.class_weights import class_weights
from meldrt.utils.seed import set_seed


def predict(head, feats):
    """feats: tensor (n, 13, 768). Returns logits as a numpy array (n, 7), with dropout off."""
    head.eval()
    with torch.no_grad():
        return head(feats).cpu().numpy()


def train_head(cfg, head_cfg, feature_root, seed, what):
    """head_cfg: the head's section of the config. feature_root: the folder its cached features are in.
    what: what a row needs to have features, for the messages ("audio" or "a face")."""
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_df = load_split("train", cfg)
    dev_df = load_split("dev", cfg)
    train_feats, train_has = load_features(train_df, "train", cfg, feature_root)
    dev_feats, dev_has = load_features(dev_df, "dev", cfg, feature_root)
    # rows without features are left out here; fusion falls back to the other models for them
    train_labels = np.array([LABEL2ID[e] for e in train_df["Emotion"]])[train_has]
    dev_labels = np.array([LABEL2ID[e] for e in dev_df["Emotion"]])[dev_has]
    print(f"train: {len(train_labels)} of {len(train_df)} rows have {what}; dev: {len(dev_labels)} of {len(dev_df)}")

    train_x = torch.from_numpy(train_feats).to(device)
    train_y = torch.from_numpy(train_labels).to(device)
    dev_x = torch.from_numpy(dev_feats).to(device)

    counts = np.bincount(train_labels, minlength=len(LABELS))
    weights = class_weights(counts.tolist(), head_cfg["class_weighting"]).to(device)

    head = LayerMixHead(hidden=head_cfg["hidden"], dropout=head_cfg["dropout"]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=head_cfg["lr"])

    best_f1, best_state, best_logits, per_epoch = -1.0, None, None, []
    for epoch in range(1, head_cfg["epochs"] + 1):
        head.train()
        order = torch.randperm(len(train_x), device=device)
        for start in range(0, len(order), head_cfg["batch_size"]):
            batch = order[start : start + head_cfg["batch_size"]]
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

    checkpoint = Path(head_cfg["checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint)

    # one row per dev utterance, NaN where there are no features, so it lines up with the other models' logits
    all_logits = np.full((len(dev_df), len(LABELS)), np.nan, dtype=np.float32)
    all_logits[dev_has] = best_logits
    run_dir = Path(head_cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "dev_logits.npy", all_logits)

    head.load_state_dict(best_state)
    summary = {
        "seed": seed,
        "class_weighting": head_cfg["class_weighting"],
        "train_rows_with_features": int(train_has.sum()),
        "dev_rows_with_features": int(dev_has.sum()),
        "best_epoch": max(per_epoch, key=lambda e: e["weighted_f1"])["epoch"],
        "best_dev_rows_with_features": f1_summary(dev_labels, best_logits.argmax(axis=1)),
        "layer_weights": [round(w, 4) for w in head.layer_weights().tolist()],
        "per_epoch": per_epoch,
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_epoch"}, indent=2))
