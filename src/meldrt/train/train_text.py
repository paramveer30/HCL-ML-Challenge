"""Fine-tunes RoBERTa-base on MELD text with dialogue context. Every choice uses dev; test is never read here."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from meldrt.config import load_config
from meldrt.data.meld import build_context, load_split
from meldrt.eval.metrics import f1_summary
from meldrt.labels import ID2LABEL, LABEL2ID, LABELS
from meldrt.models.text import encode, format_target
from meldrt.train.class_weights import class_weights
from meldrt.utils.seed import set_seed


def make_examples(df, k):
    """One context string, one 'Speaker: utterance' string, and one label id per row."""
    contexts = [build_context(df, i, k) for i in range(len(df))]
    targets = [format_target(speaker, text) for speaker, text in zip(df["Speaker"], df["Utterance"])]
    labels = [LABEL2ID[emotion] for emotion in df["Emotion"]]
    return contexts, targets, labels


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, contexts, targets, labels, max_tokens):
        self.enc = encode(tokenizer, contexts, targets, max_tokens)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids": self.enc["input_ids"][i],
            "attention_mask": self.enc["attention_mask"][i],
            "labels": self.labels[i],
        }


class WeightedTrainer(Trainer):
    """The Hugging Face Trainer, with class weights in the cross-entropy loss."""

    def __init__(self, *args, loss_weights, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_weights = loss_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weights = self.loss_weights.to(outputs.logits.device)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weights)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    return f1_summary(eval_pred.label_ids, eval_pred.predictions.argmax(axis=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="use only the first N train rows (quick check)")
    args = parser.parse_args()

    cfg = load_config()
    t = cfg["text"]
    set_seed(args.seed)

    train_df = load_split("train", cfg)
    dev_df = load_split("dev", cfg)
    if args.limit:
        train_df = train_df.iloc[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(t["model"])
    k = cfg["data"]["context_k"]
    train_set = TextDataset(tokenizer, *make_examples(train_df, k), t["max_tokens"])
    dev_set = TextDataset(tokenizer, *make_examples(dev_df, k), t["max_tokens"])

    counts = np.bincount(train_set.labels, minlength=len(LABELS))
    weights = class_weights(counts.tolist(), t["class_weighting"])
    print("class counts:", dict(zip(LABELS, counts.tolist())))
    print("loss weights:", dict(zip(LABELS, [round(w, 2) for w in weights.tolist()])))

    model = AutoModelForSequenceClassification.from_pretrained(
        t["model"], num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )
    training_args = TrainingArguments(
        output_dir=t["checkpoint_dir"],
        per_device_train_batch_size=t["batch_size"],
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=t["grad_accum"],
        learning_rate=t["lr"],
        num_train_epochs=t["epochs"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="weighted_f1",
        greater_is_better=True,
        fp16=False,
        seed=args.seed,
        logging_steps=100,
        report_to="none",
    )
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=dev_set,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        loss_weights=weights,
    )
    trainer.train()

    # the best epoch by dev weighted F1 is loaded back at this point
    dev_logits = trainer.predict(dev_set).predictions.astype(np.float32)
    dev_scores = f1_summary(dev_set.labels, dev_logits.argmax(axis=1))
    per_epoch = [
        {"epoch": log["epoch"], "weighted_f1": log["eval_weighted_f1"], "macro_f1": log["eval_macro_f1"]}
        for log in trainer.state.log_history
        if "eval_weighted_f1" in log
    ]

    run_dir = Path(t["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "dev_logits.npy", dev_logits)
    summary = {"seed": args.seed, "class_weighting": t["class_weighting"], "per_epoch": per_epoch, "best_dev": dev_scores}
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    best_dir = Path(t["checkpoint_dir"]) / "best"
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
