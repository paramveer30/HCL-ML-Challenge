"""Input format for the text emotion model: context </s></s> speaker: utterance."""
import numpy as np
import torch


def format_target(speaker, utterance):
    return f"{speaker}: {utterance}"


def encode(tokenizer, contexts, targets, max_tokens):
    """Tokenizes (context, target) pairs. If a pair is too long, the oldest context tokens are dropped first."""
    tokenizer.truncation_side = "left"
    return tokenizer(contexts, targets, truncation="only_first", max_length=max_tokens)


def predict_logits(model, tokenizer, contexts, targets, max_tokens, batch_size=32):
    """Logits for lists of contexts and targets, shape (n, 7). Runs on whatever device the model is on."""
    model.eval()
    chunks = []
    for start in range(0, len(targets), batch_size):
        enc = encode(tokenizer, contexts[start : start + batch_size], targets[start : start + batch_size], max_tokens)
        batch = tokenizer.pad(enc, return_tensors="pt").to(model.device)
        with torch.no_grad():
            chunks.append(model(**batch).logits.cpu().numpy())
    return np.concatenate(chunks)
