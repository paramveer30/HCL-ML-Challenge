import pandas as pd

from meldrt.models.text import encode, format_target
from meldrt.train.train_text import TextDataset, make_examples


class FakeTokenizer:
    """Records how it was called and returns one fake id per character of the target."""

    truncation_side = "right"

    def __call__(self, contexts, targets, truncation, max_length):
        self.call = {"contexts": contexts, "targets": targets, "truncation": truncation, "max_length": max_length}
        ids = [list(range(len(t))) for t in targets]
        return {"input_ids": ids, "attention_mask": [[1] * len(i) for i in ids]}


def make_df():
    rows = [
        ("Chandler", "So how was it?", "neutral", 0, 0),
        ("Joey", "It was great.", "joy", 0, 1),
        ("Monica", "Who ate my cake?", "anger", 1, 0),
    ]
    return pd.DataFrame(rows, columns=["Speaker", "Utterance", "Emotion", "Dialogue_ID", "Utterance_ID"])


def test_format_target():
    assert format_target("Joey", "It was great.") == "Joey: It was great."


def test_encode_drops_old_context_first():
    tok = FakeTokenizer()
    encode(tok, ["a"], ["b"], 128)
    assert tok.truncation_side == "left"
    assert tok.call["truncation"] == "only_first"
    assert tok.call["max_length"] == 128
    assert tok.call["contexts"] == ["a"] and tok.call["targets"] == ["b"]


def test_make_examples_pairs_context_target_label():
    contexts, targets, labels = make_examples(make_df(), k=2)
    assert contexts == ["", "Chandler: So how was it?", ""]
    assert targets == ["Chandler: So how was it?", "Joey: It was great.", "Monica: Who ate my cake?"]
    assert labels == [0, 1, 3]


def test_text_dataset_items():
    contexts, targets, labels = make_examples(make_df(), k=2)
    ds = TextDataset(FakeTokenizer(), contexts, targets, labels, 128)
    assert len(ds) == 3
    item = ds[1]
    assert item["labels"] == 1
    assert len(item["input_ids"]) == len("Joey: It was great.")
    assert len(item["attention_mask"]) == len(item["input_ids"])
