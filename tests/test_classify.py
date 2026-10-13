from meldrt.pipeline.classify import keep_end


class WordTokenizer:
    """One token per word, so the test needs no real tokenizer."""

    def __call__(self, text, add_special_tokens):
        self.words = text.split()
        return {"input_ids": list(range(len(self.words)))}

    def decode(self, ids):
        return " ".join(self.words[i] for i in ids)


def test_a_target_that_fits_is_left_alone():
    assert keep_end(WordTokenizer(), "Speaker: it is fine", 10) == "Speaker: it is fine"
    assert keep_end(WordTokenizer(), "one two three", 3) == "one two three"


def test_a_target_that_is_too_long_keeps_its_end():
    assert keep_end(WordTokenizer(), "one two three four five", 3) == "three four five"
