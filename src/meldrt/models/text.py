"""Input format for the text emotion model: context </s></s> speaker: utterance."""


def format_target(speaker, utterance):
    return f"{speaker}: {utterance}"


def encode(tokenizer, contexts, targets, max_tokens):
    """Tokenizes (context, target) pairs. If a pair is too long, the oldest context tokens are dropped first."""
    tokenizer.truncation_side = "left"
    return tokenizer(contexts, targets, truncation="only_first", max_length=max_tokens)
