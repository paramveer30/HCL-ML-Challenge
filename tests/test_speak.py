from meldrt.pipeline.speak import SentenceChunker, speak_while_streaming


def test_sentence_is_released_when_the_next_one_starts():
    chunker = SentenceChunker()
    assert chunker.push("Hello") == []
    assert chunker.push(" there!") == []          # could still continue, nothing follows yet
    assert chunker.push(" How") == ["Hello there!"]
    assert chunker.push(" are you?") == []
    assert chunker.flush() == "How are you?"
    assert chunker.flush() is None


def test_several_sentences_in_one_piece():
    chunker = SentenceChunker()
    assert chunker.push("One. Two! Three? Fou") == ["One.", "Two!", "Three?"]
    assert chunker.flush() == "Fou"


def test_a_number_with_a_dot_is_not_split():
    chunker = SentenceChunker()
    assert chunker.push("It costs 3.50 today. Ok") == ["It costs 3.50 today."]


def test_flush_on_empty_is_none():
    assert SentenceChunker().flush() is None


class FakeSpeaker:
    def __init__(self):
        self.said = []
        self.started = 0

    def start_reply(self):
        self.started += 1

    def say(self, sentence):
        self.said.append(sentence)


def token(text):
    return {"type": "token", "turn_id": 1, "text": text}


def test_first_sentence_is_spoken_before_the_reply_has_finished():
    speaker = FakeSpeaker()
    events = [token("I'm"), token(" sorry."), token(" Are"), token(" you ok?"), {"type": "done", "turn_id": 1}]
    said_so_far = []
    passed_on = []
    for event in speak_while_streaming(iter(events), speaker):
        passed_on.append(event)
        said_so_far.append(list(speaker.said))

    assert passed_on == events                       # the text stream is untouched
    assert speaker.started == 1
    assert said_so_far[1] == []                      # after " sorry." nothing yet: the sentence might go on
    assert said_so_far[2] == ["I'm sorry."]          # the moment the next sentence starts, the first one is sent
    assert speaker.said == ["I'm sorry.", "Are you ok?"]
