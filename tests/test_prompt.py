from meldrt.pipeline.respond import EXAMPLES, build_prompt, how_they_seem
from meldrt.schema import TurnState, View


def make_state(label="joy", confidence=0.8, text=("joy", 0.8), audio=("joy", 0.7), conflict=False,
               transcript="I got the job!"):
    return TurnState(
        turn_id=3,
        transcript=transcript,
        label=label,
        confidence=confidence,
        probs={},
        text_view=View(*text),
        audio_view=View(*audio) if audio else None,
        tone_text_conflict=conflict,
        audio_seconds=1.5,
    )


def with_face(state, label, conflict):
    state.vision_view = View(label, 0.7)
    state.face_text_conflict = conflict
    return state


HISTORY = [
    {"transcript": "I had an interview today.", "label": "neutral", "reply": "How did it go?"},
    {"transcript": "I think it went fine.", "label": "neutral", "reply": "Fingers crossed for you."},
]
FIRST_REAL = 1 + 2 * len(EXAMPLES)      # the rules, then the example turns, then the conversation


def test_first_message_is_the_rules_and_asks_for_a_short_reply():
    messages = build_prompt(make_state(), [])
    assert messages[0]["role"] == "system"
    assert "one or two short spoken sentences" in messages[0]["content"]
    assert "never ask how you can help" in messages[0]["content"]


def test_the_example_turns_come_before_the_conversation():
    messages = build_prompt(make_state(), [])
    assert len(messages) == FIRST_REAL + 1
    assert [m["role"] for m in messages[1:FIRST_REAL]] == ["user", "assistant"] * len(EXAMPLES)
    # every example shows the same shape as a real turn: the line in brackets, then the words
    assert all(m["content"].startswith("(") for m in messages[1:FIRST_REAL:2])


def test_last_message_has_how_they_seem_and_then_the_words():
    last = build_prompt(make_state(), [])[-1]
    assert last["role"] == "user"
    assert last["content"] == "(They seem happy.)\nI got the job!"


def test_history_becomes_real_chat_turns_in_order():
    messages = build_prompt(make_state(), HISTORY)[FIRST_REAL:]
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant", "user"]
    assert [m["content"] for m in messages[:4]] == [
        "I had an interview today.", "How did it go?", "I think it went fine.", "Fingers crossed for you.",
    ]
    # earlier turns carry no bracket line; only the current one does
    assert all("(" not in m["content"] for m in messages[:4])


def test_a_calm_person_gives_the_model_nothing_to_comment_on():
    assert how_they_seem(make_state(label="neutral", text=("neutral", 0.8), audio=("neutral", 0.6)), 0.5) == \
        "Nothing stands out in how they sound."


def test_low_confidence_says_it_is_hard_to_tell():
    assert how_they_seem(make_state(confidence=0.3), 0.5) == "It is hard to tell how they feel."
    assert how_they_seem(make_state(confidence=0.6), 0.5) == "They seem happy."


def test_a_voice_that_disagrees_with_the_words_is_named():
    state = make_state(label="neutral", confidence=0.5, text=("joy", 0.6), audio=("sadness", 0.55), conflict=True,
                       transcript="It's fine.")
    assert how_they_seem(state, 0.5) == "Their words sound happy, but their voice sounds sad."


def test_a_conflict_matters_more_than_low_confidence():
    state = make_state(label="neutral", confidence=0.2, text=("joy", 0.6), audio=("anger", 0.55), conflict=True)
    assert "their voice sounds angry" in how_they_seem(state, 0.5)


def test_a_face_that_disagrees_with_the_words_is_named():
    state = with_face(make_state(label="joy", text=("joy", 0.6), audio=("joy", 0.6), transcript="I am fine."), "sadness", True)
    assert how_they_seem(state, 0.5) == "Their words sound happy, but their face looks sad."


def test_voice_and_face_can_both_disagree_with_the_words():
    state = with_face(make_state(label="neutral", confidence=0.5, text=("joy", 0.6), audio=("anger", 0.55), conflict=True,
                                 transcript="It's fine."), "sadness", True)
    assert how_they_seem(state, 0.5) == "Their words sound happy, but their voice sounds angry and their face looks sad."


def test_a_face_that_agrees_changes_nothing():
    plain = build_prompt(make_state(), [])[-1]["content"]
    assert build_prompt(with_face(make_state(), "joy", False), [])[-1]["content"] == plain
