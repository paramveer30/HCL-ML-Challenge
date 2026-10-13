from meldrt.pipeline.respond import build_prompt
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


HISTORY = [
    {"transcript": "I had an interview today.", "label": "neutral", "reply": "How did it go?"},
    {"transcript": "I think it went fine.", "label": "neutral", "reply": "Fingers crossed for you."},
]


def test_first_message_is_the_rules_and_asks_for_a_short_reply():
    messages = build_prompt(make_state(), [])
    assert messages[0]["role"] == "system"
    assert "2 short sentences" in messages[0]["content"]
    assert len(messages) == 2


def test_last_message_has_the_words_and_the_emotion():
    last = build_prompt(make_state(), [])[-1]
    assert last["role"] == "user"
    assert last["content"].startswith("I got the job!")
    assert "they sound joy" in last["content"]


def test_history_becomes_real_chat_turns_in_order():
    messages = build_prompt(make_state(), HISTORY)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert [m["content"] for m in messages[1:5]] == [
        "I had an interview today.", "How did it go?", "I think it went fine.", "Fingers crossed for you.",
    ]
    # earlier turns carry no note; only the current one does
    assert all("[" not in m["content"] for m in messages[1:5])


def test_confident_and_agreeing_means_no_check_in():
    note = build_prompt(make_state(confidence=0.8, conflict=False), [])[-1]["content"]
    assert "do not assume" not in note and "unclear" not in note


def test_conflict_names_both_views_and_asks_not_to_assume():
    state = make_state(label="neutral", confidence=0.5, text=("joy", 0.6), audio=("sadness", 0.55), conflict=True,
                       transcript="It's fine.")
    note = build_prompt(state, [])[-1]["content"]
    assert "your words sound joy" in note and "your voice sounds sadness" in note
    assert "do not assume" in note
    assert "ask one short question" in note


def test_low_confidence_means_check_in_without_naming_a_feeling():
    note = build_prompt(make_state(label="sadness", confidence=0.40, conflict=False), [], low_conf=0.45)[-1]["content"]
    assert "unclear" in note and "check in" in note
    assert "sadness" not in note


def test_confidence_at_the_threshold_is_not_low():
    note = build_prompt(make_state(confidence=0.45, conflict=False), [], low_conf=0.45)[-1]["content"]
    assert "unclear" not in note


def test_works_when_there_is_no_audio_view():
    last = build_prompt(make_state(audio=None), [])[-1]
    assert last["content"].startswith("I got the job!")
