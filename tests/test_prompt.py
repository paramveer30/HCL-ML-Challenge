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


def test_prompt_has_the_words_and_the_emotion():
    prompt = build_prompt(make_state(), [])
    assert "I got the job!" in prompt
    assert "joy" in prompt


def test_prompt_asks_for_a_short_reply():
    prompt = build_prompt(make_state(), []).lower()
    assert "2 sentences" in prompt or "two sentences" in prompt


def test_confident_and_agreeing_means_no_check_in():
    prompt = build_prompt(make_state(confidence=0.8, conflict=False), []).lower()
    assert "check in" not in prompt


def test_conflict_means_check_in_and_names_both_views():
    state = make_state(label="neutral", confidence=0.5, text=("joy", 0.6), audio=("sadness", 0.55), conflict=True,
                       transcript="It's fine.")
    prompt = build_prompt(state, []).lower()
    assert "check in" in prompt
    assert "joy" in prompt and "sadness" in prompt


def test_low_confidence_means_check_in():
    prompt = build_prompt(make_state(confidence=0.40, conflict=False), [], low_conf=0.45).lower()
    assert "check in" in prompt


def test_confidence_at_the_threshold_is_not_low():
    prompt = build_prompt(make_state(confidence=0.45, conflict=False), [], low_conf=0.45).lower()
    assert "check in" not in prompt


def test_history_comes_before_the_current_turn_in_order():
    prompt = build_prompt(make_state(), HISTORY)
    positions = [prompt.index(text) for text in
                 ["I had an interview today.", "How did it go?", "I think it went fine.", "Fingers crossed for you.",
                  "I got the job!"]]
    assert positions == sorted(positions)


def test_works_when_there_is_no_audio_view():
    prompt = build_prompt(make_state(audio=None), [])
    assert "I got the job!" in prompt
