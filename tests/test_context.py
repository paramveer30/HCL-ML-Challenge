import pandas as pd

from meldrt.data.meld import build_context


def make_df():
    # two dialogues, rows in file order like the real CSV
    rows = [
        ("Chandler", "Hey.", 0, 0),
        ("Joey", "Hey!", 0, 1),
        ("Chandler", "So how was it?", 0, 2),
        ("Joey", "It was great.", 0, 3),
        ("Monica", "Who ate my cake?", 1, 0),
        ("Rachel", "Not me.", 1, 1),
    ]
    return pd.DataFrame(rows, columns=["Speaker", "Utterance", "Dialogue_ID", "Utterance_ID"])


def test_context_is_previous_two_in_order():
    assert build_context(make_df(), 3, k=2) == "Joey: Hey!\nChandler: So how was it?"


def test_context_of_first_utterance_is_empty():
    assert build_context(make_df(), 0, k=2) == ""


def test_context_is_shorter_near_dialogue_start():
    assert build_context(make_df(), 1, k=2) == "Chandler: Hey."


def test_context_never_crosses_into_another_dialogue():
    df = make_df()
    assert build_context(df, 4, k=2) == ""
    assert build_context(df, 5, k=2) == "Monica: Who ate my cake?"


def test_context_never_includes_current_or_future():
    ctx = build_context(make_df(), 2, k=2)
    assert "So how was it?" not in ctx
    assert "It was great." not in ctx


def test_context_respects_k():
    df = make_df()
    assert build_context(df, 3, k=1) == "Chandler: So how was it?"
    assert build_context(df, 3, k=0) == ""


def test_context_uses_utterance_id_not_row_order():
    # same dialogue, rows shuffled: context must still be the earlier utterances
    df = make_df().iloc[[3, 1, 0, 2, 4, 5]].reset_index(drop=True)
    assert build_context(df, 0, k=2) == "Joey: Hey!\nChandler: So how was it?"
