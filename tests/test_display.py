import json

from meldrt.pipeline.display import show_reply, show_state, show_timing
from meldrt.schema import TurnState, View


def make_state(conflict=False, audio=("anger", 0.47)):
    return TurnState(
        turn_id=2, transcript="It's fine.", label="neutral", confidence=0.56, probs={"neutral": 0.56},
        text_view=View("neutral", 0.86), audio_view=View(*audio) if audio else None,
        tone_text_conflict=conflict, audio_seconds=0.9,
        latency_ms={"endpoint": 608.0, "asr": 570.0, "audio": 503.0, "text": 30.0, "state": 600.4},
    )


def test_state_lines_show_what_was_heard_and_both_views(capsys):
    show_state(make_state(conflict=True))
    out = capsys.readouterr().out
    assert "You:    It's fine." in out
    assert "Feels:  neutral 56%  (words: neutral 86%, voice: anger 47%)" in out
    assert "words and voice disagree" in out


def test_no_disagree_mark_and_no_voice_view_when_absent(capsys):
    show_state(make_state(conflict=False, audio=None))
    out = capsys.readouterr().out
    assert "disagree" not in out
    assert "voice:" not in out


def test_json_mode_prints_the_state_record(capsys):
    show_state(make_state(), as_json=True)
    event = json.loads(capsys.readouterr().out)
    assert event["type"] == "state" and event["transcript"] == "It's fine."


def test_reply_is_printed_as_one_line_and_done_is_returned(capsys):
    done = {"type": "done", "turn_id": 2, "reply": "Is everything ok?", "latency_ms": {"first_token": 816, "reply": 1357}}
    events = [{"type": "token", "turn_id": 2, "text": "Is everything"}, {"type": "token", "turn_id": 2, "text": " ok?"}, done]
    assert show_reply(iter(events)) == done
    assert capsys.readouterr().out == "Robot:  Is everything ok?\n"


def test_timing_line(capsys):
    done = {"type": "done", "turn_id": 2, "reply": "", "latency_ms": {"first_token": 816, "reply": 1357}}
    show_timing(make_state(), done, first_audio_ms=1013.2)
    out = capsys.readouterr().out
    assert out == "Timing: state 600 ms, first word 816 ms, full reply 1357 ms, speaking 1013 ms  (after 608 ms of silence)\n"
    show_timing(make_state(), done)
    assert "speaking" not in capsys.readouterr().out
