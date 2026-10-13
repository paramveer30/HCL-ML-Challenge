from meldrt.pipeline.endpointer import Endpointer


def feed(ep, flags):
    """Pushes a list of flags and returns the list of events."""
    return [ep.push(flag) for flag in flags]


def test_thresholds_in_frames():
    ep = Endpointer(frame_ms=32, start_ms=250, end_silence_ms=600, max_turn_s=15)
    assert ep.start_frames == 8       # 8 * 32 = 256 ms, the first count that reaches 250 ms
    assert ep.end_frames == 19        # 19 * 32 = 608 ms
    assert ep.max_frames == 468       # 468 * 32 = 14,976 ms, the last count that stays under 15 s


def test_silence_never_starts_a_turn():
    assert feed(Endpointer(), [False] * 100) == [None] * 100


def test_turn_starts_on_the_eighth_speech_frame():
    events = feed(Endpointer(), [True] * 8)
    assert events == [None] * 7 + ["start"]


def test_a_short_burst_does_not_start_a_turn():
    # 7 speech frames, a gap, 7 more: the count resets at the gap
    events = feed(Endpointer(), [True] * 7 + [False] + [True] * 7)
    assert "start" not in events


def test_turn_ends_on_the_nineteenth_silent_frame():
    ep = Endpointer()
    feed(ep, [True] * 8)
    events = feed(ep, [False] * 19)
    assert events == [None] * 18 + ["end"]
    assert ep.silence_run == 19
    assert ep.turn_frames == 8 + 19


def test_a_short_pause_does_not_end_the_turn():
    ep = Endpointer()
    feed(ep, [True] * 8)
    events = feed(ep, [False] * 18 + [True] + [False] * 18)
    assert "end" not in events
    assert ep.in_turn


def test_turn_is_forced_to_end_at_the_cap():
    ep = Endpointer()
    events = feed(ep, [True] * 468)
    assert events[7] == "start"
    assert events[-1] == "end"
    assert events.count("end") == 1
    assert ep.turn_frames == 468
    assert ep.silence_run == 0        # ended by the cap, not by silence


def test_a_new_turn_can_start_after_an_end():
    ep = Endpointer()
    feed(ep, [True] * 8 + [False] * 19)
    assert not ep.in_turn
    events = feed(ep, [True] * 8)
    assert events == [None] * 7 + ["start"]
    assert ep.turn_frames == 8


def test_two_full_turns_in_one_stream():
    stream = [False] * 5 + [True] * 20 + [False] * 25 + [True] * 12 + [False] * 19
    events = [e for e in feed(Endpointer(), stream) if e]
    assert events == ["start", "end", "start", "end"]


def test_custom_settings():
    ep = Endpointer(frame_ms=10, start_ms=30, end_silence_ms=50, max_turn_s=1)
    assert (ep.start_frames, ep.end_frames, ep.max_frames) == (3, 5, 100)
    events = feed(ep, [True] * 3 + [False] * 5)
    assert events == [None, None, "start", None, None, None, None, "end"]
