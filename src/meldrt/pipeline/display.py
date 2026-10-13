"""Prints a turn in the terminal: what was heard, what the models think, and the reply as it streams."""
import json


def show_state(state, as_json=False):
    if as_json:
        print(json.dumps(state.to_event(), indent=2))
        return
    views = f"words: {state.text_view.label} {state.text_view.confidence:.0%}"
    if state.audio_view is not None:
        views += f", voice: {state.audio_view.label} {state.audio_view.confidence:.0%}"
    flag = "   << words and voice disagree" if state.tone_text_conflict else ""
    print(f"\nYou:    {state.transcript}")
    print(f"Feels:  {state.label} {state.confidence:.0%}  ({views}){flag}")


def show_reply(events):
    """Prints the reply piece by piece as it arrives. Returns the final 'done' event."""
    print("Robot:  ", end="", flush=True)
    done = None
    for event in events:
        if event["type"] == "token":
            print(event["text"], end="", flush=True)
        else:
            done = event
    print(flush=True)
    return done


def show_timing(state, done, first_audio_ms=None):
    """One line of delays, all counted from the moment the turn was judged to be over."""
    parts = [
        f"state {state.latency_ms['state']:.0f} ms",
        f"first word {done['latency_ms']['first_token']} ms",
        f"full reply {done['latency_ms']['reply']} ms",
    ]
    if first_audio_ms is not None:
        parts.append(f"speaking {first_audio_ms:.0f} ms")
    print("Timing: " + ", ".join(parts) + f"  (after {state.latency_ms['endpoint']:.0f} ms of silence)", flush=True)
