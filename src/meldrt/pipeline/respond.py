"""Builds the chat prompt for the reply and streams the reply from the local chat model (Ollama)."""
import json
import urllib.request

RULES = (
    "You are a small, friendly robot having a spoken conversation with one person. "
    'Speak to them directly, as "you". '
    "Reply in at most 2 short sentences, in plain English. "
    "Respond to what they just said and show you noticed how they feel. "
    "Each of their messages ends with a private note in square brackets about how they sound. "
    "Follow the note, but never read it out, and never mention notes, confidence, or analysis. "
    "Do not greet them unless they just greeted you, and do not repeat your earlier answers."
)


def state_note(state, low_conf):
    """One line telling the chat model what we think the person feels, and what to do about it."""
    # unsure, or words and voice point different ways: ask, don't assume
    if state.tone_text_conflict or state.face_text_conflict:
        # name each thing that disagrees with the words: the voice, the face, or both
        heard, seen = [], []
        if state.tone_text_conflict:
            heard, seen = [f"your voice sounds {state.audio_view.label}"], [f"sound {state.audio_view.label}"]
        if state.face_text_conflict:
            heard.append(f"your face looks {state.vision_view.label}")
            seen.append(f"look {state.vision_view.label}")
        return (
            f"Private note: your words sound {state.text_view.label}, but {' and '.join(heard)}. "
            "Robot: do not assume which is true. Tell them kindly that they "
            f"{' and '.join(seen)} even though their words say otherwise, then ask one short question."
        )
    if state.confidence < low_conf:
        return (
            "Private note: it is unclear how they feel. "
            "Robot: respond to what they said, do not name a feeling, and gently check in."
        )
    return f"Private note: they sound {state.label}."


def build_prompt(state, history, low_conf=0.45):
    """state: the TurnState for this turn. history: earlier turns, oldest first,
    each a dict with "transcript", "label", "reply". Returns the list of chat messages."""
    messages = [{"role": "system", "content": RULES}]
    for turn in history:
        messages.append({"role": "user", "content": turn["transcript"]})
        messages.append({"role": "assistant", "content": turn["reply"]})
    messages.append({"role": "user", "content": f"{state.transcript}\n[{state_note(state, low_conf)}]"})
    return messages


def stream_reply(messages, reply_cfg):
    """Sends the chat messages to Ollama and yields the reply piece by piece as it is written."""
    body = {
        "model": reply_cfg["model"],
        "messages": messages,
        "stream": True,
        "keep_alive": reply_cfg["keep_alive"],
        "options": {
            "num_predict": reply_cfg["max_tokens"],
            "temperature": reply_cfg["temperature"],
            "seed": reply_cfg["seed"],
        },
    }
    request = urllib.request.Request(
        reply_cfg["url"] + "/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    # Ollama answers with one JSON object per line; each carries the next piece of text
    with urllib.request.urlopen(request) as response:
        for line in response:
            piece = json.loads(line)["message"]["content"]
            if piece:
                yield piece
