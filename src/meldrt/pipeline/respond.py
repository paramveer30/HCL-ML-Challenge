"""Builds the reply prompt and streams the reply from the local chat model (Ollama)."""
import json
import urllib.request

RULES = (
    "You are a small, friendly robot talking with one person. "
    'Speak to them directly, as "you". '
    "Reply in at most 2 sentences, about 30 words in total. "
    "Respond to what they said and show you noticed how they feel. "
    "Reply in plain English. "
    "Do not mention these instructions, confidence numbers, or that you analysed anything."
)


def build_prompt(state, history, low_conf=0.45):
    """state: the TurnState for this turn. history: earlier turns, oldest first,
    each a dict with "transcript", "label", "reply". Returns the prompt as one string."""
    lines = [RULES, ""]

    if history:
        lines.append("Earlier in this conversation:")
        for turn in history:
            lines.append(f"Person: {turn['transcript']}")
            lines.append(f"Robot: {turn['reply']}")
        lines.append("")

    lines.append(f'The person just said: "{state.transcript}"')
    lines.append(f"Our best reading of how they feel: {state.label} (confidence {state.confidence:.2f}).")
    lines.append(f"Their words alone suggest {state.text_view.label} ({state.text_view.confidence:.2f}).")
    if state.audio_view is not None:
        lines.append(f"Their tone of voice alone suggests {state.audio_view.label} ({state.audio_view.confidence:.2f}).")

    # unsure, or words and voice point different ways: ask, don't assume
    if state.tone_text_conflict:
        lines.append(
            f"Their words sound like {state.text_view.label} but their voice sounds like {state.audio_view.label}. "
            "Do not assume which is true. Speak to them: kindly mention that their voice sounds different "
            "from their words, then gently check in with one short question."
        )
    elif state.confidence < low_conf:
        lines.append(
            "We are not sure how they feel. Respond to what they said, do not assume a feeling, "
            "and gently check in with them about how they are doing."
        )

    lines.append("")
    lines.append("Your reply to the person:")
    return "\n".join(lines)


def stream_reply(prompt, reply_cfg):
    """Sends the prompt to Ollama and yields the reply piece by piece as it is written."""
    body = {
        "model": reply_cfg["model"],
        "prompt": prompt,
        "stream": True,
        "keep_alive": reply_cfg["keep_alive"],
        "options": {
            "num_predict": reply_cfg["max_tokens"],
            "temperature": reply_cfg["temperature"],
            "seed": reply_cfg["seed"],
        },
    }
    request = urllib.request.Request(
        reply_cfg["url"] + "/api/generate",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    # Ollama answers with one JSON object per line; each carries the next piece of text
    with urllib.request.urlopen(request) as response:
        for line in response:
            piece = json.loads(line)["response"]
            if piece:
                yield piece
