"""Builds the chat prompt for the reply and streams the reply from the local chat model (Ollama)."""
import json
import urllib.request

RULES = (
    "You are a small companion robot sitting next to one person, talking out loud with them. "
    "Answer in one or two short spoken sentences.\n"
    "Every message from them starts with a line in round brackets that says how they seem. "
    "Let it set your tone. Never repeat or mention that line.\n"
    "A good answer picks up a concrete detail from what they just said and reacts to it the way a friend would: "
    "glad about good news, gentle about bad news, calm when they are angry. "
    "You may add one short question about that same detail. If they ask you something, just answer it.\n"
    "Only talk about how they sound or look when that line says their words and their voice or face disagree. "
    "If nothing stands out, or it is hard to tell, just answer what they said.\n"
    "Never offer help or to fetch anything, never ask how you can help, never say you understand, never give advice, "
    "never greet them unless they greeted you, and never say something you already said."
)

# a few example turns, given to the model as if they had happened, so it copies their shape
EXAMPLES = [
    ("(They seem happy.)\nI finally passed my driving test!", "You passed! No more waiting for the bus. Where are you driving first?"),
    ("(They seem sad.)\nMy flight got cancelled and I'm stuck here overnight.", "Stuck overnight, that's rough. Did they at least give you a hotel?"),
    ("(Their words sound calm, but their voice sounds sad.)\nIt's fine.", "You say it's fine, but you don't sound fine. What happened?"),
    ("(Nothing stands out in how they sound.)\nHey there.", "Hey! Good to see you. What are you up to?"),
]

# how each label reads in a sentence about a person
SEEMS = {"neutral": "calm", "joy": "happy", "surprise": "surprised", "anger": "angry",
         "sadness": "sad", "disgust": "disgusted", "fear": "worried"}


def how_they_seem(state, low_conf):
    """One short sentence telling the chat model how the person seems, from the three views."""
    if state.tone_text_conflict or state.face_text_conflict:
        # the voice, the face, or both say something else than the words
        others = []
        if state.tone_text_conflict:
            others.append(f"their voice sounds {SEEMS[state.audio_view.label]}")
        if state.face_text_conflict:
            others.append(f"their face looks {SEEMS[state.vision_view.label]}")
        return f"Their words sound {SEEMS[state.text_view.label]}, but {' and '.join(others)}."
    if state.confidence < low_conf:
        return "It is hard to tell how they feel."
    if state.label == "neutral":
        return "Nothing stands out in how they sound."
    return f"They seem {SEEMS[state.label]}."


def build_prompt(state, history, low_conf=0.5):
    """state: the TurnState for this turn. history: earlier turns, oldest first,
    each a dict with "transcript", "label", "reply". Returns the list of chat messages."""
    messages = [{"role": "system", "content": RULES}]
    for said, answer in EXAMPLES:
        messages.append({"role": "user", "content": said})
        messages.append({"role": "assistant", "content": answer})
    for turn in history:
        messages.append({"role": "user", "content": turn["transcript"]})
        messages.append({"role": "assistant", "content": turn["reply"]})
    messages.append({"role": "user", "content": f"({how_they_seem(state, low_conf)})\n{state.transcript}"})
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
            "num_ctx": reply_cfg["context_tokens"],
            "temperature": reply_cfg["temperature"],
            "seed": reply_cfg["seed"],
            # the model stops writing as soon as it starts one of these: the stock endings of an assistant
            "stop": reply_cfg["stop"],
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
