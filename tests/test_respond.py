import json

from meldrt.pipeline.respond import stream_reply

REPLY_CFG = {
    "url": "http://127.0.0.1:11434", "model": "qwen2.5:1.5b", "max_tokens": 64,
    "temperature": 0.6, "seed": 0, "keep_alive": "30m",
}
MESSAGES = [{"role": "system", "content": "rules"}, {"role": "user", "content": "hello"}]


class FakeResponse:
    """Stands in for the HTTP response: one JSON line per piece, like Ollama sends."""

    def __init__(self, pieces):
        lines = [json.dumps({"message": {"role": "assistant", "content": p}, "done": False}) for p in pieces]
        lines.append(json.dumps({"message": {"role": "assistant", "content": ""}, "done": True}))
        self.lines = [line.encode("utf-8") for line in lines]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(self.lines)


def test_stream_reply_yields_pieces_and_skips_the_empty_last_line(monkeypatch):
    sent = {}

    def fake_urlopen(request):
        sent["url"] = request.full_url
        sent["body"] = json.loads(request.data)
        return FakeResponse(["I'm ", "glad", "!"])

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert list(stream_reply(MESSAGES, REPLY_CFG)) == ["I'm ", "glad", "!"]
    assert sent["url"] == "http://127.0.0.1:11434/api/chat"
    assert sent["body"]["model"] == "qwen2.5:1.5b"
    assert sent["body"]["messages"] == MESSAGES
    assert sent["body"]["stream"] is True
    assert sent["body"]["options"] == {"num_predict": 64, "temperature": 0.6, "seed": 0}
