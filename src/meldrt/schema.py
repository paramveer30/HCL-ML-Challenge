"""The state the pipeline produces for one spoken turn."""
from dataclasses import dataclass, field


@dataclass
class View:
    """What one model thinks on its own."""

    label: str
    confidence: float
    probs: dict | None = None      # that model's probability for each of the 7 labels


@dataclass
class TurnState:
    turn_id: int
    transcript: str
    label: str                 # the fused emotion
    confidence: float          # the fused probability of that emotion
    probs: dict                # fused probability for each of the 7 labels
    text_view: View
    audio_view: View | None    # None when the turn was too short for the audio model
    tone_text_conflict: bool
    audio_seconds: float
    latency_ms: dict = field(default_factory=dict)
    vision_view: View | None = None    # None when no face was seen during the turn
    face_text_conflict: bool = False
    faces_seen: int = 0                # camera or video frames with a face during the turn

    def to_event(self):
        """The state as a JSON-ready dict."""
        views = {"text": {"label": self.text_view.label, "confidence": round(self.text_view.confidence, 3)}}
        if self.audio_view is not None:
            views["audio"] = {"label": self.audio_view.label, "confidence": round(self.audio_view.confidence, 3)}
        if self.vision_view is not None:
            views["vision"] = {"label": self.vision_view.label, "confidence": round(self.vision_view.confidence, 3)}
        return {
            "type": "state",
            "turn_id": self.turn_id,
            "transcript": self.transcript,
            "emotion": {
                "label": self.label,
                "confidence": round(self.confidence, 3),
                "probs": {name: round(p, 3) for name, p in self.probs.items()},
            },
            "views": views,
            "tone_text_conflict": self.tone_text_conflict,
            "face_text_conflict": self.face_text_conflict,
            "faces_seen": self.faces_seen,
            "audio_seconds": round(self.audio_seconds, 2),
            "latency_ms": {name: round(ms) for name, ms in self.latency_ms.items()},
        }
