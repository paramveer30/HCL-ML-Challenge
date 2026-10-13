"""Whisper small through faster-whisper: one finished turn in, text out."""
from faster_whisper import WhisperModel


class Asr:
    def __init__(self, model_name, device, compute_type, beam_size):
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self.beam_size = beam_size

    def transcribe(self, wav):
        """wav: float32 array at 16 kHz. Returns the transcript as one string."""
        segments, _ = self.model.transcribe(wav, language="en", beam_size=self.beam_size)
        return " ".join(segment.text.strip() for segment in segments)
