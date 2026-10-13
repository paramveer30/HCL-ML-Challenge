"""Whisper small through faster-whisper: one finished turn in, text out."""
from faster_whisper import WhisperModel


class Asr:
    def __init__(self, model_name, device, compute_type, beam_size, max_no_speech_prob):
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self.beam_size = beam_size
        self.max_no_speech_prob = max_no_speech_prob

    def transcribe(self, wav):
        """wav: float32 array at 16 kHz. Returns the transcript as one string, "" if no speech was heard."""
        segments, _ = self.model.transcribe(wav, language="en", beam_size=self.beam_size)
        segments = list(segments)
        # on silence or noise Whisper still writes a word (usually "you"), but it also says
        # the audio is probably not speech; we believe that and drop the turn
        if all(segment.no_speech_prob > self.max_no_speech_prob for segment in segments):
            return ""
        return " ".join(segment.text.strip() for segment in segments)
