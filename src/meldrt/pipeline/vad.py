"""Silero VAD: says speech or not for one 32 ms frame."""
import torch
from silero_vad import load_silero_vad


class Vad:
    def __init__(self, sample_rate, threshold):
        self.model = load_silero_vad()
        self.sample_rate = sample_rate
        self.threshold = threshold

    def is_speech(self, frame):
        """frame: float32 array of 512 samples."""
        with torch.no_grad():
            probability = self.model(torch.from_numpy(frame), self.sample_rate).item()
        return probability >= self.threshold

    def reset(self):
        # Silero keeps a memory of the frames it has seen; clear it between turns
        self.model.reset_states()
