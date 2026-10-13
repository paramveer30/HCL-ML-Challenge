"""Loads the two trained emotion models and gives each one's probabilities for a turn."""
from pathlib import Path

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForSequenceClassification, AutoTokenizer, WavLMModel

from meldrt.models.audio import AudioHead, clip_features
from meldrt.models.fusion import softmax
from meldrt.models.text import predict_logits


class EmotionClassifier:
    def __init__(self, cfg, device):
        self.device = device
        self.max_tokens = cfg["text"]["max_tokens"]
        self.sample_rate = cfg["audio"]["sample_rate"]
        self.min_samples = int(cfg["audio"]["min_seconds"] * self.sample_rate)

        best_dir = Path(cfg["text"]["checkpoint_dir"]) / "best"
        self.tokenizer = AutoTokenizer.from_pretrained(best_dir)
        self.text_model = AutoModelForSequenceClassification.from_pretrained(best_dir).to(device).eval()

        self.extractor = AutoFeatureExtractor.from_pretrained(cfg["audio"]["wavlm_model"])
        self.wavlm = WavLMModel.from_pretrained(cfg["audio"]["wavlm_model"]).to(device).eval()
        a = cfg["audio_head"]
        self.head = AudioHead(hidden=a["hidden"], dropout=a["dropout"]).to(device).eval()
        self.head.load_state_dict(torch.load(a["checkpoint"], map_location=device))

    def text_probs(self, context, target):
        """Probabilities over the 7 labels from the words, shape (7,)."""
        logits = predict_logits(self.text_model, self.tokenizer, [context], [target], self.max_tokens)
        return softmax(logits[0])

    def audio_probs(self, wav):
        """Probabilities from the tone of voice, shape (7,), or None if the audio is too short for WavLM."""
        if len(wav) < self.min_samples:
            return None
        # same path as training: float16 features, read back as float32
        feats = clip_features(wav, self.wavlm, self.extractor, self.sample_rate, self.device).astype(np.float32)
        with torch.no_grad():
            logits = self.head(torch.from_numpy(feats).unsqueeze(0).to(self.device))
        return softmax(logits[0].cpu().numpy())
