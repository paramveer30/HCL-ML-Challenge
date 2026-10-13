"""Loads the trained emotion models and gives each one's probabilities for a turn."""
from pathlib import Path

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForSequenceClassification, AutoTokenizer, ViTModel, WavLMModel

from meldrt.data.video import FaceFinder, face_crops
from meldrt.models.audio import AudioHead, clip_features
from meldrt.models.fusion import softmax
from meldrt.models.text import predict_logits
from meldrt.models.vision import frame_layers, mean_over_faces


def keep_end(tokenizer, target, room):
    """Returns target unchanged if it fits in room tokens, otherwise only its last room tokens as text.

    Dropping context only works while the target itself fits. A very long turn, or a recognizer
    stuck repeating itself, would otherwise make the tokenizer fail.
    """
    ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    if len(ids) <= room:
        return target
    return tokenizer.decode(ids[-room:]).strip()


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

        # the face models are only loaded when the pipeline is asked to use a camera or video
        self.use_vision = cfg["pipeline"]["use_vision"]
        if self.use_vision:
            v, h = cfg["vision"], cfg["vision_head"]
            self.finder = FaceFinder(v["detector_file"], v["min_face_score"], v["min_face_px"], v["detect_width"])
            self.face_margin, self.image_size = v["face_margin"], v["image_size"]
            self.vit = ViTModel.from_pretrained(v["model"], add_pooling_layer=False).to(device).eval()
            self.vision_head = AudioHead(hidden=h["hidden"], dropout=h["dropout"]).to(device).eval()
            self.vision_head.load_state_dict(torch.load(h["checkpoint"], map_location=device))

    def text_probs(self, context, target):
        """Probabilities over the 7 labels from the words, shape (7,)."""
        # room for the target once the 4 special tokens and a small margin are taken off
        shortened = keep_end(self.tokenizer, target, self.max_tokens - 8)
        if shortened != target:
            context, target = "", shortened
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

    def face_layers(self, frame):
        """frame: one BGR image from a camera or a video. Returns the ViT features of the largest face
        in it, shape (13, 768) float32, or None if there is no face."""
        crops = face_crops([frame], self.finder, self.face_margin, self.image_size)
        if len(crops) == 0:
            return None
        return frame_layers(crops, self.vit, self.device)[0]

    def vision_probs(self, layers):
        """layers: the face_layers results of one turn, a list of (13, 768). Probabilities from the face, shape (7,)."""
        # same path as training: average over the faces, float16 features, read back as float32
        feats = mean_over_faces(np.stack(layers)).astype(np.float32)
        with torch.no_grad():
            logits = self.vision_head(torch.from_numpy(feats).unsqueeze(0).to(self.device))
        return softmax(logits[0].cpu().numpy())
