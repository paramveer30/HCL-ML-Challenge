"""Loads the trained emotion models and gives each one's probabilities for a turn."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime
import torch
from transformers import AutoFeatureExtractor, AutoModelForSequenceClassification, AutoTokenizer, ViTModel, WavLMModel

from meldrt.data.video import FaceFinder, crop_face
from meldrt.models.audio import clip_features
from meldrt.models.fusion import softmax
from meldrt.models.head import LayerMixHead
from meldrt.models.text import predict_logits
from meldrt.models.vision import expression_probs, face_layers, mean_over_faces


def keep_end(tokenizer, target, room):
    """Returns target unchanged if it fits in room tokens, otherwise only its last room tokens as text.

    Dropping context only works while the target itself fits. A very long turn, or a recognizer
    stuck repeating itself, would otherwise make the tokenizer fail.
    """
    ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    if len(ids) <= room:
        return target
    return tokenizer.decode(ids[-room:]).strip()


@dataclass
class Face:
    """What the face models saw in one picture."""

    box: tuple                 # (x, y, w, h) of the face, in pixels of that picture
    points: list               # the detector's 5 landmarks as (x, y): eyes, nose tip, mouth corners
    layers: np.ndarray | None  # ViT features of the face, shape (13, 768), for the head trained on MELD
    expression: np.ndarray | None   # what the face shows: the expression classifier's probabilities over the 7 labels


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
        self.head = LayerMixHead(hidden=a["hidden"], dropout=a["dropout"]).to(device).eval()
        self.head.load_state_dict(torch.load(a["checkpoint"], map_location=device))

        # the face models are only loaded when the pipeline is asked to use a camera or video
        self.use_vision = cfg["pipeline"]["use_vision"]
        if self.use_vision:
            v, h = cfg["vision"], cfg["vision_head"]
            self.finder = FaceFinder(v["detector_file"], v["min_face_score"], v["min_face_px"], v["detect_width"])
            self.face_margin, self.image_size = v["face_margin"], v["image_size"]
            self.live_head = v["live_head"]
            self.expression_margin = v["expression_margin"]
            # the checkpoint also holds a classifier trained on FER2013; only the encoder under it is used
            self.vit = ViTModel.from_pretrained(v["model"], add_pooling_layer=False).to(device).eval()
            # small enough for the CPU (about 12 ms a face), so it takes no GPU memory from the other models
            self.expression = onnxruntime.InferenceSession(v["expression_file"], providers=["CPUExecutionProvider"])
            self.vision_head = LayerMixHead(hidden=h["hidden"], dropout=h["dropout"]).to(device).eval()
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

    def look(self, frame, from_camera):
        """frame: one BGR image. from_camera: True for a live camera, False for a picture of a clip's video.
        Returns a Face for the largest face in it, or None.

        Each face model is used where it was trained. A camera shows one person facing it, like the photos
        the expression classifier learned from. A MELD clip is a TV shot, which is what our own head saw.
        """
        found = self.finder.largest_face(frame)
        if found is None:
            return None
        box, points = found
        layers = expression = None
        if from_camera:
            crop = crop_face(frame, box, self.expression_margin, self.image_size)
            expression = expression_probs(crop[None], self.expression)[0]
        if not from_camera or self.live_head == "meld":
            crop = crop_face(frame, box, self.face_margin, self.image_size)
            layers = face_layers(crop[None], self.vit, self.device)[0]
        return Face(box=box, points=points, layers=layers, expression=expression)

    def vision_probs(self, faces, from_camera):
        """faces: the Face objects seen during one turn. Probabilities from the face, shape (7,)."""
        if from_camera and self.live_head == "expression":
            return np.mean([face.expression for face in faces], axis=0)
        # the same path as training: average over the faces, float16 features, read back as float32
        feats = mean_over_faces(np.stack([face.layers for face in faces])).astype(np.float32)
        with torch.no_grad():
            logits = self.vision_head(torch.from_numpy(feats).unsqueeze(0).to(self.device))
        return softmax(logits[0].cpu().numpy())
