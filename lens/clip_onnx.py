"""CLIP ViT-B/32 running through onnxruntime (no PyTorch).

Model: Xenova/clip-vit-base-patch32 on Hugging Face (quantized ONNX export of
openai/clip-vit-base-patch32). Files are downloaded once into the HF cache.
"""
from __future__ import annotations

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image
from tokenizers import Tokenizer

REPO_ID = "Xenova/clip-vit-base-patch32"
VISION_FILE = "onnx/vision_model_quantized.onnx"
TEXT_FILE = "onnx/text_model_quantized.onnx"

# CLIP preprocessing constants
SIZE = 224
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def _session(path: str) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


def _embeds_output(sess: ort.InferenceSession) -> str:
    names = [o.name for o in sess.get_outputs()]
    return next((n for n in names if n.endswith("embeds")), names[0])


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


class ClipOnnx:
    def __init__(self) -> None:
        self.vision = _session(hf_hub_download(REPO_ID, VISION_FILE))
        self.text = _session(hf_hub_download(REPO_ID, TEXT_FILE))
        self.tokenizer = Tokenizer.from_file(hf_hub_download(REPO_ID, "tokenizer.json"))
        self._vision_in = self.vision.get_inputs()[0].name
        self._vision_out = _embeds_output(self.vision)
        self._text_in = self.text.get_inputs()[0].name
        self._text_out = _embeds_output(self.text)

    @staticmethod
    def preprocess(img: Image.Image) -> np.ndarray:
        img = img.convert("RGB")
        w, h = img.size
        scale = SIZE / min(w, h)
        img = img.resize((max(SIZE, round(w * scale)), max(SIZE, round(h * scale))), Image.BICUBIC)
        w, h = img.size
        left, top = (w - SIZE) // 2, (h - SIZE) // 2
        img = img.crop((left, top, left + SIZE, top + SIZE))
        arr = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
        return arr.transpose(2, 0, 1)  # HWC -> CHW

    def embed_images(self, images: list[Image.Image], batch: int = 16) -> np.ndarray:
        out = []
        for i in range(0, len(images), batch):
            pixels = np.stack([self.preprocess(im) for im in images[i : i + batch]])
            out.append(self.vision.run([self._vision_out], {self._vision_in: pixels})[0])
        return _normalize(np.concatenate(out).astype(np.float32))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        # One prompt at a time avoids padding/attention-mask differences between exports.
        out = []
        for t in texts:
            ids = self.tokenizer.encode(t).ids[:77]
            ids = np.array([ids], dtype=np.int64)
            out.append(self.text.run([self._text_out], {self._text_in: ids})[0])
        return _normalize(np.concatenate(out).astype(np.float32))
