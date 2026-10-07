"""Teachable Lens: teach your own image classes with a few photos, or classify zero-shot.

Runs fully local on CPU: CLIP (ONNX, from Hugging Face) + a tiny scikit-learn head.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import gradio as gr
from PIL import Image

from lens.clip_onnx import ClipOnnx
from lens.head import FewShotHead

DATA_DIR = Path(__file__).resolve().parent / "data"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

_clip: ClipOnnx | None = None
head = FewShotHead()
head.load()


def clip() -> ClipOnnx:
    global _clip
    if _clip is None:
        _clip = ClipOnnx()  # first call downloads ~150 MB of ONNX weights
    return _clip


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "_", name.strip().lower()).strip("_")


def dataset_summary() -> list[list]:
    if not DATA_DIR.exists():
        return []
    rows = []
    for d in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        n = sum(1 for f in d.iterdir() if f.suffix.lower() in IMG_EXT)
        rows.append([d.name, n])
    return rows


def add_examples(class_name: str, files: list[str] | None):
    slug = _slug(class_name or "")
    if not slug:
        raise gr.Error("Give the class a name first.")
    if not files:
        raise gr.Error("Upload at least one image.")
    target = DATA_DIR / slug
    target.mkdir(parents=True, exist_ok=True)
    start = len(list(target.iterdir()))
    added = 0
    for i, f in enumerate(files):
        src = Path(f)
        if src.suffix.lower() in IMG_EXT:
            shutil.copy(src, target / f"{start + i:04d}{src.suffix.lower()}")
            added += 1
    return dataset_summary(), None, f"Added {added} image(s) to '{slug}'."


def delete_class(class_name: str):
    target = DATA_DIR / _slug(class_name or "")
    if not class_name or not target.is_dir():
        raise gr.Error("No class with that name.")
    shutil.rmtree(target)
    return dataset_summary(), f"Deleted '{target.name}'. Retrain to update the model."


def train():
    images, labels = [], []
    for name, _ in dataset_summary():
        for f in sorted((DATA_DIR / name).iterdir()):
            if f.suffix.lower() in IMG_EXT:
                images.append(Image.open(f))
                labels.append(name)
    if len(set(labels)) < 2:
        raise gr.Error("You need at least 2 classes to train.")
    emb = clip().embed_images(images)
    head.fit(emb, labels)
    head.save()
    acc = head.clf.score(emb, labels)
    return f"Trained on {len(labels)} images, {len(set(labels))} classes (train accuracy {acc:.0%})."


def classify_custom(img: Image.Image | None):
    if img is None:
        return None
    if not head.ready:
        raise gr.Error("Train a model in the Teach tab first.")
    return head.predict(clip().embed_images([img])[0])


def classify_zero_shot(img: Image.Image | None, labels_text: str):
    if img is None:
        return None
    labels = [l.strip() for l in (labels_text or "").split(",") if l.strip()]
    if len(labels) < 2:
        raise gr.Error("Type at least 2 comma-separated labels.")
    img_emb = clip().embed_images([img])[0]
    txt_emb = clip().embed_texts([f"a photo of {l}" for l in labels])
    logits = 100.0 * txt_emb @ img_emb  # CLIP's learned logit scale
    probs = np_softmax(logits)
    return {l: float(p) for l, p in zip(labels, probs)}


def np_softmax(x):
    import numpy as np

    e = np.exp(x - x.max())
    return e / e.sum()


with gr.Blocks(title="Teachable Lens") as demo:
    gr.Markdown(
        "# Teachable Lens\n"
        "Teach a classifier your own categories with a few photos each, or describe labels in words. "
        "Everything runs locally on CPU."
    )
    with gr.Tab("1 · Teach"):
        with gr.Row():
            with gr.Column():
                cls_name = gr.Textbox(label="Class name", placeholder="e.g. my_mug")
                uploads = gr.File(label="Example images (5-20 works well)", file_count="multiple",
                                  file_types=["image"], type="filepath")
                with gr.Row():
                    add_btn = gr.Button("Add examples", variant="primary")
                    del_btn = gr.Button("Delete class", variant="stop")
            with gr.Column():
                table = gr.Dataframe(headers=["class", "images"], value=dataset_summary(),
                                     interactive=False, label="Your dataset")
                train_btn = gr.Button("Train model", variant="primary")
                status = gr.Markdown()
        add_btn.click(add_examples, [cls_name, uploads], [table, uploads, status])
        del_btn.click(delete_class, [cls_name], [table, status])
        train_btn.click(train, None, status)

    with gr.Tab("2 · Classify"):
        with gr.Row():
            img_in = gr.Image(type="pil", label="Image", sources=["upload", "webcam", "clipboard"])
            out = gr.Label(num_top_classes=5, label="Your model says")
        img_in.change(classify_custom, img_in, out)

    with gr.Tab("Zero-shot"):
        gr.Markdown("No training: type labels and CLIP matches the image to the text.")
        with gr.Row():
            with gr.Column():
                zs_img = gr.Image(type="pil", label="Image", sources=["upload", "webcam", "clipboard"])
                zs_labels = gr.Textbox(label="Labels (comma-separated)", value="a cat, a dog, a car, food")
                zs_btn = gr.Button("Classify", variant="primary")
            zs_out = gr.Label(num_top_classes=5, label="Match")
        zs_btn.click(classify_zero_shot, [zs_img, zs_labels], zs_out)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
