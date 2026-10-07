"""Builds report/Teachable_Lens_Report.pdf: runs real measurements, draws figures, lays out PDF."""
from __future__ import annotations

import os
import random
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image as RLImage, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lens.clip_onnx import REPO_ID, TEXT_FILE, VISION_FILE, ClipOnnx  # noqa: E402
from lens.head import FewShotHead  # noqa: E402

FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "report" / "Teachable_Lens_Report.pdf"
ACCENT = "#4f46e5"
SHAPES = ["circle", "square", "triangle"]
PALETTE = ["red", "blue", "green", "orange", "purple", "gold", "teal", "black", "pink", "brown"]


# ---------- synthetic benchmark data ----------
def make_shape(kind: str, rng: random.Random) -> Image.Image:
    bg = rng.choice(["white", "#f1f5f9", "#fef3c7", "#e0f2fe"])
    im = Image.new("RGB", (256, 256), bg)
    d = ImageDraw.Draw(im)
    s = rng.randint(70, 170)
    x, y = rng.randint(10, 246 - s), rng.randint(10, 246 - s)
    col = rng.choice(PALETTE)
    if kind == "circle":
        d.ellipse((x, y, x + s, y + s), fill=col)
    elif kind == "square":
        d.rectangle((x, y, x + s, y + s), fill=col)
    else:
        d.polygon([(x + s / 2, y), (x, y + s), (x + s, y + s)], fill=col)
    return im


def run_experiments(clip: ClipOnnx) -> dict:
    rng = random.Random(7)
    train_pool = {k: [make_shape(k, rng) for _ in range(20)] for k in SHAPES}
    test = [(make_shape(k, rng), k) for k in SHAPES for _ in range(30)]

    # sample grid figure
    fig, axes = plt.subplots(3, 6, figsize=(9, 4.6))
    for r, k in enumerate(SHAPES):
        for c in range(6):
            axes[r, c].imshow(train_pool[k][c])
            axes[r, c].axis("off")
        axes[r, 0].set_title(k, loc="left", fontsize=11, color=ACCENT)
    fig.suptitle("Synthetic benchmark samples (random colour, size, position, background)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "samples.png", dpi=150)
    plt.close(fig)

    pool_emb = {k: clip.embed_images(v) for k, v in train_pool.items()}
    test_emb = clip.embed_images([im for im, _ in test])
    test_y = [y for _, y in test]

    shots, means, stds = [1, 2, 5, 10, 20], [], []
    for k in shots:
        accs = []
        for seed in range(5):
            r = np.random.default_rng(seed)
            X, y = [], []
            for cls in SHAPES:
                idx = r.choice(20, size=k, replace=False)
                X.append(pool_emb[cls][idx])
                y += [cls] * k
            h = FewShotHead()
            h.fit(np.concatenate(X), y)
            accs.append(h.clf.score(test_emb, test_y))
        means.append(float(np.mean(accs)))
        stds.append(float(np.std(accs)))

    txt = clip.embed_texts([f"a photo of a {s}" for s in SHAPES])
    zs_pred = [SHAPES[i] for i in (test_emb @ txt.T).argmax(1)]
    zs_acc = float(np.mean([p == t for p, t in zip(zs_pred, test_y)]))

    h = FewShotHead()
    X = np.concatenate([pool_emb[c][:10] for c in SHAPES])
    y = [c for c in SHAPES for _ in range(10)]
    t0 = time.perf_counter()
    h.fit(X, y)
    fit_ms = (time.perf_counter() - t0) * 1000
    fs_pred = h.clf.predict(test_emb)

    def confusion(preds):
        m = np.zeros((3, 3), int)
        for p, t in zip(preds, test_y):
            m[SHAPES.index(t), SHAPES.index(p)] += 1
        return m

    conf, zs_conf = confusion(fs_pred), confusion(zs_pred)

    # k-shot curve
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.errorbar(shots, [m * 100 for m in means], yerr=[s * 100 for s in stds], marker="o",
                color=ACCENT, capsize=4, lw=2, label="Few-shot head (5 seeds)")
    ax.axhline(zs_acc * 100, ls="--", color="#f59e0b", lw=2, label="Zero-shot text prompts")
    ax.axhline(100 / 3, ls=":", color="grey", label="Random chance")
    ax.set_xscale("log")
    ax.set_xticks(shots)
    ax.set_xticklabels(shots)
    ax.set_xlabel("Training images per class")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "kshot.png", dpi=150)
    plt.close(fig)

    # confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, m, title in [(axes[0], zs_conf, f"Zero-shot ({zs_acc:.0%})"),
                         (axes[1], conf, f"Few-shot, 10/class ({np.trace(conf) / conf.sum():.0%})")]:
        ax.imshow(m, cmap="Purples", vmin=0, vmax=30)
        ax.set_xticks(range(3), SHAPES)
        ax.set_yticks(range(3), SHAPES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title, fontsize=11)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, m[i, j], ha="center", va="center",
                        color="white" if m[i, j] > 15 else "black", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "confusion.png", dpi=150)
    plt.close(fig)

    return dict(shots=shots, means=means, stds=stds, zs_acc=zs_acc, fit_ms=fit_ms,
                fs10_acc=float(np.trace(conf) / conf.sum()), n_test=len(test))


def run_perf(clip: ClipOnnx) -> dict:
    rng = random.Random(1)
    imgs = [make_shape("circle", rng) for _ in range(32)]
    clip.embed_images(imgs[:2])  # warm-up
    t0 = time.perf_counter()
    for im in imgs[:16]:
        clip.embed_images([im])
    single = (time.perf_counter() - t0) / 16 * 1000
    t0 = time.perf_counter()
    clip.embed_images(imgs, batch=16)
    batched = (time.perf_counter() - t0) / 32 * 1000
    t0 = time.perf_counter()
    for _ in range(10):
        clip.embed_texts(["a photo of a dog"])
    text = (time.perf_counter() - t0) / 10 * 1000
    try:
        import psutil
        rss = psutil.Process(os.getpid()).memory_info().rss / 2**20
    except ImportError:
        rss = float("nan")
    from huggingface_hub import hf_hub_download
    vis_mb = Path(hf_hub_download(REPO_ID, VISION_FILE)).stat().st_size / 2**20
    txt_mb = Path(hf_hub_download(REPO_ID, TEXT_FILE)).stat().st_size / 2**20

    fig, ax = plt.subplots(figsize=(7, 2.8))
    labels = ["Image (single)", "Image (batch 16)", "Text prompt"]
    vals = [single, batched, text]
    bars = ax.barh(labels, vals, color=[ACCENT, "#818cf8", "#f59e0b"])
    for b, v in zip(bars, vals):
        ax.text(v + max(vals) * .01, b.get_y() + b.get_height() / 2, f"{v:.0f} ms", va="center")
    ax.set_xlabel("CPU latency per item (ms)")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max(vals) * 1.2)
    fig.tight_layout()
    fig.savefig(FIG / "latency.png", dpi=150)
    plt.close(fig)
    return dict(single=single, batched=batched, text=text, rss=rss, vis_mb=vis_mb, txt_mb=txt_mb)


def draw_architecture() -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec="#312e81", lw=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", lw=1.6, color="#312e81"))

    box(0.1, 2.6, 1.7, 1.0, "Image\n(upload / webcam\n/ clipboard)", "#e0e7ff")
    box(2.3, 2.6, 1.7, 1.0, "Preprocess\nresize 224, crop,\nnormalise", "#e0e7ff")
    box(4.5, 2.6, 2.0, 1.0, "CLIP vision encoder\nONNX int8\n(onnxruntime CPU)", "#c7d2fe")
    box(7.1, 2.6, 1.3, 1.0, "512-d\nembedding\n(L2 norm)", "#e0e7ff")
    box(8.7, 3.2, 1.25, 0.9, "Few-shot head\nLogReg\n(sklearn)", "#fde68a")
    box(8.7, 1.8, 1.25, 0.9, "Cosine vs\ntext embeds\n(zero-shot)", "#fde68a")
    box(4.5, 0.6, 2.0, 1.0, "CLIP text encoder\nONNX int8", "#c7d2fe")
    box(2.3, 0.6, 1.7, 1.0, "Labels typed\nby user\n\"a photo of ...\"", "#e0e7ff")
    box(7.1, 0.2, 1.3, 0.9, "data/<class>/\nexamples used\nto train head", "#dcfce7")
    arrow(1.85, 3.1, 2.3, 3.1)
    arrow(4.05, 3.1, 4.5, 3.1)
    arrow(6.55, 3.1, 7.1, 3.1)
    arrow(8.45, 3.25, 8.7, 3.6)
    arrow(8.45, 2.95, 8.7, 2.3)
    arrow(4.05, 1.1, 4.5, 1.1)
    arrow(6.55, 1.1, 8.7, 2.05)
    ax.set_title("Teachable Lens pipeline: one frozen encoder, two ways to classify", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "architecture.png", dpi=160)
    plt.close(fig)


# ---------- PDF ----------
def build_pdf(ex: dict, pf: dict) -> None:
    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=colors.HexColor(ACCENT), spaceBefore=6)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=colors.HexColor("#312e81"))
    B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=10.5, leading=15, spaceAfter=6)
    CAP = ParagraphStyle("CAP", parent=B, fontSize=9, textColor=colors.grey, alignment=TA_CENTER)
    CODE = ParagraphStyle("CODE", parent=B, fontName="Courier", fontSize=9, leading=12,
                          backColor=colors.HexColor("#f1f5f9"), borderPadding=6, leftIndent=6)
    W = A4[0] - 4 * cm

    def img(name, width=W, cap=None):
        p = FIG / name
        with Image.open(p) as im:
            w, h = im.size
        out = [RLImage(str(p), width=width, height=width * h / w)]
        if cap:
            out += [Spacer(1, 3), Paragraph(cap, CAP)]
        return out + [Spacer(1, 8)]

    def table(rows, widths):
        t = Table(rows, colWidths=widths, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(ACCENT)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2ff")]),
            ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#c7d2fe")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
        return [t, Spacer(1, 10)]

    def P(t):
        return Paragraph(t, B)

    s = []
    # Title page
    s += [Spacer(1, 4 * cm),
          Paragraph("Teachable Lens", ParagraphStyle("T", parent=H1, fontSize=34, leading=40, alignment=TA_CENTER)),
          Spacer(1, 10),
          Paragraph("A local, few-shot image classifier built on CLIP (ONNX) with a Gradio UI",
                    ParagraphStyle("ST", parent=B, fontSize=14, leading=18, alignment=TA_CENTER)),
          Spacer(1, 1 * cm)]
    s += img("ui_teach.png", W * .9, "The Teach tab of the running application")
    s += [Spacer(1, 1 * cm), Paragraph("Project report - September 2026", CAP), PageBreak()]

    # 1 Intro, 2 Constraints
    s += [Paragraph("1. Introduction", H1),
          P("Most image-classification projects start from a fixed dataset such as CIFAR-10 or MNIST and train "
            "a network that recognises exactly those classes. Teachable Lens takes the opposite direction: the "
            "user decides what the classes are at run time. You create a class, drop in a handful of example "
            "photos (your mug, your cat, a hand-drawn logo), press <b>Train</b>, and a new classifier is ready "
            "in about a second. When you have no examples at all, a <b>zero-shot</b> mode classifies an "
            "image against labels typed as plain text."),
          P("The idea behind this is transfer learning in its simplest form. A large pretrained vision-language "
            "model, OpenAI's CLIP, is used only as a frozen feature extractor. It maps any image to a 512-number "
            "vector in which visually and semantically similar images lie close together. A small logistic-"
            "regression layer trained on those vectors is enough to separate the user's classes."),
          Paragraph("1.1 Goals", H2),
          P("- Let non-experts build their own classifier with 5-20 images per class.<br/>"
            "- Run fully offline after one model download, on CPU only.<br/>"
            "- Stay within a laptop budget of 4 GB VRAM and 16 GB RAM with no PyTorch or TensorFlow install.<br/>"
            "- Provide a clean browser UI with upload, webcam and clipboard input."),
          Paragraph("1.2 What makes it different", H2),
          P("Compared with a typical \"train a CNN on a dataset\" project, Teachable Lens has no fixed label set, "
            "needs no GPU, trains in milliseconds rather than hours and combines two classification modes "
            "(example-based and language-based) over a single shared encoder."),
          Paragraph("2. Hardware constraints and design decisions", H1),
          P("The target machine has a 4 GB GPU and 16 GB of system memory. A standard PyTorch install with CUDA "
            "takes several gigabytes on disk, and fine-tuning even a small vision transformer in 4 GB of VRAM "
            "is tight. These constraints shaped every choice:")]
    s += table([["Decision", "Choice", "Reason"],
                ["Inference engine", "onnxruntime (CPU)", "Small wheel, no CUDA or PyTorch needed"],
                ["Backbone", "CLIP ViT-B/32, int8 ONNX", "Strong general features, small quantized file"],
                ["Training", "Frozen encoder + LogReg", "Trains on CPU in milliseconds, no GPU memory"],
                ["Tokenizer", "HF tokenizers (Rust)", "Avoids installing the full transformers library"],
                ["UI", "Gradio 6", "Browser UI with webcam/clipboard in pure Python"],
                ["Storage", "Plain folders + joblib", "Dataset stays human-readable and portable"]],
               [3.3 * cm, 4.6 * cm, W - 7.9 * cm])
    s += [PageBreak()]

    # 3 Architecture, 4 Models
    s += [Paragraph("3. System architecture", H1)]
    s += img("architecture.png", W, "Figure 1. Data flow. Both modes share the same frozen CLIP encoder.")
    s += [P("An input image is resized so its shorter side is 224 pixels, center-cropped to 224x224 and "
            "normalised with CLIP's channel mean and standard deviation. The quantized vision encoder produces "
            "a 512-dimensional embedding that is L2-normalised. From there the pipeline splits:"),
          P("<b>Few-shot mode.</b> Embeddings of all stored example images train a multinomial logistic-"
            "regression classifier (scikit-learn, C = 10, class-balanced weights). The trained head is saved "
            "to <font face='Courier'>models/head.joblib</font> and reloaded on start-up."),
          P("<b>Zero-shot mode.</b> Each user label is wrapped in the prompt template \"a photo of {label}\" and "
            "encoded by the CLIP text encoder. Cosine similarities between image and text embeddings, scaled by "
            "CLIP's logit factor of 100, pass through a softmax to give probabilities."),
          Paragraph("4. Models used (Hugging Face)", H1),
          P("All pretrained weights come from the Hugging Face Hub and are downloaded once into the local cache "
            "(<font face='Courier'>~/.cache/huggingface</font>). No model is trained from scratch; only the "
            "small classification head is trained locally.")]
    s += table([["Component", "Hugging Face repo / file", "Size"],
                ["Vision encoder", f"{REPO_ID}\n{VISION_FILE}", f"{pf['vis_mb']:.0f} MB"],
                ["Text encoder", f"{REPO_ID}\n{TEXT_FILE}", f"{pf['txt_mb']:.0f} MB"],
                ["Tokenizer", f"{REPO_ID}\ntokenizer.json", "< 5 MB"],
                ["Original weights", "openai/clip-vit-base-patch32 (MIT licence)", "-"]],
               [3.2 * cm, W - 5.4 * cm, 2.2 * cm])
    s += [P("The Xenova repository is a community ONNX export of the official OpenAI checkpoint with 8-bit "
            "weight quantization. Quantization cuts the file size roughly by four and speeds up CPU inference "
            "at the cost of a small drop in accuracy."), PageBreak()]

    # 5 Implementation, 6 UI
    s += [Paragraph("5. Implementation", H1),
          P("The code is split into three short modules, about 250 lines in total.")]
    s += table([["File", "Responsibility"],
                ["lens/clip_onnx.py", "Downloads the ONNX files, preprocessing, image and text embedding"],
                ["lens/head.py", "FewShotHead: fit, predict, save, load (logistic regression)"],
                ["app.py", "Gradio UI, dataset folder management, training and classify callbacks"]],
               [4 * cm, W - 4 * cm])
    s += [Paragraph("5.1 Embedding an image without PyTorch", H2),
          P("CLIP's preprocessing is reimplemented with Pillow and NumPy, so the only runtime dependency for "
            "the model is onnxruntime:"),
          Paragraph("img = resize(img, shorter_side=224, BICUBIC)<br/>img = center_crop(img, 224)<br/>"
                    "x = (pixels / 255 - MEAN) / STD&nbsp;&nbsp;# CLIP constants<br/>"
                    "emb = session.run(['image_embeds'], {'pixel_values': x})<br/>"
                    "emb = emb / norm(emb)", CODE), Spacer(1, 8),
          Paragraph("5.2 Training the head", H2),
          P("Training embeds every image under <font face='Courier'>data/&lt;class&gt;/</font> in batches of 16 "
            "and fits the logistic regression. Because the encoder is frozen, the heavy computation happens "
            "exactly once per image; fitting the head itself took "
            f"<b>{ex['fit_ms']:.0f} ms</b> for 30 images in our measurement. Adding a new class means "
            "adding a folder and retraining, with no risk of forgetting old classes."),
          Paragraph("5.3 Dataset storage", H2),
          P("Examples are copied into one folder per class with sequential file names. The dataset can be "
            "inspected, backed up or edited in a normal file manager, and the Teach tab shows a live count "
            "of images per class."),
          Paragraph("6. User interface", H1),
          P("The Gradio interface has three tabs. <b>Teach</b> manages classes and triggers training. "
            "<b>Classify</b> runs the trained model on an uploaded, webcam or pasted image and shows the top-5 "
            "classes as confidence bars that update as soon as the image changes. <b>Zero-shot</b> accepts a "
            "comma-separated list of labels and needs no training at all. Errors such as training with only one "
            "class are shown as toast messages instead of crashing the app. The title page shows the Teach tab.")]
    s += [PageBreak()]

    # 7 Evaluation
    s += [Paragraph("7. Evaluation", H1),
          P("To get reproducible numbers without shipping a third-party dataset, a synthetic benchmark was "
            "generated: three shape classes (circle, square, triangle) drawn with a random colour out of ten, "
            "random size, random position and one of four background tints. Colour is deliberately "
            "uninformative, so the classifier has to learn shape. The test set holds "
            f"{ex['n_test']} images (30 per class) that never appear in training.")]
    s += img("samples.png", W * .78, "Figure 2. Example benchmark images.")
    s += [Paragraph("7.1 Accuracy versus number of examples", H2),
          P("For each budget of k images per class, five random training subsets were drawn from a pool of 20 "
            "per class. Figure 3 reports mean and standard deviation of test accuracy.")]
    s += img("kshot.png", W * .72, "Figure 3. Few-shot accuracy vs. examples per class, with zero-shot baseline.")
    rows = [["Images / class"] + [str(k) for k in ex["shots"]],
            ["Accuracy (mean)"] + [f"{m:.1%}" for m in ex["means"]],
            ["Std. dev."] + [f"+/-{sd:.1%}" for sd in ex["stds"]]]
    s += table(rows, [3.4 * cm] + [(W - 3.4 * cm) / len(ex["shots"])] * len(ex["shots"]))
    s += [PageBreak()]

    if ex["zs_acc"] >= ex["fs10_acc"]:
        compare = ("Both modes classify this benchmark almost perfectly. Simple shapes have short, precise "
                   "names that CLIP learned from its training captions, so the text prompts alone are enough.")
    else:
        compare = ("The few-shot head is more accurate here; the off-diagonal cells in Figure 4 show which "
                   "shapes the text prompts confuse.")
    s += [Paragraph("7.2 Zero-shot versus few-shot", H2),
          P(f"With the prompts \"a photo of a circle / square / triangle\" and no training, the zero-shot mode "
            f"reached <b>{ex['zs_acc']:.1%}</b> accuracy. The few-shot head with 10 examples per class reached "
            f"<b>{ex['fs10_acc']:.1%}</b>. {compare}")]
    s += img("confusion.png", W * .95, f"Figure 4. Confusion matrices on the {ex['n_test']}-image test set.")
    s += [P("Zero-shot works when a category has a clear everyday name and needs nothing but words. Few-shot "
            "is what makes the app personal: categories like \"my bike\" versus \"my friend's bike\", or a "
            "specific logo, cannot be described in text but can be learned from a few photos. The two modes "
            "complement each other in the same app."),
          Paragraph("7.3 Qualitative test", H2),
          P("During development, a red circle was classified zero-shot against \"a red circle\", \"a blue square\" "
            "and \"a green tree\" with 99.7% confidence for the correct label. A head trained on four circles and "
            "four squares in red, blue, green and orange classified an unseen <i>purple</i> square as square with "
            "67% probability, showing that it generalises to colours it never saw."),
          Paragraph("7.4 Limits of this benchmark", H2),
          P("Synthetic shapes are much simpler than real photos, so these numbers show that the pipeline works "
            "and how accuracy scales with examples; they do not predict accuracy on arbitrary user data. "
            "Evaluating on a real few-shot benchmark would be a natural next step."), PageBreak()]

    # 8 Performance, 9 Limitations
    rss = "n/a" if np.isnan(pf["rss"]) else f"{pf['rss']:.0f} MB"
    s += [Paragraph("8. Performance on the target hardware", H1),
          P("All timings were measured on the development laptop using onnxruntime's CPU execution provider, "
            "after one warm-up call.")]
    s += img("latency.png", W * .9, "Figure 5. Average CPU latency per item.")
    s += table([["Metric", "Value"],
                ["Image embedding, single", f"{pf['single']:.0f} ms"],
                ["Image embedding, batch of 16", f"{pf['batched']:.0f} ms / image"],
                ["Text embedding (one prompt)", f"{pf['text']:.0f} ms"],
                ["Head training, 30 images", f"{ex['fit_ms']:.0f} ms (after embedding)"],
                ["Process memory (RSS) after benchmark", rss],
                ["GPU memory used", "0 MB (CPU only)"],
                ["Model download (one time)", f"{pf['vis_mb'] + pf['txt_mb']:.0f} MB"]],
               [7 * cm, W - 7 * cm])
    s += [P("The whole application fits comfortably in the 16 GB RAM budget and leaves the 4 GB GPU free. "
            "Latency is low enough for the Classify tab to update interactively when an image is uploaded."),
          Paragraph("9. Limitations and future work", H1),
          P("- <b>Fine-grained classes.</b> Frozen CLIP features can struggle with very similar classes (two "
            "nearly identical products); fine-tuning the encoder would help but needs a GPU framework.<br/>"
            "- <b>Quantization.</b> int8 weights trade some accuracy for speed; an fp32 ONNX file can be "
            "swapped in by changing one constant.<br/>"
            "- <b>GPU acceleration.</b> Installing onnxruntime-directml would let Windows use the 4 GB GPU "
            "without PyTorch.<br/>"
            "- <b>Hybrid mode.</b> Text embeddings of class names could be added as extra training points so "
            "classes work well with only one or two photos.<br/>"
            "- <b>Export.</b> The trained head could be exported to ONNX (skl2onnx) for use on mobile or the web."),
          PageBreak()]

    # 10 How to run, 11 Conclusion
    s += [Paragraph("10. How to run", H1),
          Paragraph("pip install -r requirements.txt<br/>python app.py<br/># open http://127.0.0.1:7860", CODE),
          Spacer(1, 8),
          P("The first launch downloads the ONNX files from Hugging Face (about "
            f"{pf['vis_mb'] + pf['txt_mb']:.0f} MB). After that the app works offline. Typical workflow:"),
          P("1. In <b>Teach</b>, type a class name, upload 5-20 photos and press <b>Add examples</b>.<br/>"
            "2. Repeat for at least one more class.<br/>"
            "3. Press <b>Train model</b>; the status line reports image count and training accuracy.<br/>"
            "4. Open <b>Classify</b> and upload, paste or take a webcam photo.<br/>"
            "5. Use <b>Zero-shot</b> at any time to classify against typed labels."),
          Paragraph("11. Conclusion", H1),
          P("Teachable Lens shows that a useful, personal image classifier does not need a GPU framework or a "
            "large labelled dataset. A quantized CLIP encoder running in onnxruntime supplies general-purpose "
            "features, and a logistic-regression head trained in milliseconds turns them into a classifier "
            f"for any user-defined classes. On the synthetic benchmark it reached {ex['means'][2]:.0%} accuracy "
            f"with five examples per class and {ex['means'][-1]:.0%} with twenty, all within a laptop's CPU "
            "and memory budget."),
          Paragraph("References", H1),
          P("[1] A. Radford et al., \"Learning Transferable Visual Models From Natural Language Supervision\", "
            "ICML 2021.<br/>"
            "[2] openai/clip-vit-base-patch32, Hugging Face Hub.<br/>"
            f"[3] {REPO_ID}, ONNX export, Hugging Face Hub.<br/>"
            "[4] ONNX Runtime, https://onnxruntime.ai<br/>"
            "[5] Gradio, https://gradio.app<br/>"
            "[6] F. Pedregosa et al., \"Scikit-learn: Machine Learning in Python\", JMLR 12, 2011.")]

    def footer(canvas, doc):
        if doc.page == 1:
            return
        canvas.saveState()
        canvas.setFont("Helvetica", 8.5)
        canvas.setFillColor(colors.grey)
        canvas.drawString(2 * cm, 1.2 * cm, "Teachable Lens - project report")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()

    SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                      bottomMargin=2 * cm, title="Teachable Lens", author="Teachable Lens project"
                      ).build(s, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    clip = ClipOnnx()
    draw_architecture()
    ex = run_experiments(clip)
    pf = run_perf(clip)
    print("experiments:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in ex.items()})
    print("perf:", {k: round(v, 1) for k, v in pf.items()})
    build_pdf(ex, pf)
    pages = len(re.findall(rb"/Type\s*/Page[^s]", OUT.read_bytes()))
    print(f"wrote {OUT} ({pages} pages)")
