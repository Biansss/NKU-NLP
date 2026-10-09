"""Generate figures for the NLP text-classification report.

Run with the ``nlp-hw1`` conda environment. For every figure both a vector PDF
(embedded into the LaTeX report) and a 300-dpi PNG (for quick preview) are
written into ``report/images``.
"""

import os
import sys

HW1 = r"D:\aaadep\grade_4\自然语言\NKU-NLP\hw1"
sys.path.insert(0, HW1)

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_utils import load_nyt, split_nyt, tokenize

OUT = os.path.join(HW1, "report", "images")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

C_ACC = "#3b6fb6"
C_F1 = "#e08a3c"
C_BUS, C_POL, C_SPO = "#4C72B0", "#DD8452", "#55A868"


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"))
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- data
texts, labels = load_nyt()
splits = split_nyt(texts, labels)
classes = ["business", "politics", " sports".strip()]
split_names = ["train", "val", "test"]
counts = {
    s: [sum(1 for y in splits[s][1] if y == c) for c in classes]
    for s in split_names
}
print("class counts per split:", counts)
print("split sizes:", {s: len(splits[s][1]) for s in split_names})

# ---------------------------------- Figure 1: class distribution / split
fig, ax = plt.subplots(figsize=(6.4, 3.9))
x = np.arange(len(split_names))
w = 0.26
for i, (c, color) in enumerate(zip(classes, [C_BUS, C_POL, C_SPO])):
    vals = [counts[s][i] for s in split_names]
    bars = ax.bar(x + (i - 1) * w, vals, w,
                  label=c.capitalize(), color=color, edgecolor="white")
    ax.bar_label(bars, fontsize=8, padding=1)
ax.set_xticks(x)
ax.set_xticklabels(["Train (80%)", "Validation (10%)", "Test (10%)"])
ax.set_ylabel("Number of documents")
ax.set_title("Class distribution across the stratified splits")
ax.legend(frameon=False, ncol=3, loc="upper center",
          bbox_to_anchor=(0.5, 1.0))
ax.set_ylim(0, max(max(counts[s]) for s in split_names) * 1.20)
save(fig, "fig_class_dist")

# ---------------------------------- Figure 2: six experiments comparison
methods = ["Binary\nBoW", "Word\nFrequency", "GloVe",
           "W2V\n(AG News)", "W2V\n(NYT)", "BERT"]
acc = [0.9826, 0.9826, 0.9774, 0.9748, 0.9740, 0.9774]
f1 = [0.9576, 0.9621, 0.9468, 0.9406, 0.9391, 0.9517]

fig, ax = plt.subplots(figsize=(7.8, 4.2))
# family background bands (drawn first so the bars sit on top)
ax.axvspan(-0.5, 1.5, color="#3b6fb6", alpha=0.07)
ax.axvspan(1.5, 4.5, color="#55a868", alpha=0.07)
ax.axvspan(4.5, 5.5, color="#c44e52", alpha=0.07)
ax.text(0.5, 1.0115, "Task 1: Bag-of-Words",
        ha="center", va="center", fontsize=9, color="#3b6fb6")
ax.text(3.0, 1.0115, "Task 2: averaged word embeddings",
        ha="center", va="center", fontsize=9, color="#3d8b49")
ax.text(5.0, 1.0115, "Task 3",
        ha="center", va="center", fontsize=9, color="#c44e52")

x = np.arange(len(methods))
b1 = ax.bar(x - w / 2, acc, w, label="Accuracy",
            color=C_ACC, edgecolor="white")
b2 = ax.bar(x + w / 2, f1, w, label="Macro-F1",
            color=C_F1, edgecolor="white")
ax.bar_label(b1, labels=[f"{v:.4f}" for v in acc], fontsize=7,
             padding=2, rotation=90)
ax.bar_label(b2, labels=[f"{v:.4f}" for v in f1], fontsize=7,
             padding=2, rotation=90)
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylim(0.90, 1.02)
ax.set_ylabel("Score")
ax.set_title("Performance on the NYT Test Set (n = 1152)")
ax.legend(frameon=False, loc="lower right", ncol=2)
save(fig, "fig_results_bar")

# ---------------------------------- Figure 3: BERT per-epoch validation
ep = [1, 2, 3]
val_acc = [0.9783, 0.9809, 0.9800]
val_f1 = [0.9527, 0.9599, 0.9545]
test_acc, test_f1 = 0.9774, 0.9517

fig, ax = plt.subplots(figsize=(6.6, 4.0))
ax.plot(ep, val_acc, "-o", color=C_ACC, lw=2, ms=7, label="Validation Accuracy")
ax.plot(ep, val_f1, "-s", color=C_F1, lw=2, ms=7, label="Validation Macro-F1")
ax.axhline(test_acc, color=C_ACC, ls="--", lw=1.2, alpha=0.7)
ax.axhline(test_f1, color=C_F1, ls="--", lw=1.2, alpha=0.7)
ax.text(3.06, test_acc, "Test Acc 0.9774", va="center", ha="left",
        fontsize=8.5, color=C_ACC)
ax.text(3.06, test_f1, "Test F1 0.9517", va="center", ha="left",
        fontsize=8.5, color=C_F1)
for xi, yi in zip(ep, val_acc):
    ax.annotate(f"{yi:.4f}", (xi, yi), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=8.5, color=C_ACC)
for xi, yi in zip(ep, val_f1):
    ax.annotate(f"{yi:.4f}", (xi, yi), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=8.5, color=C_F1)
ax.set_xticks(ep)
ax.set_xlim(0.8, 3.55)
ax.set_ylim(0.947, 0.990)
ax.set_xlabel("Epoch")
ax.set_ylabel("Score")
ax.set_title("BERT validation metrics over 3 epochs")
ax.legend(frameon=False, loc="upper center", ncol=2,
          bbox_to_anchor=(0.5, -0.12))
save(fig, "fig_bert_epochs")

# ---------------------------------- Figure 4: document length vs 64
print("tokenizing all NYT documents with NLTK for length stats ...")
lens = np.array([len(tokenize(t)) for t in texts])
mean_len = lens.mean()
median_len = int(np.median(lens))
p25 = int(np.percentile(lens, 25))
p75 = int(np.percentile(lens, 75))
over64 = (lens > 64).mean() * 100.0
print(f"length mean={mean_len:.1f} median={median_len} "
      f"p25={p25} p75={p75} min={lens.min()} max={lens.max()} "
      f"frac>64={over64:.2f}%")

fig, ax = plt.subplots(figsize=(7.0, 4.0))
ax.hist(np.clip(lens, 0, 1500), bins=60, color="#4C72B0",
        edgecolor="white", alpha=0.9)
ymax = ax.get_ylim()[1]
ax.axvline(64, color="#c44e52", lw=2)
ax.text(72, ymax * 0.92, "max_length = 64", color="#c44e52", fontsize=10.5)
ax.text(0.985, 0.60,
        f"mean = {mean_len:.0f} tokens\n"
        f"median = {median_len} tokens\n"
        f"{over64:.1f}% of documents exceed 64 tokens",
        transform=ax.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#999999"))
ax.set_xlabel("Document length in NLTK tokens (long tail clipped at 1500)")
ax.set_ylabel("Number of documents")
ax.set_title("NYT document-length distribution versus BERT's truncation length")
save(fig, "fig_doc_length")

print("ALL FIGURES DONE")
