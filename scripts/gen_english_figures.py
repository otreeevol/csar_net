# -*- coding: utf-8 -*-
"""Regenerate all paper figures with ENGLISH labels from cached results (no retraining).

Inputs (results_mt/metrics/):
  confusion_{ce,cbam_ce,cbam_ldam}.json   -> confusion_mt.png
  history_{ce,cbam_ce,ldam,cbam_ldam,cs_cbam}.json -> training_mt.png
  feat_{ce,cbam_ce,cbam_ldam}.npy + labels_test.npy -> tsne_mt.png
  multiseed.json                          -> fig_recall_bar.png
  ablation_extra.json + diag.json + multiseed.json -> fig_ablation_bars.png

Outputs: submit/figures/*.png (overwrites the Chinese-labelled versions).
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRIC = os.path.join(ROOT, "results_mt", "metrics")
OUT = os.path.join(ROOT, "submit", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

CLASSES = ["Blowhole", "Break", "Crack", "Fray", "Free", "Uneven"]
PANEL3 = ("ce", "cbam_ce", "cbam_ldam")  # left / middle / right
TITLE3 = {"ce": "CE (baseline)", "cbam_ce": "CBAM+CE", "cbam_ldam": "CBAM+LDAM (CSAR-Net)"}


def load(name):
    with open(os.path.join(METRIC, name), encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# 1. Confusion matrices (3 panels)
# --------------------------------------------------------------------------- #
def fig_confusion():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, k in zip(axes, PANEL3):
        cm = np.array(load(f"confusion_{k}.json")["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues")
        for i in range(6):
            for j in range(6):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
        ax.set_xticks(range(6)); ax.set_yticks(range(6))
        ax.set_xticklabels(CLASSES, rotation=45, fontsize=8)
        ax.set_yticklabels(CLASSES, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(TITLE3[k])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "confusion_mt.png"), dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 2. Training curves (G-mean + accuracy)
# --------------------------------------------------------------------------- #
def fig_training():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    labels = {"ce": "CE (baseline)", "cbam_ce": "CBAM+CE", "ldam": "LDAM",
              "cbam_ldam": "CBAM+LDAM (CSAR-Net)", "cs_cbam": "CS-CBAM"}
    for name, lab in labels.items():
        h = load(f"history_{name}.json")
        axes[0].plot(h["test_gmean"], label=lab)
        axes[1].plot(h["test_acc"], label=lab)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Test G-mean")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Test accuracy")
    axes[0].set_title("Test-set G-mean (CS-CBAM collapses)")
    axes[1].set_title("Test-set accuracy")
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "training_mt.png"), dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 3. t-SNE of penultimate features (3 panels)
# --------------------------------------------------------------------------- #
def fig_tsne():
    from sklearn.manifold import TSNE
    labels = np.load(os.path.join(METRIC, "labels_test.npy"))
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.2))
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    for ax, k in zip(axes, PANEL3):
        feats = np.load(os.path.join(METRIC, f"feat_{k}.npy"))
        X2 = tsne.fit_transform(feats)
        for c in range(6):
            ax.scatter(X2[labels == c, 0], X2[labels == c, 1], s=12, alpha=0.6, label=CLASSES[c])
        ax.set_title(TITLE3[k]); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=8, markerscale=1.5, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "tsne_mt.png"), dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 4. Per-class recall bars
# --------------------------------------------------------------------------- #
def fig_recall_bar():
    s = load("multiseed.json")["summary"]
    ce = s["ce"]["recall_mean"]
    cb = s["cbam_ce"]["recall_mean"]
    cs = s["cbam_ldam"]["recall_mean"]
    x = np.arange(len(CLASSES))
    w = 0.27
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    ax.bar(x - w, ce, w, label="CE (baseline)", color="#9db8d2")
    ax.bar(x, cb, w, label="CBAM+CE", color="#4a86c8")
    ax.bar(x + w, cs, w, label="CBAM+LDAM (CSAR-Net)", color="#d9a441")
    ax.set_xticks(x); ax.set_xticklabels(CLASSES, fontsize=9)
    ax.set_ylabel("Recall", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.axhline(0, color="k", lw=0.6)
    ax.legend(fontsize=8, ncol=3, loc="upper right", frameon=False)
    for xi in x:
        ax.axvline(xi + 0.5 * w, color="gray", lw=0.3, alpha=0.4)
    ax.set_title("Per-class recall: CBAM protects rare classes; LDAM regresses", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_recall_bar.png"), dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 5. Ablation summary bars
# --------------------------------------------------------------------------- #
def _bar(ax, labels, means, stds, color="#4a86c8", title="", ylim=None, baseline=None):
    x = np.arange(len(labels))
    ax.bar(x, means, 0.55, yerr=stds, capsize=3, color=color,
           error_kw=dict(lw=1, ecolor="#333"))
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Macro G-mean", fontsize=8)
    if baseline is not None:
        ax.axhline(baseline, color="r", lw=1, ls="--", alpha=0.7)
        ax.text(len(labels) - 0.5, baseline + 0.004, "CE (baseline)", color="r",
                fontsize=7, ha="right")
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=9)


def fig_ablation_bars():
    abl = load("ablation_extra.json")["summary"]
    diag = load("diag.json")["summary"]
    ce_g = load("multiseed.json")["summary"]["ce"]["macro_gmean_mean"]

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))

    pos = ["cbam_s2", "cbam_s3", "cbam_s4", "cbam_s34"]
    _bar(axes[0], ["Stage 2", "Stage 3", "Stage 4", "Stage 3+4"],
         [abl[k]["macro_gmean_mean"] for k in pos],
         [abl[k]["macro_gmean_std"] for k in pos],
         title="(a) CBAM injection position", ylim=(0.55, 0.85), baseline=ce_g)

    sub = ["cbam_ch", "cbam_sp", "cbam_both"]
    _bar(axes[1], ["Channel only", "Spatial only", "Channel+Spatial"],
         [abl[k]["macro_gmean_mean"] for k in sub],
         [abl[k]["macro_gmean_std"] for k in sub],
         title="(b) CBAM sub-module decomposition", ylim=(0.55, 0.85), baseline=ce_g)

    Cs = ["0.1", "0.3", "0.5", "1.0"]
    marg = [abl[f"ldam_nodrw_C{c}"]["macro_gmean_mean"] for c in ("0.1", "0.3", "1")]
    marg.insert(2, diag["ldam_nodrw"]["macro_gmean_mean"])  # C=0.5 from diag
    full = [abl[f"cbam_ldam_C{c}"]["macro_gmean_mean"] for c in ("0.1", "0.3", "1")]
    full.insert(2, load("multiseed.json")["summary"]["cbam_ldam"]["macro_gmean_mean"])
    x = np.arange(len(Cs))
    w = 0.36
    axes[2].bar(x - w / 2, marg, w, label="Margin-only LDAM", color="#b0bec5")
    axes[2].bar(x + w / 2, full, w, label="CBAM+LDAM", color="#d9a441")
    axes[2].set_xticks(x); axes[2].set_xticklabels([f"C={c}" for c in Cs], fontsize=8)
    axes[2].set_ylabel("Macro G-mean", fontsize=8)
    axes[2].axhline(ce_g, color="r", lw=1, ls="--", alpha=0.7)
    axes[2].axhline(abl["cbam_both"]["macro_gmean_mean"], color="#4a86c8", lw=1, ls=":", alpha=0.8)
    axes[2].text(0.02, ce_g + 0.004, "CE", color="r", fontsize=7)
    axes[2].text(0.02, abl["cbam_both"]["macro_gmean_mean"] + 0.004, "CBAM", color="#4a86c8", fontsize=7)
    axes[2].legend(fontsize=7, frameon=False)
    axes[2].set_ylim(0.55, 0.85)
    axes[2].set_title("(c) LDAM margin coefficient C", fontsize=9)

    fig.suptitle("Ablation summary: feature attention is the only robust lever", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, "fig_ablation_bars.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    fig_confusion(); print("confusion_mt.png (English)")
    fig_training(); print("training_mt.png (English)")
    fig_tsne(); print("tsne_mt.png (English)")
    fig_recall_bar(); print("fig_recall_bar.png (English)")
    fig_ablation_bars(); print("fig_ablation_bars.png (English)")
    print("done")
