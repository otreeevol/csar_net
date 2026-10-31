# -*- coding: utf-8 -*-
"""磁瓦真实缺陷：论文用图生成。

在 seed=42 上重训 5 个核心配置（与主实验同种子、同超参），记录逐 epoch 历史，
并保存 ce/ldam 的混淆矩阵 + 最优权重，输出 3 张图到 results_mt/figures/：
  training_curves.png   5 配置的 G-mean / Acc 曲线（含 CS-CBAM 塌缩）
  confusion_matrices.png ce vs ldam 混淆矩阵
  tsne.png              ce vs ldam 特征 t-SNE
"""
import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import mt_data
from .models import ResNet18CCPR
from .train import get_device, set_seed, train_model
from .evaluate import metrics_from_pairs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT, "results_mt")
METRIC_DIR = os.path.join(RESULT_DIR, "metrics")
CKPT_DIR = os.path.join(RESULT_DIR, "checkpoints")
FIG_DIR = os.path.join(RESULT_DIR, "figures")
for d in (METRIC_DIR, CKPT_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

TRAIN_CFG = dict(epochs=40, lr=1e-3, weight_decay=1e-4,
                 gamma=0.1, temperature=0.07, ldam_C=0.5, ldam_drw=True)
CONFIGS = {
    "ce":        dict(use_cs_cbam=False, lam=0.0, loss="ce"),
    "cbam_ce":   dict(use_cs_cbam=True,  lam=0.0, loss="ce"),
    "ldam":      dict(use_cs_cbam=False, lam=0.0, loss="ldam"),
    "cbam_ldam": dict(use_cs_cbam=True,  lam=0.0, loss="ldam"),
    "cs_cbam":   dict(use_cs_cbam=True,  lam=0.5, loss="ce"),
}
CN = mt_data.CLASS_NAMES_CN


def train_and_collect(name, cfg, train_loader, test_loader, counts, device):
    set_seed(42)
    model = ResNet18CCPR(num_classes=6, use_cs_cbam=cfg["use_cs_cbam"],
                         use_proj=False, lam=cfg["lam"], in_channels=3,
                         stem="imagenet")
    train_cfg = dict(TRAIN_CFG)
    train_cfg["loss"] = cfg["loss"]
    train_cfg["use_cfc"] = False
    history, best_state = train_model(model, train_loader, test_loader, train_cfg,
                                      counts, device, seed=42, verbose=False)

    # 保存逐 epoch 历史
    with open(os.path.join(METRIC_DIR, f"history_{name}.json"), "w", encoding="utf-8") as f:
        json.dump(history, f)

    model.eval()
    preds, labels = [], []
    feats = []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits, ft, _ = model(xb.to(device))
            preds.append(logits.argmax(1).cpu())
            labels.append(yb)
            feats.append(ft.cpu())
    m = metrics_from_pairs(torch.cat(labels).numpy(), torch.cat(preds).numpy(),
                           class_names=mt_data.CLASS_NAMES)
    feats = torch.cat(feats).numpy()
    lab = torch.cat(labels).numpy()

    if name in ("ce", "ldam"):
        torch.save(best_state, os.path.join(CKPT_DIR, f"mt_{name}.pth"))
        with open(os.path.join(METRIC_DIR, f"confusion_{name}.json"), "w", encoding="utf-8") as f:
            json.dump({"confusion_matrix": m["confusion_matrix"]}, f)
        np.save(os.path.join(METRIC_DIR, f"feat_{name}.npy"), feats)
        np.save(os.path.join(METRIC_DIR, "labels_test.npy"), lab)
    return history, m


def plot_curves():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = {"ce": "CE (基线)", "cbam_ce": "CBAM+CE", "ldam": "LDAM",
              "cbam_ldam": "CBAM+LDAM", "cs_cbam": "CS-CBAM"}
    for name, lab in labels.items():
        with open(os.path.join(METRIC_DIR, f"history_{name}.json"), encoding="utf-8") as f:
            h = json.load(f)
        axes[0].plot(h["test_gmean"], label=lab)
        axes[1].plot(h["test_acc"], label=lab)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("G-mean (测试集)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (测试集)")
    axes[0].set_title("测试集 G-mean 曲线（CS-CBAM 塌缩）")
    axes[1].set_title("测试集准确率曲线")
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "training_curves.png"), dpi=150); plt.close(fig)


def plot_cm():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    title = {"ce": "基线 (CE)", "ldam": "LDAM"}
    for ax, k in zip(axes, ("ce", "ldam")):
        with open(os.path.join(METRIC_DIR, f"confusion_{k}.json"), encoding="utf-8") as f:
            cm = np.array(json.load(f)["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues")
        for i in range(6):
            for j in range(6):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
        ax.set_xticks(range(6)); ax.set_yticks(range(6))
        ax.set_xticklabels(CN, rotation=45, fontsize=9)
        ax.set_yticklabels(CN, fontsize=9)
        ax.set_xlabel("预测"); ax.set_ylabel("真实")
        ax.set_title(title[k])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "confusion_matrices.png"), dpi=150); plt.close(fig)


def plot_tsne():
    from sklearn.manifold import TSNE
    labels = np.load(os.path.join(METRIC_DIR, "labels_test.npy"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    title = {"ce": "基线 (CE)", "ldam": "LDAM"}
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    for ax, k in zip(axes, ("ce", "ldam")):
        feats = np.load(os.path.join(METRIC_DIR, f"feat_{k}.npy"))
        X2 = tsne.fit_transform(feats)
        for c in range(6):
            ax.scatter(X2[labels == c, 0], X2[labels == c, 1], s=8, alpha=0.6, label=CN[c])
        ax.set_title(title[k]); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=7, markerscale=1.5, loc="best")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "tsne.png"), dpi=150); plt.close(fig)


def main():
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True  # 关闭算法自动调优，保证跨 run 可复现
    device = get_device()
    print(f"设备: {device}", flush=True)
    train_loader, test_loader, counts, _ = mt_data.build_dataset()

    for name, cfg in CONFIGS.items():
        history, m = train_and_collect(name, cfg, train_loader, test_loader, counts, device)
        print(f"[{name}] seed42 G-mean={m['macro_gmean']:.4f} Acc={m['accuracy']:.4f}", flush=True)

    plot_curves(); print("已出训练曲线", flush=True)
    plot_cm(); print("已出混淆矩阵", flush=True)
    plot_tsne(); print("已出 t-SNE", flush=True)
    print("完成。", flush=True)


if __name__ == "__main__":
    main()
