# -*- coding: utf-8 -*-
"""磁瓦真实缺陷：论文图重制——展示"明星配置" CBAM / CBAM+LDAM。

主图的混淆矩阵与 t-SNE 之前画的是 ce vs ldam（展示失败配置），与论文叙事
（CBAM 是稳定涨点、CBAM+LDAM 最优）不符。本脚本在 seed=42 重训 cbam_ce 与
cbam_ldam（ce 的特征已存），重制 3 面板混淆矩阵与 t-SNE：
  confusion_matrices.png  ce vs cbam_ce vs cbam_ldam
  tsne.png                ce vs cbam_ce vs cbam_ldam
训练曲线 training_curves.png 已正确（含 5 配置），不动。
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
FIG_DIR = os.path.join(RESULT_DIR, "figures")
os.makedirs(METRIC_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

TRAIN_CFG = dict(epochs=40, lr=1e-3, weight_decay=1e-4,
                 gamma=0.1, temperature=0.07, ldam_C=0.5, ldam_drw=True)
CONFIGS = {
    "cbam_ce":   dict(use_cs_cbam=True,  lam=0.0, loss="ce"),
    "cbam_ldam": dict(use_cs_cbam=True,  lam=0.0, loss="ldam"),
}
CN = mt_data.CLASS_NAMES_CN
PANELS = ("ce", "cbam_ce", "cbam_ldam")
TITLE = {"ce": "基线 (CE)", "cbam_ce": "CBAM+CE", "cbam_ldam": "CBAM+LDAM"}


def train_and_save(name, cfg, train_loader, test_loader, counts, device):
    set_seed(42)
    model = ResNet18CCPR(num_classes=6, use_cs_cbam=cfg["use_cs_cbam"],
                         use_proj=False, lam=cfg["lam"], in_channels=3,
                         stem="imagenet")
    train_cfg = dict(TRAIN_CFG)
    train_cfg["loss"] = cfg["loss"]
    train_cfg["use_cfc"] = False
    train_model(model, train_loader, test_loader, train_cfg, counts, device,
                seed=42, verbose=False)

    model.eval()
    preds, labels, feats = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits, ft, _ = model(xb.to(device))
            preds.append(logits.argmax(1).cpu())
            labels.append(yb)
            feats.append(ft.cpu())
    m = metrics_from_pairs(torch.cat(labels).numpy(), torch.cat(preds).numpy(),
                           class_names=mt_data.CLASS_NAMES)
    np.save(os.path.join(METRIC_DIR, f"feat_{name}.npy"), torch.cat(feats).numpy())
    with open(os.path.join(METRIC_DIR, f"confusion_{name}.json"), "w", encoding="utf-8") as f:
        json.dump({"confusion_matrix": m["confusion_matrix"]}, f)
    return m


def plot_cm():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, k in zip(axes, PANELS):
        with open(os.path.join(METRIC_DIR, f"confusion_{k}.json"), encoding="utf-8") as f:
            cm = np.array(json.load(f)["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues")
        for i in range(6):
            for j in range(6):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
        ax.set_xticks(range(6)); ax.set_yticks(range(6))
        ax.set_xticklabels(CN, rotation=45, fontsize=8)
        ax.set_yticklabels(CN, fontsize=8)
        ax.set_xlabel("预测"); ax.set_ylabel("真实")
        ax.set_title(TITLE[k])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "confusion_matrices.png"), dpi=150); plt.close(fig)


def plot_tsne():
    from sklearn.manifold import TSNE
    labels = np.load(os.path.join(METRIC_DIR, "labels_test.npy"))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    for ax, k in zip(axes, PANELS):
        feats = np.load(os.path.join(METRIC_DIR, f"feat_{k}.npy"))
        X2 = tsne.fit_transform(feats)
        for c in range(6):
            ax.scatter(X2[labels == c, 0], X2[labels == c, 1], s=8, alpha=0.6, label=CN[c])
        ax.set_title(TITLE[k]); ax.set_xticks([]); ax.set_yticks([])
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
        m = train_and_save(name, cfg, train_loader, test_loader, counts, device)
        print(f"[{name}] seed42 G-mean={m['macro_gmean']:.4f} Acc={m['accuracy']:.4f}", flush=True)

    plot_cm(); print("已出混淆矩阵（3面板）", flush=True)
    plot_tsne(); print("已出 t-SNE（3面板）", flush=True)
    print("完成。", flush=True)


if __name__ == "__main__":
    main()
