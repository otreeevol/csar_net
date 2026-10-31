# -*- coding: utf-8 -*-
"""磁瓦真实缺陷（Magnetic-Tile-Defect）长尾分类：多种子主实验。

沿用 CCPR 的解耦框架（ResNet18 + CBAM + LDAM + CS-CBAM），输入改为真实灰度缺陷图
（3 通道、224×224）。对 5 个核心配置各跑 3 个随机种子，输出 mean±std，
写入 results_mt/metrics/multiseed.json；同时保存 ce/ldam 的 seed42 最优权重供 t-SNE。

配置说明（与 CCPR 一致）：
  ce        基线（无注意力、交叉熵）
  cbam_ce   普通 CBAM（lam=0，无类别偏置）+ CE
  ldam      LDAM 间隔损失（损失层成本敏感）
  cbam_ldam CBAM + LDAM
  cs_cbam   成本敏感 CBAM（lam=0.5，把 λ·log(N/n_c) 注入注意力 logit）——预期塌缩
"""
import os
import json
import time
import numpy as np
import torch

from . import mt_data
from .models import ResNet18CCPR
from .train import get_device, set_seed, train_model
from .evaluate import metrics_from_pairs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT, "results_mt")
METRIC_DIR = os.path.join(RESULT_DIR, "metrics")
CKPT_DIR = os.path.join(RESULT_DIR, "checkpoints")
os.makedirs(METRIC_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

OUT = os.path.join(METRIC_DIR, "multiseed.json")

TRAIN_CFG = dict(epochs=40, lr=1e-3, weight_decay=1e-4,
                 gamma=0.1, temperature=0.07, ldam_C=0.5, ldam_drw=True)

CONFIGS = {
    "ce":        dict(use_cs_cbam=False, lam=0.0, loss="ce"),
    "cbam_ce":   dict(use_cs_cbam=True,  lam=0.0, loss="ce"),
    "ldam":      dict(use_cs_cbam=False, lam=0.0, loss="ldam"),
    "cbam_ldam": dict(use_cs_cbam=True,  lam=0.0, loss="ldam"),
    "cs_cbam":   dict(use_cs_cbam=True,  lam=0.5, loss="ce"),
}

SEEDS = [42, 123, 2024]
METRIC_KEYS = ("accuracy", "macro_gmean", "macro_f1", "mcc")
SAVE_CKPT = ("ce", "ldam")  # 供 t-SNE 可视化


def run_one(name, cfg, train_loader, test_loader, class_counts, device, seed):
    set_seed(seed)
    model = ResNet18CCPR(num_classes=6, use_cs_cbam=cfg["use_cs_cbam"],
                         use_proj=False, lam=cfg["lam"], in_channels=3,
                         stem="imagenet")

    train_cfg = dict(TRAIN_CFG)
    train_cfg["loss"] = cfg["loss"]
    train_cfg["use_cfc"] = False

    _, best_state = train_model(model, train_loader, test_loader, train_cfg,
                                class_counts, device, seed=seed, verbose=False)

    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits, _, _ = model(xb.to(device))
            preds.append(logits.argmax(1).cpu())
            labels.append(yb)
    m = metrics_from_pairs(torch.cat(labels).numpy(), torch.cat(preds).numpy(),
                           class_names=mt_data.CLASS_NAMES)
    if name in SAVE_CKPT and seed == SEEDS[0]:
        torch.save(best_state, os.path.join(CKPT_DIR, f"mt_{name}.pth"))
    return m


def main():
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True  # 关闭算法自动调优，保证跨 run 可复现
    device = get_device()
    print(f"设备: {device}", flush=True)
    train_loader, test_loader, class_counts, _ = mt_data.build_dataset()
    print(f"训练每类 {class_counts.tolist()}，测试集 {len(test_loader.dataset)}",
          flush=True)

    per_run = {name: {} for name in CONFIGS}
    summary = {}
    for name, cfg in CONFIGS.items():
        t0 = time.time()
        rec = {k: [] for k in METRIC_KEYS}
        recalls = []
        for seed in SEEDS:
            m = run_one(name, cfg, train_loader, test_loader, class_counts,
                        device, seed)
            per_run[name][str(seed)] = {k: m[k] for k in METRIC_KEYS}
            per_run[name][str(seed)]["recall"] = m["recall"]
            for k in METRIC_KEYS:
                rec[k].append(m[k])
            recalls.append(m["recall"])
            print(f"  [{name}] seed={seed}  acc={m['accuracy']:.4f} "
                  f"G-mean={m['macro_gmean']:.4f}", flush=True)

        arr = {k: np.array(rec[k]) for k in METRIC_KEYS}
        summary[name] = {
            **{f"{k}_mean": float(arr[k].mean()) for k in METRIC_KEYS},
            **{f"{k}_std": float(arr[k].std()) for k in METRIC_KEYS},
            "recall_mean": [float(x) for x in np.mean(recalls, axis=0)],
            "recall_std": [float(x) for x in np.std(recalls, axis=0)],
        }
        print(f"  [{name}] G-mean = {summary[name]['macro_gmean_mean']:.4f} "
              f"± {summary[name]['macro_gmean_std']:.4f}  ({time.time()-t0:.1f}s)",
              flush=True)
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump({"seeds": SEEDS, "class_names": mt_data.CLASS_NAMES,
                       "per_run": per_run, "summary": summary},
                      f, ensure_ascii=False, indent=2)

    print("\n========== 磁瓦缺陷多种子汇总（mean ± std） ==========", flush=True)
    for name in CONFIGS:
        s = summary[name]
        print(f"{name:10s} G-mean {s['macro_gmean_mean']:.4f}±{s['macro_gmean_std']:.4f}  "
              f"Acc {s['accuracy_mean']:.4f}±{s['accuracy_std']:.4f}", flush=True)
    print(f"\n完成，结果见 {OUT}", flush=True)


if __name__ == "__main__":
    main()
