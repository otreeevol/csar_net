# -*- coding: utf-8 -*-
"""磁瓦真实缺陷：成本敏感诊断——定位 LDAM 不稳定的来源。

主实验发现 LDAM（间隔 + 类平衡重加权 DRW）在真实小样本上不稳定（2/3 种子退化）。
本脚本隔离两个分量：
  ldam_nodrw  LDAM 间隔（C=0.5）但去掉 DRW 重加权（drw=False）
  cs_ce       逆频率加权 CE（无间隔）——最简单的成本敏感损失
各 3 seed，写入 results_mt/metrics/diag.json。
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
METRIC_DIR = os.path.join(ROOT, "results_mt", "metrics")
os.makedirs(METRIC_DIR, exist_ok=True)

OUT = os.path.join(METRIC_DIR, "diag.json")

TRAIN_CFG = dict(epochs=40, lr=1e-3, weight_decay=1e-4,
                 gamma=0.1, temperature=0.07, ldam_C=0.5, ldam_drw=True)

CONFIGS = {
    "ldam_nodrw": dict(use_cs_cbam=False, loss="ldam", ldam_drw=False),
    "cs_ce":      dict(use_cs_cbam=False, loss="cs_ce"),
}

SEEDS = [42, 123, 2024]
METRIC_KEYS = ("accuracy", "macro_gmean", "macro_f1", "mcc")


def run_one(name, cfg, train_loader, test_loader, class_counts, device, seed):
    set_seed(seed)
    model = ResNet18CCPR(num_classes=6, use_cs_cbam=cfg["use_cs_cbam"],
                         use_proj=False, lam=cfg.get("lam", 0.0), in_channels=3,
                         stem="imagenet")
    train_cfg = dict(TRAIN_CFG)
    train_cfg["loss"] = cfg["loss"]
    train_cfg["use_cfc"] = False
    train_cfg["ldam_drw"] = cfg.get("ldam_drw", True)

    train_model(model, train_loader, test_loader, train_cfg, class_counts,
                device, seed=seed, verbose=False)

    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits, _, _ = model(xb.to(device))
            preds.append(logits.argmax(1).cpu())
            labels.append(yb)
    return metrics_from_pairs(torch.cat(labels).numpy(), torch.cat(preds).numpy(),
                              class_names=mt_data.CLASS_NAMES)


def main():
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True  # 关闭算法自动调优，保证跨 run 可复现
    device = get_device()
    print(f"设备: {device}", flush=True)
    train_loader, test_loader, class_counts, _ = mt_data.build_dataset()

    summary = {}
    for name, cfg in CONFIGS.items():
        t0 = time.time()
        rec = {k: [] for k in METRIC_KEYS}
        recalls = []
        for seed in SEEDS:
            m = run_one(name, cfg, train_loader, test_loader, class_counts, device, seed)
            for k in METRIC_KEYS:
                rec[k].append(m[k])
            recalls.append(m["recall"])
            print(f"  [{name}] seed={seed} acc={m['accuracy']:.4f} "
                  f"G-mean={m['macro_gmean']:.4f}", flush=True)
        arr = {k: np.array(rec[k]) for k in METRIC_KEYS}
        summary[name] = {f"{k}_mean": float(arr[k].mean()) for k in METRIC_KEYS}
        summary[name].update({f"{k}_std": float(arr[k].std()) for k in METRIC_KEYS})
        summary[name]["recall_mean"] = [float(x) for x in np.mean(recalls, axis=0)]
        print(f"  [{name}] G-mean = {summary[name]['macro_gmean_mean']:.4f} "
              f"± {summary[name]['macro_gmean_std']:.4f} ({time.time()-t0:.1f}s)", flush=True)
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump({"seeds": SEEDS, "class_names": mt_data.CLASS_NAMES,
                       "summary": summary}, f, ensure_ascii=False, indent=2)

    print(f"\n完成，见 {OUT}", flush=True)


if __name__ == "__main__":
    main()
