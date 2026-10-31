# -*- coding: utf-8 -*-
"""磁瓦真实缺陷：特征层注意力模块消融（SE / ECA / Coordinate Attention）。

与主实验的 cbam_ce 并列，验证"特征层注意力在真实长尾缺陷场景是否普遍中性"。
各配置 3 seed（CE 损失），输出 mean±std，写入 results_mt/metrics/attention_ablation.json。
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

OUT = os.path.join(METRIC_DIR, "attention_ablation.json")

TRAIN_CFG = dict(epochs=40, lr=1e-3, weight_decay=1e-4,
                 gamma=0.1, temperature=0.07, ldam_C=0.5, ldam_drw=True)

ATTN_TYPES = ("se", "eca", "ca")
SEEDS = [42, 123, 2024]
METRIC_KEYS = ("accuracy", "macro_gmean", "macro_f1", "mcc")


def run_one(attn_type, train_loader, test_loader, class_counts, device, seed):
    set_seed(seed)
    model = ResNet18CCPR(num_classes=6, use_cs_cbam=False, use_proj=False,
                         attn_type=attn_type, in_channels=3, stem="imagenet")
    train_cfg = dict(TRAIN_CFG)
    train_cfg["loss"] = "ce"
    train_cfg["use_cfc"] = False

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
    for attn in ATTN_TYPES:
        t0 = time.time()
        rec = {k: [] for k in METRIC_KEYS}
        recalls = []
        for seed in SEEDS:
            m = run_one(attn, train_loader, test_loader, class_counts, device, seed)
            for k in METRIC_KEYS:
                rec[k].append(m[k])
            recalls.append(m["recall"])
            print(f"  [{attn}] seed={seed} G-mean={m['macro_gmean']:.4f}", flush=True)

        arr = {k: np.array(rec[k]) for k in METRIC_KEYS}
        summary[attn] = {f"{k}_mean": float(arr[k].mean()) for k in METRIC_KEYS}
        summary[attn].update({f"{k}_std": float(arr[k].std()) for k in METRIC_KEYS})
        summary[attn]["recall_mean"] = [float(x) for x in np.mean(recalls, axis=0)]
        summary[attn]["recall_std"] = [float(x) for x in np.std(recalls, axis=0)]
        print(f"  [{attn}] G-mean = {summary[attn]['macro_gmean_mean']:.4f} "
              f"± {summary[attn]['macro_gmean_std']:.4f} ({time.time()-t0:.1f}s)", flush=True)
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump({"seeds": SEEDS, "class_names": mt_data.CLASS_NAMES,
                       "summary": summary}, f, ensure_ascii=False, indent=2)

    print("\n========== 磁瓦注意力消融（CE，mean±std） ==========", flush=True)
    for attn in ATTN_TYPES:
        s = summary[attn]
        print(f"{attn:5s} G-mean {s['macro_gmean_mean']:.4f}±{s['macro_gmean_std']:.4f}  "
              f"Acc {s['accuracy_mean']:.4f}", flush=True)
    print(f"\n完成，见 {OUT}", flush=True)


if __name__ == "__main__":
    main()
