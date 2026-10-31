# -*- coding: utf-8 -*-
"""磁瓦真实缺陷：补充消融——补全论文的消融面。

在已有主实验（ce/cbam_ce/ldam/cbam_ldam/cs_cbam）与诊断（ldam_nodrw/cs_ce）基础上，
补做三组消融（各 3 seed，写入 results_mt/metrics/ablation_extra.json）：

  A. CBAM 注入位置（attn_stages）：layer2 / layer3 / layer4 / layer3+4（默认）
  B. CBAM 子模块分解（attn_mode）：channel-only / spatial-only / both（完整 CBAM）
  C. LDAM 间隔系数 C 敏感性：
       margin-only（无 DRW）下 C ∈ {0.1, 0.3, 1.0}（C=0.5 已有 diag.json）
       完整 CSAR-Net（CBAM+LDAM, DRW 开）下 C ∈ {0.1, 0.3, 1.0}（C=0.5 已有主实验）
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

OUT = os.path.join(METRIC_DIR, "ablation_extra.json")

TRAIN_CFG = dict(epochs=40, lr=1e-3, weight_decay=1e-4,
                 gamma=0.1, temperature=0.07, ldam_C=0.5, ldam_drw=True)

SEEDS = [42, 123, 2024]
METRIC_KEYS = ("accuracy", "macro_gmean", "macro_f1", "mcc")


def make_configs():
    """返回 (name -> cfg)。cfg 含模型与损失层所有可调项。"""
    cfgs = {}
    # A. CBAM 注入位置（CE 损失，普通 CBAM lam=0）
    for stages, label in [((2,), "cbam_s2"), ((3,), "cbam_s3"),
                          ((4,), "cbam_s4"), ((3, 4), "cbam_s34")]:
        cfgs[label] = dict(use_cs_cbam=True, lam=0.0, loss="ce",
                           attn_stages=stages, attn_mode="both")
    # B. CBAM 子模块分解（layer3+4，CE 损失）
    for mode, label in [("channel", "cbam_ch"), ("spatial", "cbam_sp"),
                        ("both", "cbam_both")]:
        cfgs[label] = dict(use_cs_cbam=True, lam=0.0, loss="ce",
                           attn_stages=(3, 4), attn_mode=mode)
    # C1. LDAM 间隔系数 C（margin-only，无 DRW）
    for C in (0.1, 0.3, 1.0):
        cfgs[f"ldam_nodrw_C{C:g}"] = dict(use_cs_cbam=False, lam=0.0, loss="ldam",
                                          ldam_drw=False, ldam_C=C)
    # C2. LDAM 间隔系数 C（完整 CSAR-Net，DRW 开）
    for C in (0.1, 0.3, 1.0):
        cfgs[f"cbam_ldam_C{C:g}"] = dict(use_cs_cbam=True, lam=0.0, loss="ldam",
                                         ldam_drw=True, ldam_C=C)
    return cfgs


def run_one(name, cfg, train_loader, test_loader, class_counts, device, seed):
    set_seed(seed)
    model = ResNet18CCPR(num_classes=6, use_cs_cbam=cfg["use_cs_cbam"],
                         use_proj=False, lam=cfg.get("lam", 0.0), in_channels=3,
                         stem="imagenet", attn_stages=cfg.get("attn_stages", (3, 4)),
                         attn_mode=cfg.get("attn_mode", "both"))
    train_cfg = dict(TRAIN_CFG)
    train_cfg["loss"] = cfg["loss"]
    train_cfg["use_cfc"] = False
    train_cfg["ldam_drw"] = cfg.get("ldam_drw", True)
    train_cfg["ldam_C"] = cfg.get("ldam_C", 0.5)

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

    cfgs = make_configs()
    summary = {}
    for name, cfg in cfgs.items():
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
