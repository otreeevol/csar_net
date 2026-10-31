# -*- coding: utf-8 -*-
"""训练引擎：Adam + 余弦退火，支持 CS-CBAM / LDAM / CFC 组合。"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from . import losses


def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loader(X, y, batch_size, shuffle=True):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)


def train_model(model, train_loader, test_loader, cfg, class_counts, device,
                seed=42, verbose=True):
    """训练并返回 (history, best_state)。

    cfg 字段：epochs, lr, weight_decay, loss(ce/cs_ce/ldam), use_cfc,
              gamma, temperature, ldam_C, ldam_drw, ldam_beta。
    """
    set_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    epochs = cfg["epochs"]
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch, eta_min=1e-6)

    # CS-CBAM 的类别不平衡偏置（log(N/n_c)）
    log_bias = losses.class_log_bias(class_counts).to(device)  # [K]

    history = {"train_loss": [], "test_acc": [], "test_gmean": []}
    best_gmean = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            class_bias = log_bias[yb] if model.use_cs_cbam else None
            logits, feat, proj = model(xb, class_bias)

            # 主损失
            if cfg["loss"] == "ldam":
                loss = losses.ldam_loss(logits, yb, class_counts,
                                        C=cfg.get("ldam_C", 0.5),
                                        drw=cfg.get("ldam_drw", True),
                                        beta=cfg.get("ldam_beta", 0.9999))
            elif cfg["loss"] == "cs_ce":
                w = losses.inverse_freq_weight(class_counts).to(device)
                loss = F.cross_entropy(logits, yb, weight=w)
            else:  # ce
                loss = F.cross_entropy(logits, yb)

            # 对比损失（CFC）
            if cfg.get("use_cfc", False) and proj is not None:
                c_loss = losses.info_nce_loss(proj, yb, cfg.get("temperature", 0.07))
                loss = loss + cfg.get("gamma", 0.1) * c_loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item() * xb.size(0)

        running /= max(1, len(train_loader.dataset))
        history["train_loss"].append(running)

        # 每轮评估（测试集小，开销可忽略）
        acc, gmean, _ = evaluate(model, test_loader, device)
        history["test_acc"].append(acc)
        history["test_gmean"].append(gmean)
        if gmean > best_gmean:
            best_gmean = gmean
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}/{epochs}  loss={running:.4f}  acc={acc:.4f}  G-mean={gmean:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_state


def evaluate(model, test_loader, device):
    """在测试集上计算 (accuracy, macro G-mean, preds, labels)。"""
    from .evaluate import metrics_from_pairs
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits, _, _ = model(xb.to(device))
            preds.append(logits.argmax(1).cpu())
            labels.append(yb.cpu())
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    m = metrics_from_pairs(labels, preds)
    return m["accuracy"], m["macro_gmean"], (preds, labels)
