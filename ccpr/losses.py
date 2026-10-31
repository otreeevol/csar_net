# -*- coding: utf-8 -*-
"""损失函数：交叉熵 / 成本敏感 CE / LDAM 间隔损失 / InfoNCE 对比损失。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def class_log_bias(class_counts):
    """计算每类的 log(N/n_c) 偏置（用于 CS-CBAM）。

    返回 [K] 张量；少数类 n_c 小 -> log(N/n_c) 大。
    """
    n = torch.as_tensor(class_counts, dtype=torch.float32)
    N = n.sum()
    return torch.log(N / n.clamp(min=1))


def inverse_freq_weight(class_counts):
    """逆频率权重 w_c ∝ N/n_c（成本敏感 CE）。"""
    n = torch.as_tensor(class_counts, dtype=torch.float32)
    w = n.sum() / n.clamp(min=1)
    return w / w.sum() * len(w)


def effective_num_weight(class_counts, beta=0.9999):
    """类平衡重加权（有效样本数，(1-β)/(1-β^n_c)）。"""
    n = torch.as_tensor(class_counts, dtype=torch.float32)
    eff = (1.0 - beta) / (1.0 - beta ** n.clamp(min=1))
    return eff / eff.sum() * len(eff)


def ldam_loss(logits, targets, class_counts, C=0.5, s=30.0, drw=False, beta=0.9999):
    """LDAM 间隔损失（Cao et al., NeurIPS 2019）。

    对真实类 logit 减去间隔 Δ_c = C / n_c^(1/4)，少数类间隔更大，收紧
    泛化界并纠正 Softmax 频率偏置。drw=True 时叠加类平衡重加权。
    """
    n = torch.as_tensor(class_counts, dtype=logits.dtype, device=logits.device)
    margin = C / (n ** 0.25)  # [K]
    onehot = F.one_hot(targets, logits.size(1)).to(logits.dtype)
    z = logits - margin.unsqueeze(0) * onehot  # 仅真实类减间隔
    if drw:
        eff = (1.0 - beta) / (1.0 - beta ** n.clamp(min=1))
        w = eff / eff.sum() * len(eff)
        return F.cross_entropy(z, targets, weight=w)
    return F.cross_entropy(z, targets)


def info_nce_loss(features, labels, temperature=0.07):
    """监督 InfoNCE 对比损失（同类拉近、异类推开，排除自身）。

    features: [B, D]（内部做 L2 归一化）；labels: [B]。
    """
    device = features.device
    B = features.size(0)
    features = F.normalize(features, dim=1)
    sim = features @ features.T / temperature  # [B, B]
    mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()  # 同类
    mask = mask - torch.eye(B, device=device)  # 去除自身
    denom_mask = 1.0 - torch.eye(B, device=device)
    neg_sum = (torch.exp(sim) * denom_mask).sum(dim=1)  # [B]
    pos_sum = (torch.exp(sim) * mask).sum(dim=1)  # [B]
    loss = -torch.log(pos_sum / neg_sum.clamp(min=1e-8) + 1e-8)
    has_pos = mask.sum(dim=1) > 0
    if has_pos.sum() == 0:
        return torch.zeros((), device=device)
    return loss[has_pos].mean()
