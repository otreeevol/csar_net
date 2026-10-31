# -*- coding: utf-8 -*-
"""磁瓦表面缺陷（Magnetic-Tile-Defect）真实工业数据集加载。

数据来源：Surface-Defect-Detection-master / Magnetic-Tile-Defect（真实磁瓦表面缺陷实拍图）。
6 类、天然长尾：Free(无缺陷) 952 / Blowhole 115 / Uneven 103 / Break 85 / Crack 57 / Fray 32。

图像为灰度（L 模式）、尺寸不一（短边 103~392）。这里统一：
  灰度 -> 3 通道（复制）-> Resize 到固定边长（BILINEAR，一次性）。
全部图像一次性读入内存缓存（约 200MB，uint8），训练时只做廉价的水平/垂直翻转增强，
避免逐 epoch 重复 open/resize/rotate 的 IO 开销。
"""
import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "magnetic_tile")

# 类别按目录名排序（目录带 MT_ 前缀，展示名去掉前缀）
CLASS_DIRS = ["MT_Blowhole", "MT_Break", "MT_Crack", "MT_Fray", "MT_Free", "MT_Uneven"]
CLASS_NAMES = ["Blowhole", "Break", "Crack", "Fray", "Free", "Uneven"]
CLASS_NAMES_CN = ["气孔", "破损", "裂纹", "磨损", "无缺陷", "不平"]


def _list_paths():
    paths, labels = [], []
    for c, name in enumerate(CLASS_DIRS):
        d = os.path.join(DATA_DIR, name)
        for p in sorted(glob.glob(os.path.join(d, "*.jpg"))):
            paths.append(p)
            labels.append(c)
    return paths, np.array(labels, dtype=np.int64)


def _load_cache(paths, size):
    """一次性读入并 resize 所有图像，返回 uint8 numpy 数组列表。"""
    cache = []
    for p in paths:
        img = Image.open(p).convert("L").convert("RGB")
        img = img.resize((size, size), Image.BILINEAR)
        cache.append(np.asarray(img, dtype=np.uint8))
    return cache


class MagneticTileDataset(Dataset):
    """从内存缓存索引图像；训练集做水平/垂直翻转增强（廉价）。"""

    def __init__(self, cache, labels, indices, train=True):
        self.cache = cache
        self.labels = labels
        self.indices = indices
        self.train = train

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img = self.cache[idx]  # (H, W, 3) uint8
        if self.train:
            if random.random() < 0.5:
                img = img[:, ::-1, :]
            if random.random() < 0.5:
                img = img[::-1, :, :]
        img = np.ascontiguousarray(img)  # 翻转后可能负步长，强制 C 连续
        t = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).div_(255.0)
        return t, int(self.labels[idx])


def build_dataset(test_frac=0.2, base_seed=42, size=224, batch_size=32):
    """分层切分训练/测试，返回 (train_loader, test_loader, class_counts, class_names)。

    数据划分用 base_seed 固定（保证各模型配置公平比较），训练随机性由训练时 seed 控制。
    class_counts 为【训练集】每类样本数（供 LDAM / CS-CBAM 使用）。
    """
    from sklearn.model_selection import train_test_split
    paths, labels = _list_paths()
    cache = _load_cache(paths, size)
    idx = np.arange(len(paths))
    tr_idx, te_idx = train_test_split(
        idx, test_size=test_frac, stratify=labels, random_state=base_seed)

    train_ds = MagneticTileDataset(cache, labels, tr_idx, train=True)
    test_ds = MagneticTileDataset(cache, labels, te_idx, train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    counts = np.bincount(labels[tr_idx], minlength=len(CLASS_NAMES))
    return train_loader, test_loader, counts.astype(np.int64), CLASS_NAMES


if __name__ == "__main__":
    paths, labels = _list_paths()
    print(f"总样本 {len(paths)}，每类 {np.bincount(labels).tolist()}")
    tr, te, counts, names = build_dataset()
    print(f"训练每类 {counts.tolist()}，测试集 {len(te.dataset)}")
