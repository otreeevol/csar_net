# -*- coding: utf-8 -*-
"""合成控制图数据生成与 Gramian 角场（GAF）变换。

控制图模式识别（CCPR）基准：六类异常模式，时间序列长度 T=20，
生成公式 x(t) = tau(t) + d(t)，其中 tau(t) ~ N(0, NOISE_STD) 为随机噪声，
d(t) 为各类别的模式函数（见 generate_pattern）。

一维序列经 Gramian 角场（Gramian Angular Field, GAF）变换转为
2×20×20 二维图（GASF + GADF 两通道），编码时序的角相关结构，
喂给 ResNet18。
"""
import os
import numpy as np

T = 20  # 时间序列长度
CLASS_NAMES = ["Uptrend", "Downtrend", "Upshift", "Downshift", "Cyclical", "Systematic"]
CLASS_NAMES_CN = ["上升趋势", "下降趋势", "向上偏移", "向下偏移", "周期性", "系统性"]
NUM_CLASSES = len(CLASS_NAMES)

# 各类别模式参数（幅度，配合噪声 N(0,NOISE_STD) 调至"可区分但不平凡"）
NOISE_STD = 0.5  # 噪声标准差（平衡数据约 0.88 准确率，可辨但不平凡）
D1 = 0.10        # 趋势斜率（Uptrend/Downtrend）
D2 = 1.0         # 偏移幅度（Upshift/Downshift）
D3 = 1.0         # 周期幅度（Cyclical）
D4 = 0.3         # 系统性交替幅度（Systematic，最难类，幅度最小）
OMEGA = 8.0      # 周期


def generate_pattern(pattern, seed=None, T=T, d1=D1, d2=D2, d3=D3, d4=D4, omega=OMEGA,
                     noise_std=NOISE_STD):
    """生成单条长度为 T 的控制图时间序列。"""
    rng = np.random.RandomState(seed)
    t = np.arange(T, dtype=np.float64)
    tau = rng.randn(T) * noise_std
    if pattern == "Uptrend":
        d = d1 * t
    elif pattern == "Downtrend":
        d = -d1 * t
    elif pattern == "Upshift":
        change = rng.randint(2, T - 2)
        d = np.where(t >= change, d2, 0.0)
    elif pattern == "Downshift":
        change = rng.randint(2, T - 2)
        d = np.where(t >= change, -d2, 0.0)
    elif pattern == "Cyclical":
        d = d3 * np.sin(2.0 * np.pi * t / omega)
    elif pattern == "Systematic":
        d = d4 * ((-1.0) ** t)
    else:
        raise ValueError(f"未知模式: {pattern}")
    return tau + d


def to_gaf(x):
    """一维序列 -> 2×20×20 Gramian 角场图（GASF + GADF 两通道）。

    先将序列线性缩放到 [-1,1]，取极坐标角 phi=arccos(x)，再分别计算
    GASF = cos(phi_i + phi_j)（和角场，对称，编码绝对角相关）与
    GADF = sin(phi_i - phi_j)（差角场，反对称，编码时间方向），二者
    堆叠为 2 通道图像，完整保留序列的形状与方向信息。
    """
    x = np.asarray(x, dtype=np.float64)
    xmin, xmax = x.min(), x.max()
    if xmax - xmin < 1e-8:
        x = np.zeros_like(x)
    else:
        x = (x - xmin) / (xmax - xmin) * 2.0 - 1.0  # [-1, 1]
    phi = np.arccos(np.clip(x, -1.0, 1.0))
    gasf = np.cos(phi[:, None] + phi[None, :])
    gadf = np.sin(phi[:, None] - phi[None, :])
    return np.stack([gasf, gadf], axis=0).astype(np.float32)  # [2, T, T]


def _to_gaf_batch(x):
    """批量 GAF 变换：x [n, T] -> [n, 2, T, T]。"""
    xmin = x.min(axis=1, keepdims=True)
    xmax = x.max(axis=1, keepdims=True)
    rng = xmax - xmin
    xn = np.where(rng < 1e-8, 0.0, (x - xmin) / (rng + 1e-8) * 2.0 - 1.0)
    phi = np.arccos(np.clip(xn, -1.0, 1.0))
    gasf = np.cos(phi[:, :, None] + phi[:, None, :])
    gadf = np.sin(phi[:, :, None] - phi[:, None, :])
    return np.stack([gasf, gadf], axis=1).astype(np.float32)  # [n, 2, T, T]


def _gen_class_samples(pattern, n, base_seed):
    """向量化生成某模式 n 条样本的 GAF 图数组 [n, 2, T, T]。"""
    rng = np.random.RandomState(base_seed)
    t = np.arange(T, dtype=np.float64)
    tau = rng.randn(n, T) * NOISE_STD  # [n, T]
    if pattern == "Uptrend":
        d = D1 * t
    elif pattern == "Downtrend":
        d = -D1 * t
    elif pattern == "Upshift":
        change = rng.randint(2, T - 2, size=n)
        d = np.where(t[None, :] >= change[:, None], D2, 0.0)
    elif pattern == "Downshift":
        change = rng.randint(2, T - 2, size=n)
        d = np.where(t[None, :] >= change[:, None], -D2, 0.0)
    elif pattern == "Cyclical":
        d = D3 * np.sin(2.0 * np.pi * t / OMEGA)
    elif pattern == "Systematic":
        d = D4 * ((-1.0) ** t)
    else:
        raise ValueError(f"未知模式: {pattern}")
    d = np.broadcast_to(d, (n, T))
    return _to_gaf_batch(tau + d)


def build_dataset(ir="1:100", test_per_class=300, base_seed=42):
    """构建长尾训练集 + 平衡测试集。

    ir='1:100'：头类(Uptrend/Downtrend)3000、中类(Upshift/Downshift)1500、
              尾类(Cyclical/Systematic)30。
    ir='1:20' ：头类1000、中类500、尾类50。
    测试集每类 test_per_class 张（平衡），保证 G-mean 评估公平。
    """
    counts = {
        "1:100": {"head": 3000, "mid": 1500, "tail": 30},
        "1:20": {"head": 1000, "mid": 500, "tail": 50},
    }
    c = counts[ir]
    per_class_train = [
        c["head"], c["head"], c["mid"], c["mid"], c["tail"], c["tail"],
    ]  # Uptrend, Downtrend, Upshift, Downshift, Cyclical, Systematic

    X_train, y_train = [], []
    X_test, y_test = [], []
    for cls in range(NUM_CLASSES):
        # 训练集
        imgs = _gen_class_samples(CLASS_NAMES[cls], per_class_train[cls], base_seed + cls * 100000)
        X_train.append(imgs)
        y_train.append(np.full(per_class_train[cls], cls, dtype=np.int64))
        # 测试集
        timgs = _gen_class_samples(CLASS_NAMES[cls], test_per_class, base_seed + 999999 + cls * 100000)
        X_test.append(timgs)
        y_test.append(np.full(test_per_class, cls, dtype=np.int64))

    X_train = np.concatenate(X_train, 0)
    y_train = np.concatenate(y_train, 0)
    X_test = np.concatenate(X_test, 0)
    y_test = np.concatenate(y_test, 0)

    # 打乱训练集（测试集保持类内有序即可）
    idx = np.random.RandomState(base_seed).permutation(len(y_train))
    return (X_train[idx], y_train[idx]), (X_test, y_test), per_class_train


def save_dataset(path_dir, ir="1:100"):
    """生成并保存数据集为 .npz，返回各类样本数。"""
    os.makedirs(path_dir, exist_ok=True)
    (Xtr, ytr), (Xte, yte), per_cls = build_dataset(ir=ir)
    out = os.path.join(path_dir, f"ccpr_{ir.replace(':', '_')}.npz")
    np.savez_compressed(out, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    return out, per_cls


if __name__ == "__main__":
    # 自检：生成数据并打印信息
    (Xtr, ytr), (Xte, yte), per_cls = build_dataset("1:100")
    print("训练集:", Xtr.shape, ytr.shape, "测试集:", Xte.shape, yte.shape)
    print("每类训练样本数:", list(zip(CLASS_NAMES, per_cls)))
    print("GAF 图取值范围:", Xtr.min(), Xtr.max())
    # 抽查一条序列与 GAF 图
    x = generate_pattern("Uptrend", seed=0)
    print("Uptrend 序列:", np.round(x, 3))
    print("GAF 形状:", to_gaf(x).shape)
