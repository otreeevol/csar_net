# -*- coding: utf-8 -*-
"""评估指标：准确率、每类召回/精确率/F1、宏 G-mean、宏 F1、MCC、混淆矩阵。"""
import numpy as np
from sklearn.metrics import matthews_corrcoef

from .data_gen import CLASS_NAMES, CLASS_NAMES_CN


def metrics_from_pairs(labels, preds, class_names=CLASS_NAMES):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    K = len(class_names)
    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[t, p] += 1

    recall = np.zeros(K)
    precision = np.zeros(K)
    f1 = np.zeros(K)
    for c in range(K):
        tp = cm[c, c]
        fn = cm[c].sum() - tp
        fp = cm[:, c].sum() - tp
        recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c]) if (precision[c] + recall[c]) > 0 else 0.0

    # 宏 G-mean = (∏ recall_c)^(1/K)，对 0 做平滑
    recall_safe = np.clip(recall, 1e-6, 1.0)
    macro_gmean = float(np.exp(np.mean(np.log(recall_safe))))
    macro_f1 = float(np.mean(f1))
    accuracy = float(cm.trace() / cm.sum())
    mcc = float(matthews_corrcoef(labels, preds)) if K > 1 else 0.0

    return {
        "accuracy": accuracy,
        "macro_gmean": macro_gmean,
        "macro_f1": macro_f1,
        "mcc": mcc,
        "recall": recall.tolist(),
        "precision": precision.tolist(),
        "f1": f1.tolist(),
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
    }


def metrics_summary_line(m):
    return (f"acc={m['accuracy']:.4f}  G-mean={m['macro_gmean']:.4f}  "
            f"F1={m['macro_f1']:.4f}  MCC={m['mcc']:.4f}")
