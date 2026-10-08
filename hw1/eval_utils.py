"""
分类评价指标：Accuracy 与 Macro-F1。

三个 Task 统一调用本模块，保证指标口径一致。
"""

from typing import Dict, Sequence


def accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    """返回准确率：分类正确样本数占总样本数的比例。"""
    y_true, y_pred = list(y_true), list(y_pred)
    if len(y_true) == 0:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def macro_f1(y_true: Sequence, y_pred: Sequence) -> float:
    """返回 Macro-F1：先计算每个类别的 F1，再对各类别做无加权平均。

    在类别不均衡时，Macro-F1 对每个类别同等加权，相较 Accuracy
    更能反映模型在小类别上的分类表现。某类别既无真实样本也无预测时
    其 F1 记为 0（与 sklearn 的 zero_division=0 口径一致）。
    """
    y_true, y_pred = list(y_true), list(y_pred)
    classes = sorted(set(y_true) | set(y_pred))
    if not classes:
        return 0.0

    f1_scores = []
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        f1_scores.append(f1)

    return sum(f1_scores) / len(classes)


def evaluate(y_true: Sequence, y_pred: Sequence,
             experiment: str = "") -> Dict[str, float]:
    """计算并打印 Accuracy 与 Macro-F1，返回结果字典以便汇总。"""
    acc = accuracy(y_true, y_pred)
    f1 = macro_f1(y_true, y_pred)
    result = {
        "experiment": experiment,
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(f1), 4),
    }
    print(f"[{experiment or 'result'}] "
          f"Accuracy={result['accuracy']:.4f}  Macro-F1={result['macro_f1']:.4f}")
    return result
