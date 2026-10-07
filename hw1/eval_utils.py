"""
分类评价指标：Accuracy 与 Macro-F1。

三个 Task 统一调用本模块，保证指标口径一致。
"""

from typing import Dict, Sequence


def accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    """返回准确率：分类正确样本数占总样本数的比例。"""
    raise NotImplementedError


def macro_f1(y_true: Sequence, y_pred: Sequence) -> float:
    """返回 Macro-F1：先计算每个类别的 F1，再对各类别做无加权平均。

    在类别不均衡时，Macro-F1 对每个类别同等加权，相较 Accuracy
    更能反映模型在小类别上的分类表现。
    """
    raise NotImplementedError


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
