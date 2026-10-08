"""
Task 1：Bag-of-Words 文本表示 + Logistic Regression。

包含两种词袋表示，均在 NYT 测试集上报告 Accuracy 与 Macro-F1：
    1. Binary Bag of Words：词在文档中出现记 1，否则记 0；
    2. Word Frequency：以词在文档中的出现次数作为特征。
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from data_utils import load_nyt, split_nyt, tokenize
from eval_utils import evaluate


def build_bow_features(train_texts, val_texts, test_texts, binary: bool):
    """将文本转换为词袋特征矩阵。

    参数:
        binary: True 表示二值词袋（0/1），False 表示词频计数。

    返回:
        (X_train, X_val, X_test) 三个稀疏矩阵。

    注意:
        词表仅在训练集上拟合（fit），验证/测试集只做 transform，
        避免测试集信息泄漏到特征构建阶段。
    """
    # analyzer 直接使用本项目统一的分词器（其内部已转小写）；
    # binary=True 得到 0/1 二值词袋，binary=False 得到词频计数。
    vectorizer = CountVectorizer(analyzer=tokenize, binary=binary)
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_val, X_test


def train_and_predict_lr(X_train, y_train, X_test):
    """在特征矩阵上训练 Logistic Regression，并返回测试集预测标签。"""
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    return clf.predict(X_test)


def run_one(splits, binary: bool):
    """运行一个词袋子实验并在测试集上评估。"""
    tr_x, tr_y = splits["train"]
    va_x, _ = splits["val"]
    te_x, te_y = splits["test"]

    X_train, X_val, X_test = build_bow_features(tr_x, va_x, te_x, binary)
    print(f"词袋矩阵形状 train={X_train.shape} test={X_test.shape}")

    y_pred = train_and_predict_lr(X_train, tr_y, X_test)
    name = "Task1-Binary-BoW" if binary else "Task1-Word-Frequency"
    return evaluate(te_y, y_pred, experiment=name)


def main():
    texts, labels = load_nyt()
    splits = split_nyt(texts, labels)

    results = [
        run_one(splits, binary=True),
        run_one(splits, binary=False),
    ]

    print("\n========== Task 1 ==========")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
