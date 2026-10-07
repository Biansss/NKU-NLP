"""
Task 2：Word Embedding 文本表示 + Logistic Regression。

统一使用 100 维词向量，文档向量由文档内有效词向量的平均值得到，
分类器为 Logistic Regression。包含三组实验，均在 NYT 测试集上
报告 Accuracy 与 Macro-F1：
    1. 预训练 GloVe（glove.6B.100d）；
    2. 在 AG News 语料上训练的 Word2Vec；
    3. 在 NYT 训练集语料上训练的 Word2Vec。

为统一接口，GloVe 与 gensim Word2Vec 的词向量均整理为
{单词: np.ndarray} 字典后再做文档聚合。
"""

import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression

from data_utils import load_nyt, load_ag, split_nyt, tokenize, GLOVE_TXT
from eval_utils import evaluate

VECTOR_SIZE = 100


def load_glove(path: str = GLOVE_TXT, limit: int = None) -> dict:
    """读取 GloVe 文本格式词向量。

    每行格式为「单词 + 100 个以空格分隔的数值」。

    参数:
        limit: 仅读取前 limit 行（调试用），None 表示读取全部。

    返回:
        {word: np.ndarray(shape=(100,), dtype=float32)} 字典。
    """
    raise NotImplementedError


def train_word2vec(sentences, vector_size: int = VECTOR_SIZE) -> Word2Vec:
    """在分词语料上训练 Word2Vec 模型。

    参数:
        sentences: 分词语料，每个元素为一篇文档的词列表 List[List[str]]。

    返回:
        训练完成的 gensim Word2Vec 模型。
    """
    raise NotImplementedError


def w2v_to_dict(model: Word2Vec) -> dict:
    """将 gensim 模型的词向量导出为 {单词: np.ndarray}，与 GloVe 接口统一。"""
    return {word: np.asarray(model.wv[word], dtype=np.float32)
            for word in model.wv.index_to_key}


def doc_vector(tokens, embeddings: dict) -> np.ndarray:
    """将一篇文档聚合为单个词向量（有效词向量的平均值）。

    仅对在 embeddings 中存在的词（有效词）求平均，未登录词（OOV）跳过；
    若文档不含任何有效词则返回全 0 向量，避免除零。
    """
    raise NotImplementedError


def corpus_to_matrix(texts, embeddings: dict) -> np.ndarray:
    """将语料中每篇文档聚合为平均词向量，并堆叠为特征矩阵。

    返回:
        shape 为 (文档数, 100) 的 np.ndarray。
    """
    raise NotImplementedError


def run_with_embeddings(splits, embeddings: dict, experiment: str):
    """对给定词向量构建文档特征，训练 Logistic Regression 并在测试集评估。"""
    tr_x, tr_y = splits["train"]
    _va_x, _va_y = splits["val"]
    te_x, te_y = splits["test"]

    X_train = corpus_to_matrix(tr_x, embeddings)
    X_test = corpus_to_matrix(te_x, embeddings)
    print(f"{experiment} 特征矩阵 train={X_train.shape} test={X_test.shape}")

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, tr_y)
    y_pred = clf.predict(X_test)
    return evaluate(te_y, y_pred, experiment=experiment)


def main():
    nyt_texts, nyt_labels = load_nyt()
    splits = split_nyt(nyt_texts, nyt_labels)
    tr_x, _ = splits["train"]

    results = []

    # 预训练 GloVe
    glove_emb = load_glove()
    results.append(run_with_embeddings(splits, glove_emb, "Task2-GloVe"))

    # 在 AG News 语料上训练 Word2Vec（AG 仅提供文本）
    ag_texts = load_ag()
    ag_sentences = [tokenize(t) for t in ag_texts]
    model_ag = train_word2vec(ag_sentences)
    results.append(run_with_embeddings(
        splits, w2v_to_dict(model_ag), "Task2-W2V-AGNews"))

    # 在 NYT 训练集语料上训练 Word2Vec（仅使用训练集文本，避免数据泄漏）
    nyt_sentences = [tokenize(t) for t in tr_x]
    model_nyt = train_word2vec(nyt_sentences)
    results.append(run_with_embeddings(
        splits, w2v_to_dict(model_nyt), "Task2-W2V-NYT"))

    print("\n========== Task 2 ==========")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
