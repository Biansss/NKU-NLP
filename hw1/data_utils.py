"""
数据加载、数据集划分与分词等公共工具。

所有实验均通过本模块获取同一份 train/validation/test 划分，
以保证不同文本表示方法之间的结果可比、可复现。
"""

import os
from typing import Dict, List, Tuple

# 数据与预训练文件均位于代码目录下的 HW-1/
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "HW-1")

NYT_CSV = os.path.join(DATA_DIR, "nyt.csv")
AG_CSV = os.path.join(DATA_DIR, "ag.csv")
GLOVE_TXT = os.path.join(DATA_DIR, "glove.6B.100d.txt")

# 固定随机种子，保证数据划分可复现
RANDOM_SEED = 42

TEST_RATIO = 0.1   # 测试集 10%
VAL_RATIO = 0.1    # 验证集 10%（其余 80% 为训练集）

# 划分集合：名称 -> (文本列表, 标签列表)
Split = Dict[str, Tuple[List[str], List[str]]]


def ensure_nltk_data() -> None:
    """确保 NLTK 分词所需的 punkt / punkt_tab 资源可用，缺失时自动下载。"""
    import nltk
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(os.path.join("tokenizers", pkg))
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception as exc:
                print(f"[warn] 下载 nltk 资源 {pkg} 失败: {exc}；"
                      f"请手动执行 nltk.download('{pkg}')")


def load_nyt(path: str = NYT_CSV) -> Tuple[List[str], List[str]]:
    """读取 NYT 数据集。

    返回:
        (texts, labels)：两个等长列表，texts[i] 为第 i 条新闻文本，
        labels[i] 为其类别字符串。
    """
    raise NotImplementedError


def load_ag(path: str = AG_CSV) -> List[str]:
    """读取 AG News 数据集，仅返回文本列（用于训练 Word2Vec）。"""
    raise NotImplementedError


def split_nyt(texts: List[str], labels: List[str],
              seed: int = RANDOM_SEED) -> Split:
    """随机打乱并按 80%/10%/10% 划分训练集、验证集、测试集。

    使用固定随机种子保证划分可复现；划分时文本与标签需同步，避免错位。
    可采用分层抽样以使各集合的类别比例与整体一致。

    返回:
        含 "train"/"val"/"test" 三个键的字典，每个键对应 (texts, labels)。
    """
    raise NotImplementedError


def tokenize(text: str) -> List[str]:
    """英文分词并统一转为小写。

    用于词袋与 Word2Vec（GloVe 词表为全小写，统一小写可减少未登录词并
    保证各方法预处理口径一致）。BERT 使用其自带分词器，不调用本函数。
    """
    raise NotImplementedError


def make_label_map(labels: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """根据标签集合生成类别字符串与整数 id 之间的双向映射。"""
    classes = sorted(set(labels))
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for c, i in label2id.items()}
    return label2id, id2label


if __name__ == "__main__":
    ensure_nltk_data()
    texts, labels = load_nyt()
    print("NYT 总条数:", len(texts))

    splits = split_nyt(texts, labels)
    for name in ("train", "val", "test"):
        x, y = splits[name]
        print(f"{name:5s} 条数={len(x):5d}  示例标签={y[0]}")

    print("分词示例:", tokenize(texts[0])[:10])
