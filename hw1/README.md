# 作业一：文本分类实现

分别采用 **Bag-of-Words**、**Word2Vec / GloVe 词向量** 与 **预训练语言模型 BERT**
三种文本表示，在 NYT 新闻数据集上完成三分类（sports / business / politics），
并以 **Accuracy** 与 **Macro-F1** 统一评价、对比不同表示方法的效果。

## 任务概述

| Task | 文本表示 | 分类器 | 实验设置 |
|---|---|---|---|
| Task 1 | Bag-of-Words | Logistic Regression | Binary BoW、Word Frequency 两种 |
| Task 2 | 100 维词向量，文档向量取平均 | Logistic Regression | 预训练 GloVe、AG News 自训 Word2Vec、NYT 自训 Word2Vec |
| Task 3 | bert-base-uncased（微调） | 模型自带分类头 | `max_length=64`，`3 epochs` |

## 数据集

- **NYT**（`nyt.csv`，列为 `text, label`）：主数据集，共 11,519 条，
  含 sports / business / politics 三类。随机打乱后按 **80% / 10% / 10%**
  划分为训练集、验证集、测试集，所有实验共用同一划分，最终在测试集上评价。
- **AG News**（`ag.csv`，列为 `text`）：共 90,000 条，仅用于训练 Word2Vec 词向量。

## 目录结构

```
.
├── HW-1/                       # 数据与预训练模型（体积较大，不纳入 git，需自行准备）
│   ├── nyt.csv                 #   NYT 主数据集
│   ├── ag.csv                  #   AG News，用于训练 Word2Vec
│   ├── glove.6B.100d.txt       #   GloVe 100 维词向量
│   └── bert-base-uncased/      #   本地 BERT 模型（可离线加载）
├── data_utils.py               # 数据加载、固定划分、分词、标签映射
├── eval_utils.py               # Accuracy / Macro-F1
├── task1_bow.py                # Task 1：词袋 + Logistic Regression
├── task2_word2vec.py           # Task 2：GloVe / Word2Vec + Logistic Regression
├── task3_bert.py               # Task 3：BERT 微调
├── env_check.py                # 依赖 / GPU / NLTK 资源自检
├── smoke_test.py               # GPU、gensim 与 transformers 兼容性冒烟测试
├── verify_assets.py            # 数据、GloVe 与本地 BERT 可用性校验
├── download_bert.py            # 下载 bert-base-uncased 到本地
├── extract_glove.py            # 从 glove.6B.zip 提取 100 维向量
├── requirements.txt
└── README.md
```

## 环境配置

推荐 Python 3.11，依赖见 `requirements.txt`：

```bash
conda create -n nlp-hw1 python=3.11 -y
conda activate nlp-hw1
pip install -r requirements.txt
```

实测版本：torch 2.14.1+cu126（GPU）、transformers 5.19.0、scikit-learn 1.9.1、
gensim 4.4.0、nltk 3.10.3、accelerate 1.15.0。

> 注意：PyPI 默认安装的 torch 为 CPU 版本；如需 GPU 版，请按
> <https://pytorch.org/get-started/locally/> 选择对应 CUDA 的 index-url 安装。
> transformers v5 中训练参数与分词器参数分别为 `eval_strategy`、
> `processing_class`，代码已据此编写。

## 数据与预训练模型准备

`HW-1/` 目录不纳入版本管理，需按下列方式准备：

1. 将 `nyt.csv`、`ag.csv` 放入 `HW-1/`。
2. GloVe：下载 <http://nlp.stanford.edu/data/glove.6B.zip> 放入 `HW-1/`，
   运行 `python extract_glove.py`，仅提取所需的 `glove.6B.100d.txt`。
3. BERT：运行 `python download_bert.py` 下载到 `HW-1/bert-base-uncased/`。
   国内网络可先设置镜像：
   ```powershell
   $env:HF_ENDPOINT = "https://hf-mirror.com"
   $env:HF_HUB_DISABLE_XET = "1"
   ```
   `task3_bert.py` 会优先离线加载该本地目录，缺失时才回退到在线仓库。

## 运行方式

```bash
conda activate nlp-hw1

python env_check.py       # 检查依赖、GPU 与 NLTK 资源
python verify_assets.py   # 检查数据、GloVe 与本地 BERT
python data_utils.py      # 检查数据读取与划分（train≈9215 / val≈1152 / test≈1152）
python task1_bow.py       # Task 1
python task2_word2vec.py  # Task 2
python task3_bert.py      # Task 3
```

## 实验设置

- NYT 随机打乱后以固定随机种子（`RANDOM_SEED=42`）按 80% / 10% / 10% 划分，
  三个 Task 复用同一划分，结果统一在测试集上报告。
- 词向量统一为 **100 维**，文档向量为有效词向量的平均；Task 1、2 分类器均为
  Logistic Regression。
- BERT 使用 `bert-base-uncased`，最大序列长度 64，训练 3 个 epoch。

## 实验结果

| 实验 | Accuracy | Macro-F1 |
|---|---|---|
| Task 1 — Binary BoW + LR | 0.9826 | 0.9576 |
| Task 1 — Word Frequency + LR | 0.9826 | 0.9621 |
| Task 2 — GloVe 平均向量 + LR | - | - |
| Task 2 — AG News 自训 Word2Vec + LR | - | - |
| Task 2 — NYT 自训 Word2Vec + LR | - | - |
| Task 3 — BERT fine-tune | - | - |

## 备注

- 加载 BERT 时出现 `UNEXPECTED ... cls.predictions / cls.seq_relationship` 与
  `MISSING classifier.weight / classifier.bias` 属于正常现象：前者为未使用的
  预训练 MLM/NSP 头，后者为下游分类任务随机初始化、需要微调的分类头。
- NYT 类别分布不均衡（sports 占多数），因此 Accuracy 可能偏高，
  应以 Macro-F1 综合衡量各类别表现。

## 参考

- Pennington, J., Socher, R., & Manning, C. D. (2014). *GloVe: Global Vectors for Word Representation.*
- Mikolov, T., et al. (2013). *Efficient Estimation of Word Representations in Vector Space.*
- Devlin, J., et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.*
