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
│   └── bert-base-uncased/      #   本地 BERT 模型（可选，缺失则自动在线下载）
├── data_utils.py               # 数据加载、固定划分、分词、标签映射
├── eval_utils.py               # Accuracy / Macro-F1
├── task1_bow.py                # Task 1：词袋 + Logistic Regression
├── task2_word2vec.py           # Task 2：GloVe / Word2Vec + Logistic Regression
├── task3_bert.py               # Task 3：BERT 微调
├── requirements.txt            # 依赖清单
└── README.md                   # 运行说明
```

## 环境配置

推荐 Python 3.11，依赖见 `requirements.txt`：

```bash
conda create -n nlp-hw1 python=3.11 -y
conda activate nlp-hw1
pip install -r requirements.txt
```

实测版本：torch 2.14.1+cu126（GPU）、transformers 5.19.0、accelerate 1.15.0、
scikit-learn 1.9.1、gensim 4.4.0、nltk 3.10.3、pandas 3.0.6、numpy 2.4.6。

> 注意：PyPI 默认安装的 torch 为 CPU 版本；如需 GPU 版，请按
> <https://pytorch.org/get-started/locally/> 选择对应 CUDA 的 index-url 安装。
> transformers v5 中训练参数与分词器参数分别为 `eval_strategy`、
> `processing_class`，代码已据此编写。

## 数据与预训练模型准备

`HW-1/` 目录体积较大、不纳入版本管理，请按下列方式自行准备：

1. 将 `nyt.csv`、`ag.csv` 放入 `HW-1/`。
2. **GloVe**：从 <http://nlp.stanford.edu/data/glove.6B.zip> 下载压缩包，
   解压后把 `glove.6B.100d.txt` 放入 `HW-1/`（本作业只使用 100 维这一份）。
3. **BERT**：`task3_bert.py` 默认优先加载本地目录 `HW-1/bert-base-uncased/`；
   若该目录不存在，则自动从 Hugging Face 下载 `google-bert/bert-base-uncased`。
   国内网络可先设置镜像环境变量：
   ```powershell
   $env:HF_ENDPOINT = "https://hf-mirror.com"
   $env:HF_HUB_DISABLE_XET = "1"
   ```
   如需预先下载到本地，可执行：
   ```bash
   huggingface-cli download google-bert/bert-base-uncased \
       --local-dir HW-1/bert-base-uncased
   ```

## 运行方式

```bash
conda activate nlp-hw1

python data_utils.py       # 数据读取与划分自检（train≈9215 / val≈1152 / test≈1152）
python task1_bow.py        # Task 1
python task2_word2vec.py   # Task 2
python task3_bert.py       # Task 3
```

## 实验设置

- NYT 随机打乱后以固定随机种子（`RANDOM_SEED=42`）按 80% / 10% / 10% 划分，
  三个 Task 复用同一划分，结果统一在测试集上报告。
- 词向量统一为 **100 维**，文档向量为有效词向量的平均；Task 1、2 分类器均为
  Logistic Regression。
- BERT 使用 `bert-base-uncased`，最大序列长度 64，训练 3 个 epoch，
  训练 batch size 16；GPU 上自动启用半精度（fp16）。

## 实验结果

| 实验 | Accuracy | Macro-F1 |
|---|---|---|
| Task 1 — Binary BoW + LR | 0.9826 | 0.9576 |
| Task 1 — Word Frequency + LR | 0.9826 | 0.9621 |
| Task 2 — GloVe 平均向量 + LR | 0.9774 | 0.9468 |
| Task 2 — AG News 自训 Word2Vec + LR | 0.9722 | 0.9358 |
| Task 2 — NYT 自训 Word2Vec + LR | 0.9731 | 0.9377 |
| Task 3 — BERT fine-tune | 0.9792 | 0.9582 |

## 结果分析

- **词袋整体略优于平均词向量**：平均词向量把整篇文档压缩为单个 100 维向量，
  丢失了词频与区分性词汇信息，因此 Task 2 略低于 Task 1。
- **Task 2 三组对比**：预训练 GloVe 效果最好（语料规模大、词覆盖全）；
  在 NYT 训练集上自训的 Word2Vec 略优于在 AG News 上自训，说明同领域语料
  学到的词向量对目标任务更有针对性（AG News 语料虽大但领域不同）。
- **BERT 明显优于 Task 2 的平均词向量**（Macro-F1 0.9582 对 0.94 上下），
  体现了上下文相关表示与微调的优势；但并未超过使用全文的词袋
  （词频 BoW Macro-F1 0.9621）。
- **BERT 未超越词袋的原因**：本任务为主题分类、类别词汇区分度高，词袋 + LR
  已接近性能天花板；更关键的是按要求 `max_length=64` 截断，而 NYT 文档平均
  约 637 词，BERT 实际只看到文章开头，丢失大量主体内容，而 Task 1/2 使用了
  全文。验证集在第 2 个 epoch 达到峰值（Accuracy 0.9852 / Macro-F1 0.9663），
  第 3 个 epoch 略有回落，存在轻微过拟合迹象。
- NYT 类别分布不均衡（sports 占多数），Accuracy 整体偏高，应以 Macro-F1
  综合衡量模型在 business / politics 等小类别上的表现。

## 备注

- 加载 BERT 时出现 `UNEXPECTED ... cls.predictions / cls.seq_relationship` 与
  `MISSING classifier.weight / classifier.bias` 属于正常现象：前者为未使用的
  预训练 MLM/NSP 头，后者为下游分类任务随机初始化、需要微调的分类头。
