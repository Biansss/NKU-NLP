"""
Task 3：微调预训练语言模型 BERT 进行文本分类。

模型为 bert-base-uncased，最大序列长度 64，训练 3 个 epoch，
在 NYT 测试集上报告 Accuracy 与 Macro-F1。微调基于 transformers
的 Trainer 完成。
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from data_utils import load_nyt, split_nyt, make_label_map, DATA_DIR
from eval_utils import evaluate

# 优先加载随项目下载的本地模型目录，缺失时回退到 Hugging Face 在线仓库
_LOCAL_BERT = os.path.join(DATA_DIR, "bert-base-uncased")
MODEL_NAME = _LOCAL_BERT if os.path.isdir(_LOCAL_BERT) \
    else "google-bert/bert-base-uncased"

MAX_LENGTH = 64
EPOCHS = 3
TRAIN_BATCH = 16
EVAL_BATCH = 32


def tokenize_texts(texts, tokenizer):
    """使用 BERT 分词器将文本编码为 input_ids 与 attention_mask。

    统一截断、填充到 max_length=MAX_LENGTH，超长文本在该长度处截断。
    返回可按样本索引的批量编码（不预先转为张量，由 NewsDataset 转换）。
    """
    raise NotImplementedError


class NewsDataset(Dataset):
    """封装分词结果与标签，供 Trainer 使用。"""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.as_tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.as_tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    """计算评估阶段的 Accuracy 与 Macro-F1，作为 Trainer 的指标回调。"""
    raise NotImplementedError


def main():
    texts, labels = load_nyt()
    splits = split_nyt(texts, labels)
    tr_x, tr_y = splits["train"]
    va_x, va_y = splits["val"]
    te_x, te_y = splits["test"]

    label2id, id2label = make_label_map(tr_y)
    to_id = lambda ys: [label2id[y] for y in ys]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = NewsDataset(tokenize_texts(tr_x, tokenizer), to_id(tr_y))
    val_ds = NewsDataset(tokenize_texts(va_x, tokenizer), to_id(va_y))
    test_ds = NewsDataset(tokenize_texts(te_x, tokenizer), to_id(te_y))

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir="./results/bert",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=100,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    out = trainer.predict(test_ds)
    pred_ids = np.argmax(out.predictions, axis=1)
    pred_labels = [id2label[i] for i in pred_ids]
    result = evaluate(te_y, pred_labels, experiment="Task3-BERT")
    print("\n========== Task 3 ==========")
    print(result)


if __name__ == "__main__":
    main()
