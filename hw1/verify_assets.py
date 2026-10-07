"""资产端到端验证：检查数据/GloVe/本地BERT是否齐备，并在 GPU 上真实跑一次 BERT 前向。"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "HW-1")

print("==== 1. 数据文件 ====")
for f in ["nyt.csv", "ag.csv", "glove.6B.100d.txt",
          os.path.join("bert-base-uncased", "model.safetensors")]:
    p = os.path.join(DATA, f)
    print(f"[{'OK' if os.path.exists(p) else 'MISS'}] {f}")

print("\n==== 2. GloVe 格式校验（每行应为 1 个词 + 100 个数）====")
with open(os.path.join(DATA, "glove.6B.100d.txt"), encoding="utf8") as f:
    for i in range(3):
        parts = f.readline().rstrip().split(" ")
        print(f"  词={parts[0]:<10} 字段数={len(parts)}  首维={parts[1]}")
        assert len(parts) == 101, "字段数不是 101，GloVe 维度不对！"
print("GloVe 100 维格式正确")

print("\n==== 3. 本地 BERT 离线加载 + GPU 前向 ====")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

mp = os.path.join(DATA, "bert-base-uncased")
tok = AutoTokenizer.from_pretrained(mp)
model = AutoModelForSequenceClassification.from_pretrained(mp, num_labels=3)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

texts = ["the team won the football match",
         "stocks rose after the quarterly earnings report"]
enc = tok(texts, padding=True, truncation=True, max_length=64,
          return_tensors="pt").to(device)
with torch.no_grad():
    out = model(**enc)
print("logits 形状 =", tuple(out.logits.shape), " 所在设备 =", out.logits.device)
assert tuple(out.logits.shape) == (2, 3)
print("\n全部资产验证通过：BERT 可在", device.upper(), "上离线运行。")
