"""深度冒烟测试：GPU 实算、gensim 训练、transformers 5.x API 核对。"""
import inspect

print("==== 1. PyTorch GPU 实算 ====")
import torch
if torch.cuda.is_available():
    a = torch.randn(1024, 1024, device="cuda")
    b = torch.randn(1024, 1024, device="cuda")
    c = (a @ b).sum().item()
    print("GPU 矩阵乘法成功, 结果标量 =", round(c, 2), " 设备 =", a.device)
else:
    print("CUDA 不可用！")

print("\n==== 2. gensim Word2Vec 迷你训练（验证 numpy2 下编译扩展可用）====")
from gensim.models import Word2Vec
sents = [["apple", "banana", "fruit"], ["apple", "pie", "dessert"],
         ["banana", "split", "dessert"], ["fruit", "salad", "apple"]] * 20
m = Word2Vec(sentences=sents, vector_size=100, min_count=1, window=3, epochs=5)
print("Word2Vec 训练成功, 'apple' 向量 shape =", m.wv["apple"].shape)

print("\n==== 3. accelerate（Trainer 依赖）====")
try:
    import accelerate
    print("[OK] accelerate", accelerate.__version__)
except Exception as e:
    print("[MISS] 未安装 accelerate：", e)

print("\n==== 4. transformers 5.x 关键参数名核对 ====")
from transformers import TrainingArguments, Trainer
ta_params = set(inspect.signature(TrainingArguments.__init__).parameters)
tr_params = set(inspect.signature(Trainer.__init__).parameters)
print("TrainingArguments 含 eval_strategy       :", "eval_strategy" in ta_params)
print("TrainingArguments 含 evaluation_strategy :", "evaluation_strategy" in ta_params)
print("Trainer 含 processing_class              :", "processing_class" in tr_params)
print("Trainer 含 tokenizer(旧名)               :", "tokenizer" in tr_params)
import transformers
print("transformers 版本:", transformers.__version__)
