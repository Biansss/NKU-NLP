"""
预下载 bert-base-uncased 到项目本地目录（走 hf-mirror 国内镜像）。
下载完成后，task3_bert.py 里的 MODEL_NAME 指向本地目录即可完全离线运行。
"""
import os

# 必须在 import huggingface_hub 之前设置镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download

TARGET = r"D:\aaadep\grade_4\自然语言\hw1\HW-1\bert-base-uncased"

# 只下载 PyTorch(safetensors) 推理/微调必需文件，跳过 tf/flax/大 bin 以省空间
PATTERNS = [
    "config.json",
    "vocab.txt",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "model.safetensors",
]

if __name__ == "__main__":
    path = snapshot_download(
        repo_id="google-bert/bert-base-uncased",
        local_dir=TARGET,
        allow_patterns=PATTERNS,
        max_workers=4,
    )
    print("模型已下载到:", path)
    for f in sorted(os.listdir(path)):
        size = os.path.getsize(os.path.join(path, f))
        print(f"  {f}  ({size/1e6:.1f} MB)" if size > 1e6 else f"  {f}")
