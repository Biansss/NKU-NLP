"""环境自检脚本：运行 `python env_check.py`，确认作业所需依赖与 GPU 是否就绪。"""
import importlib

LIBS = ["numpy", "pandas", "sklearn", "scipy", "nltk",
        "gensim", "torch", "transformers", "huggingface_hub"]

print("==== 依赖导入检查 ====")
for m in LIBS:
    try:
        mod = importlib.import_module(m)
        print(f"[OK]   {m:16s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"[FAIL] {m:16s} {type(e).__name__}: {str(e)[:200]}")

print("\n==== PyTorch / CUDA 检查 ====")
try:
    import torch
    print("torch 版本      :", torch.__version__)
    print("torch 编译的CUDA :", torch.version.cuda)
    print("cuda.is_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU             :", torch.cuda.get_device_name(0))
except Exception as e:
    print("[FAIL] torch 检查失败:", e)

print("\n==== NLTK 分词资源检查 ====")
import nltk
for pkg in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{pkg}")
        print(f"[OK]   {pkg}")
    except LookupError:
        print(f"[MISS] {pkg} 未下载")
