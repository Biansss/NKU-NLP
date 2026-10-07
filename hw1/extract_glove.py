"""从已下载的 glove.6B.zip 中只提取 glove.6B.100d.txt（其余维度不需要），随后删除 zip。"""
import os
import shutil
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(HERE, "HW-1", "glove.6B.zip")
TARGET = os.path.join(HERE, "HW-1", "glove.6B.100d.txt")
MEMBER = "glove.6B.100d.txt"

with zipfile.ZipFile(ZIP_PATH) as z:
    print("zip 内文件:", z.namelist())
    with z.open(MEMBER) as src, open(TARGET, "wb") as dst:
        shutil.copyfileobj(src, dst)

print(f"已提取: {TARGET}  ({os.path.getsize(TARGET)/1e6:.1f} MB)")
os.remove(ZIP_PATH)
print("已删除压缩包:", ZIP_PATH)
