"""
1.1 手写 Dataset 类 - 教学版

Dataset 到底是啥? 一句话: 它只是一个"按编号取数据"的抽象接口,
只需要实现两个方法:
    __len__   -> 数据总数 (让 DataLoader 知道要取多少个刻度)
    __getitem__ -> 给定序号 idx, 返回一份样本 (x, y)

就这么简单。本章的每一行代码都在解释"为什么这样做"。
"""
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# 全部图片 97% 是 PNG。PNG 可能带 Alpha(4 通道), 我们后面写的卷积网络只认 3 通道(RGB),
# 因此统一 convert("RGB")。另外这一步也把 jpg/png 的各种细微差异抹平, 保证喂给网络的
# 数据永远是 [0,255] 的 3 通道位图。
RGB_CHANNELS = 3


class PersonDataset(Dataset):
    """
    11 人物识别数据集。

    设计决策 1: 文件清单(路径+标签)装在哪里?
      放 __init__ 里一次性扫描。4 千多张图, listdir + 排序只要几十毫秒。
      如果放在 __getitem__ 里每取一张扫一次盘, 训练一个 epoch 要扫描 4384 次,
      时间都花在文件系统上了 —— 记住: 大数据的第一戒律是"别重复扫描"(0.1 节已踩过)。

    设计决策 2: 类别编号为什么这么定?
      先 sorted() 再编号, 即 class_to_idx = {"1.洛瑶":0, "10.伤心肉松小贝":1, ...}。
      编号顺序与文件系统无关, 任何人 clone 代码跑出来的 label 都一致 -> 可复现。
      这里 index 不需要手写, 用 sorted(names) 的 enumerate 一行搞定。

    设计决策 3: 为什么统一缩放到 img_size?
      神经网络的全连接层要求固定输入维度; 卷积层虽不挑尺寸, 但 DataLoader 打包 batch
      时要求每张图形状完全一致, 否则 torch.stack 直接报错。
      尺寸选多少? 一张 256x256 RGB 图占 GPU 显存 = 256*256*3*4 字节 ≈ 786 KB,
      batch=64 时约为 50 MB(仅输入); 若选 2048: 一张就 50 MB, 一个 batch 3.2 GB,
      还没算模型显存, RTX 5070 (12 GB) 也吃不消。256 是"能看清人脸 + 显存友好"的平衡点。

    设计决策 4: 为什么接口是固定的 (path, label) 元组?
      __getitem__ 返回什么格式, 网络就收到什么。我们返回 (tensor_x, int_label) 别的不带,
      保持纯净; 以后想加"人物名", 可以直接从 class_to_idx 反查, 不必塞进返回值。
    """
    def __init__(self, data_dir="./data", img_size=256, transform=None):
        self.data_dir = data_dir
        self.img_size = img_size
        # transform: 预留的钩子。第 1.2 节会传 torchvision.transforms.Compose 进来;
        # 传了就用外部的(它是"政策"), 没传就用内部默认管线(它是"保底")。
        # 这种"参数可注入"的写法是框架扩展开口的教科书样例。
        self.transform = transform

        # ---- 一次扫描: 建类表 + 文件清单 ----
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"数据集目录不存在: {data_dir}")

        # 每个子文件夹是一个人。sorted() 保证类别编号稳定。
        persons = sorted(p for p in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, p)))
        self.class_to_idx = {person: i for i, person in enumerate(persons)}

        # 倒查表: 0 -> "1.洛瑶"。调试/推理时把模型输出(编号)翻译回人名, 就是靠它。
        self.idx_to_class = {i: p for p, i in self.class_to_idx.items()}

        # 文件清单: [(完整路径, 类别编号), ...]
        # 这里只按扩展名过滤, 不打开图片 —— 打开校验是 0.1 节职责, 这里信数据已被清洗过。
        self.images = []
        for person in persons:
            person_dir = os.path.join(data_dir, person)
            for fname in sorted(os.listdir(person_dir)):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                    continue  # 跳过非图片(理论上已被 clean_data.py 隔离, 这行是双保险)
                self.images.append((os.path.join(person_dir, fname), self.class_to_idx[person]))

        print(f"PersonDataset 就绪: {len(self.class_to_idx)} 人, {len(self.images)} 张, 输出尺寸 {img_size}x{img_size}")

    def __len__(self):
        # DataLoader 用 len() 算一个 epoch 的步数: steps = ceil(len / batch_size)
        return len(self.images)

    def __getitem__(self, idx):
        path, label = self.images[idx]

        # ---- 读图(三步, 一步也不能省) ----
        img = Image.open(path)            # 1. PIL 打开(懒加载, 只读文件头)
        # 2. 降噪: 统一转 RGB, 吸掉 Alpha, PIL 会做后台合成。
        #    注意 convert 返回"新图", 要接住返回值, 别写 img.convert("RGB") 不接。
        img = img.convert("RGB")

        # 3. 统一尺寸。
        #    Resampling.LANCZOS 是高质量下采样滤镜(抗锯齿), 比默认的 NEAREST 更平滑,
        #    对"缩小→保留人脸特征"最重要; 放大场景其实无中生有像素, 但这里只是归一化操作。
        img = img.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)

        if self.transform is not None:
            return self.transform(img), label   # 有外部管线(1.2 节)就走它

        # ---- 默认管线: PIL 图 -> 网络输入的 tensor ----
        # 走 numpy 中转但有零拷贝技巧: from_numpy 共享内存, 不复制(代价是都要是 numpy 数组)。
        arr = np.array(img)               # (H, W, 3) uint8, 值域 [0, 255]
        x = torch.from_numpy(arr).permute(2, 0, 1)  # (H,W,C) -> (C,H,W): 网络的约定排序
        # from_numpy 是"零拷贝 view", 保持 uint8; 我们手动转 float + 归一化
        x = x.to(torch.float32) / 255.0    # [0,255] -> [0,1] float32
        # 为什么 float32 不 float64: GPU 上 32 位是标准, 精度足够且省一半显存/带宽
        return x, label


if __name__ == "__main__":
    # ---- 自测段: 直接运行 python scripts/person_dataset.py ----
    ds = PersonDataset()
    print(f"\n类别映射表 ({len(ds.class_to_idx)} 类):")
    for name, idx in sorted(ds.class_to_idx.items(), key=lambda kv: kv[1]):
        print(f"  [{idx}] -> {name}")

    print(f"\nlen(dataset) = {len(ds)}")
    x, y = ds[0]
    print(f"单个样本: x.shape={x.shape}, x.dtype={x.dtype}, y={y} (类型 {type(y).__name__})")

    # 数据统计: 每类先看一眼第一张是否真的不同(粗验)
    assert x.shape == (3, 256, 256), "输出张量形状不对"
    print(f"\n自测通过: 第一张图像素范围 [{x.min():.3f}, {x.max():.3f}] (应在 0~1 内)")
