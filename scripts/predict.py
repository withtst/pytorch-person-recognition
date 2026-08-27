"""
5.1 推理 CLI - 教学版

终局: 给一张图 -> 输出"她是 X(92%)"。

工程要点(全部是前几节埋的地雷回踩):
  1. 结构一致: 训练时是 resnet18+fc(11), 推理时也必须同样拼装(2.5 节 state_dict 的契约)
  2. 预处理一致: 224 + ImageNet 归一化(4.1 节"暗号"); 忘了暗号 = 模型瞬间失明,
     准确率掉回随机(测试: 把 Normalize 拿掉试试看惨状?)
  3. eval() + no_grad: 没 Dropout/BN 的推理行为了(虽此模型无 BN, 纪律常在)
  4. load_state_dict 时 map_location 告诉它"从 cuda 缓存搬到哪", 无 GPU 也能加载

用法:
  python scripts/predict.py --image data/9.媛媛/xxx.png
  python scripts/predict.py --demo 5      # 抽查 5 张训练集图片(假装它们没被见过)
"""
import argparse
import os
import random
import sys

import torch
import torch.nn as nn
import torchvision.models as M
import torchvision.transforms as T
from PIL import Image

from person_dataset import PersonDataset

BEST_FILE = os.path.join("runs", "best_transfer.pt")
IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model(checkpoint: dict) -> nn.Module:
    """
    拼装与训练完全同构的模型: ResNet18 骨干(用权重时直接 load 覆盖) + 11 类头。
    state_dict 保真契约(2.5 节): 键名形状全对上, load 成功才算"复活成功"。
    """
    model = M.resnet18(weights=None)          # 无预训练权重, 纯骨架
    model.fc = nn.Linear(model.fc.in_features, 11)
    state = model.state_dict()
    missing = [k for k in state.keys() if k not in checkpoint]
    if missing:
        print(f"警告: checkpoint 缺少 {len(missing)} 个键 (结构不一致!)")
    model.load_state_dict(checkpoint)
    return model


def transform_pipeline() -> T.Compose:
    """与 4.1 训练/验证完全一致的预处理(暗号必须对齐)."""
    return T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.LANCZOS),
        T.ToTensor(),
        T.Normalize(IMGNET_MEAN, IMGNET_STD),
    ])


def infer(model, path_or_tensor, idx_to_class, topk=3):
    """单图推理 -> [(人名, 概率), ...] 前 topk."""
    t = transform_pipeline()(Image.open(path_or_tensor).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(t)[0], dim=0)
    vals, ids = probs.topk(topk)
    return [(idx_to_class[i.item()], v.item()) for i, v in zip(ids, vals)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="单张图片路径")
    ap.add_argument("--demo", type=int, default=0, metavar="N", help="抽查 N 张训练集图片")
    args = ap.parse_args()

    torch.manual_seed(42)
    ds = PersonDataset()
    model = build_model(torch.load(BEST_FILE, map_location="cpu")).to(DEVICE)
    model.eval()
    print(f"模型就绪: {BEST_FILE}  设备 {DEVICE}\n")

    if args.image:
        name, prob = infer(model, args.image, ds.idx_to_class)[0]
        print(f"判读: 这张图更像是 [ {name} ] (置信度 {prob:.1%})")
        for i, (nm, p) in enumerate(infer(model, args.image, ds.idx_to_class)):
            print(f"  Top{i+1}: {nm:14s} {p:.1%}")

    if args.demo > 0:
        print(f"\n抽查 {args.demo} 张(演示性质, 图片来自训练集):")
        samples = random.sample(range(len(ds)), min(args.demo, len(ds)))
        for i in samples:
            path, label = ds.images[i]
            top1, p1 = infer(model, path, ds.idx_to_class)[0]
            truth = ds.idx_to_class[label]
            mark = "OK" if top1 == truth else "XX"
            print(f"  {mark} 真: {truth:14s} -> 判: {top1:14s} {p1:.1%}  ({os.path.basename(path)})")


if __name__ == "__main__":
    main()
