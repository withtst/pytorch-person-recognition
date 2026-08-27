"""
2.5 验证策略与模型保存 - 教学版

上一节我们用 268MB 的 state_dict 存档了 best 模型; 今天学"存档的解剖学":
  1. state_dict 里到底装了什么(逐层的名字和形状 = 模型自组织的地图)
  2. 怎么"复活"它(load_state_dict: 只认结构, 不认历史)
  3. 为什么推荐 state_dict 而不是 torch.save(model)(整只鸡和鸡汤的区别)
  4. 复活后的模型正式亮相: 在验证集上交出 85.2% 的成绩单
"""
import json
import os

import torch

from person_cnn import PersonCNN
from split_dataset import get_datasets

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BEST_FILE = os.path.join("runs", "best_model.pt")


def accuracy(model, loader):
    correct, n = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item()
            n += x.size(0)
    return correct / n


def main():
    torch.manual_seed(42)
    print(f"设备: {DEVICE}")

    # ========================================================================
    # 实验 1: 解剖 state_dict
    # ========================================================================
    print("=" * 64)
    print("【实验 1】best_model.pt 的解剖报告")
    raw = torch.load(BEST_FILE, map_location="cpu")
    print(f"  文件总大小: {os.path.getsize(BEST_FILE) / 1e6:.1f} MB, "
          f"保存的类型: {type(raw).__name__} (就是个 Python 字典!)")
    print(f"  ├─ 共 {len(raw)} 个条目, 每键 = '层名.参数名'(层名能对上 2.1 节)")
    for name, param in list(raw.items())[:4]:
        print(f"  ├─ {name:20s} shape={tuple(param.shape)} dtype={param.dtype}")
    print("  └─ 特点: 只有【数字】——结构信息(几层? 通道数?)不在文件里,")
    print("     复活要靠代码里的模型定义(这就是必须和结构'门当户对'的原因)。")

    # ========================================================================
    # 实验 2: 复活(load)
    # ========================================================================
    print("\n" + "=" * 64)
    print("【实验 2】复活 best 模型并验证")
    model = PersonCNN().to(DEVICE)               # 新造一个"空骨架"
    model.load_state_dict(raw)                   # 把数字灌进去(键名必须一一对上)
    print("  load_state_dict 成功: 若类型/形状对不上, 它会直接抛错(严格匹配)。")
    print("  常见手误: 训练时用 .half() 存的 fp16，加载时模型却是 fp32 —— 对齐即可")

    # 正式亮相: 全套验证指标
    _, val_ds = get_datasets()
    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=32, shuffle=False, num_workers=4,
        pin_memory=(DEVICE == "cuda"))
    acc = accuracy(model, loader)
    print(f"\n  复活模型验证准确率 = {acc:.2%}  (训练时记录的 best = 85.2%: 全对齐!)")

    # ========================================================================
    # 实验 3: 为什么宣传 state_dict 而不是 torch.save(model)
    # ========================================================================
    print("\n" + "=" * 64)
    print("【实验 3】torch.save(model) 的三大硬伤")
    print("  1. save 整个对象 = 把代码定义也打包进去(等于把菜谱和菜一起打包),")
    print("     体积更大(还要 pickle 整个类);")
    print("  2. 加载端必须 import 相同类, 否则 pickle 失败(换文件夹就跑不动);")
    print("  3. 安全风险: pickle 本质可执行代码, 加载来历不明模型 = 执行恶意代码。")
    print("  结论: state_dict(参数 + 结构信任)是生产标准; 模型文件 = 配置 + 权值, 不打包代码。")

    # ========================================================================
    # 实验 4: 逐类体检(3.3 节混淆矩阵的前奏)
    # ========================================================================
    print("\n" + "=" * 64)
    print("【实验 4】逐类准确率(下一节混淆矩阵的探头)")
    per_class = {i: [0, 0] for i in range(11)}        # [对, 总]
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            for t, p in zip(y.tolist(), pred.tolist()):
                per_class[t][1] += 1
                if p == t:
                    per_class[t][0] += 1
    idx_to_class = val_ds.dataset.idx_to_class
    for idx in sorted(per_class):
        correct, total = per_class[idx]
        name = idx_to_class.get(idx, "?")
        print(f"  [{idx:>2}] {name:20s} {correct}/{total} = {correct/total:.1%}")
    print("  观察: 人数少的类(tifa 16 张)准确率波动大 —— 小样本的体检报告总有噪声,")
    print("        下次有类跌到 80% 以下, 别急着改模型, 先看它 val 里才几张(1.4 节)。")


if __name__ == "__main__":
    main()
