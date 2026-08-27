"""
3.2 正则化与数据增强 - 教学版(迎战过拟合)

装备清单(对照 3.1 诊断书):
  武器1: 真实统计量 Normalize —— 1.2 节伏笔兑现: 0.5/0.5 是"拍脑袋",
         实测通道均值≈[0.60, 0.55, 0.51], 标准差≈0.29; 用真实值才叫归一化。
  武器2: 数据增强 —— RandomHorizontalFlip+ColorJitter:
         让模型每轮看到的"同一个人照片"都不一样(右脸/左脸/调色/曝光差异)。
  武器3: weight_decay —— 梯度下降顺手刮一层"L2 剪影", 压制超大权重。
  武器4: Dropout(0.3) —— 2.1 节加了开关; 全连接前随机熄火 30% 神经元,
         逼网络"多路并行备份", 不能靠单个神经元硬背答案。
对照组: 2.4 节(同一结构/8 epochs/无增强/无正则)的 history.json + best 存档。
"""
import json
import os
import time

import torch
import torch.nn as nn
import torchvision.transforms as T

from person_cnn import PersonCNN
from split_dataset import get_datasets

EPOCHS = 16                      # 3.2 铁证: 带正则化的模型收敛更慢, 8 轮学不完,
                                 # 加深到 16 轮再看真实增益(学费已付, 见输出)
BATCH = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4                  # 武器3: Adam 的 L2 剪裁系数(常用 1e-2~1e-5)
DROPOUT = 0.3                        # 武器4
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

HISTORY_OUT = os.path.join("runs", "history_aug.json")
BEST_OUT = os.path.join("runs", "best_aug.pt")
BASELINE_HISTORY = os.path.join("runs", "history.json")


def compute_real_stats(dataset, n=600):
    """武器0: 用数据集的真实通道 mean/std 替换 0.5 猜猜猜(1.2 节欠的账)."""
    import random
    idxs = random.sample(range(len(dataset)), n)
    stack = torch.stack([dataset[i][0] for i in idxs])      # (n,3,256,256)
    mean = stack.mean(dim=(0, 2, 3))
    std = stack.std(dim=(0, 2, 3))
    return mean.tolist(), std.tolist()


def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = loss_fn(model(x), y)
            total_loss += loss.item() * x.size(0)
            correct += (model(x).argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def main():
    torch.manual_seed(SEED)
    print(f"设备: {DEVICE}\n")

    # ---- 武器 1: 真实统计量 ----
    ds_train_plain, _ = get_datasets()            # 拿全量做统计(仅索引, 不花解码)
    ds_full = ds_train_plain.dataset              # Subset -> 内层 PersonDataset
    mean, std = compute_real_stats(ds_full, n=600)
    print(f"【武器1】真实通道统计量: mean={['%.3f' % m for m in mean]}, "
          f"std={['%.3f' % s for s in std]}")
    print("        对比 1.2 节拍脑袋的 0.50/0.50 —— 数据真相 != 设计假设\n")

    # ---- 管线组装: 训练戴"增援腕带", 验证拿"纯净报告" ----
    train_transform = T.Compose([
        # 武器2-1: 水平翻转(人偶美学: 左右脸对这种人脸任务几乎等价 -> 免费 2x 数据)
        T.RandomHorizontalFlip(p=0.5),
        # 武器2-2: 色板颤抖(亮度/对比/饱和度 ±30%), 模拟同一照片的不同后期调色
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        T.Resize((256, 256), interpolation=T.InterpolationMode.LANCZOS),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    val_transform = T.Compose([
        T.Resize((256, 256), interpolation=T.InterpolationMode.LANCZOS),
        T.ToTensor(),
        T.Normalize(mean, std),                # 归一化参数 train/val 必须一致!
    ])

    train_ds, val_ds = get_datasets(
        train_transform=train_transform, val_transform=val_transform)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True)

    # ---- 武器 3+4 装配 ----
    model = PersonCNN(dropout=DROPOUT).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # weight_decay 何德何能? 数学: loss += wd/2 * ||w||^2 -> 梯度多一项 wd*w,
    # 等于每次更新都拽着权重往 0 缩 —— 大权重本可以为你一人"应援", 现在谁也不许独大。

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_state = -1.0, None
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,} "
          f"(含 Dropout 开关, 参数量不变)\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        model.train()
        run_loss, correct, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * x.size(0)
            correct += (model(x).argmax(1) == y).sum().item()
            n += x.size(0)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn)

        history["train_loss"].append(round(run_loss / n, 4))
        history["train_acc"].append(round(correct / n, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch:2d} | 训练 {run_loss/n:.4f}/{correct/n:.1%} "
              f"| 验证 {val_loss:.4f}/{val_acc:.1%} | {time.perf_counter()-t0:.0f}s")

    torch.save(best_state, BEST_OUT)
    with open(HISTORY_OUT, "w", encoding="utf-8") as fh:
        json.dump(history, fh, ensure_ascii=False, indent=2)

    # ---- 与 2.4 基准对决 ----
    print("\n" + "=" * 64)
    print("【战后复盘】2.4 基准 vs 3.2 增强(同一结构, 8 epochs)")
    with open(BASELINE_HISTORY, encoding="utf-8") as fh:
        base = json.load(fh)
    base_best = max(base["val_acc"])
    my_best = max(history["val_acc"])
    print(f"  基准(2.4): best 验证 acc = {base_best:.2%} (拐点在 Epoch 7)")
    print(f"  本轮(3.2): best 验证 acc = {my_best:.2%} "
          f"(增益 {'+' if my_best >= base_best else '-'}{abs(my_best-base_best)*100:.1f} pp)")
    print(f"  末轮鸿沟: 训练 {history['train_acc'][-1]:.1%} vs 验证 {history['val_acc'][-1]:.1%} "
          f"(基准末期 15.4 pp)")
    print("  记忆之针: 同一套武器也可以帮到上一轮 48% 的雪蕊呀 —— 3.3 节混淆矩阵见分晓。")


if __name__ == "__main__":
    main()
