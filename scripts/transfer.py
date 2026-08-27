"""
4.1 迁移学习(ResNet18 预训练) - 教学版

前提认知: ImageNet 上 ResNet18 已经看了 128 万张图, 学会了("边缘-纹理-部件-物体"的四级金字塔)。
我们的 4384 张图与 ImageNet 同属"自然图像" —— 底层特征(边缘/轮廓/眼鼻)通用!
于是工程策略: 让 ResNet 的【骨干】当免费的提取器(冻结), 只学【最后一步】:
    把我们要的 11 人从它学到的 1000 类特征空间里分出来。

三笔账(输入/归一化/参数):
  1. 输入: ResNet 默认吃 224x224(不是我们的 256) -> Resize((224,224))
  2. 归一化: 预训练模型"只认"当年训练数据的统计量:
     ImageNet mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
     (1.2 节学过的 Normalize 的 T 用法: 这里的"暗号"必须照抄, 换颜色体系模型就瞎了)
  3. 参数: 骨干卷积全部 requires_grad_(False) -> 不回传梯度(前向照跑);
     optimizer 只注册新头部层(2.3 实验4 的伏笔解密: 优化器只认识它注册的人)。
"""
import json
import os
import time

import torch
import torch.nn as nn
import torchvision.models as M
import torchvision.transforms as T

from split_dataset import get_datasets

EPOCHS = 6
BATCH = 32
LR = 1e-3
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]

HISTORY_OUT = os.path.join("runs", "history_transfer.json")
BEST_OUT = os.path.join("runs", "best_transfer.pt")


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

    # ---- 数据管线: 224 + ImageNet 暗号 ----
    train_transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.Resize((224, 224), interpolation=T.InterpolationMode.LANCZOS),
        T.ToTensor(),
        T.Normalize(IMGNET_MEAN, IMGNET_STD),      # 暗号!
    ])
    val_transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.LANCZOS),
        T.ToTensor(),
        T.Normalize(IMGNET_MEAN, IMGNET_STD),
    ])
    train_ds, val_ds = get_datasets(train_transform=train_transform,
                                    val_transform=val_transform)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # ---- 抬进"前人智慧" ----
    model = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1)
    # 冰封骨干: 7x7 卷积到最后的 conv5 全部不许学(requires_grad=False)
    for p in model.parameters():
        p.requires_grad_(False)
    # 换头: torchvision 的 fc 输出 1000, 换成我们要的 11
    # 新层默认 requires_grad=True, 全网络只有它能学 -> 梯度只出现在此
    model.fc = nn.Linear(model.fc.in_features, 11)
    model = model.to(DEVICE)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"ResNet18 参数: 总 {n_total:,} + 新头 -> 可训练 {n_trainable:,} "
          f"(仅占 {(n_trainable/n_total*100):.1f}%)")
    print(f"对比: 2.4 从头训练的 PersonCNN 67M 全部要学 —— 这就是'站在巨人肩膀上'的预算表")

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    # 优化器: 只注册 fc 的参数。骨干虽在前向跑, 但这里没登记, 梯度再大也不动它。
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_state = -1.0, None
    print(f"开始训练(只练头部 {len(train_ds)} 张 x {EPOCHS} epochs)\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        model.train()
        run_loss, correct, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)          # 只传播到 fc(骨干无梯度=不更新)
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
        print(f"Epoch {epoch}/{EPOCHS} | 训练 {run_loss/n:.4f}/{correct/n:.1%} "
              f"| 验证 {val_loss:.4f}/{val_acc:.1%} | {time.perf_counter()-t0:.0f}s")

    torch.save(best_state, BEST_OUT)
    with open(HISTORY_OUT, "w", encoding="utf-8") as fh:
        json.dump(history, fh, ensure_ascii=False, indent=2)

    print("\n" + "=" * 64)
    print("【决战登记】三条路线的验证 acc 名录")
    print(f"  2.4 从头训练 67M:   best = 85.18% (过拟合, 训练 100%)")
    print(f"  3.2 正则化 67M:     best = 84.61% (16 epochs, 平滑但没赢分)")
    print(f"  4.1 迁移 11M 微小头: best = {best_val_acc:.2%}")


if __name__ == "__main__":
    main()
