"""
2.4 完整训练循环 - 教学版

今天总装: dataset(1.1-1.4) + model(2.1) + loss(2.2) + optimizer(2.3) = 训练循环

一次 epoch 的完整世界:
    for batch in train_loader:
        x, y  ->  GPU
        out = model(x)                # 前向
        loss = loss_fn(out, y)        # 质检
        loss.backward()               # 逆向传播(2.3 学过的梯度累积源)
        optimizer.step()              # 迈一步
        optimizer.zero_grad()         # 清旧账(顺序见教材惯例: 先 step 后清也行,
                                      # 但千万别在 backward 前清)
训练循环之外 3 条铁律:
    - train()/eval() 分流: 某些层(BN/Dropout)在两种模式行为不同, 训练用 train()
    - 验证用 torch.no_grad() 包住: 验证不改参数, 别建计算图(省显存且快)
    - 打印值要 .item() 转标量: 否则整个 batch 的图都挂在你 print 上泄漏显存

首战心态预告: 我们的网络 67M 参数(99.9% 在 fc1), 数据只有 4k 张、
并且未做任何增强(用朴素管线照常训练)。预期结果:
    训练集准确率 -> 95%+  (模型背住了答案)
    验证集准确率 -> 40-60% (没见过题, 露馅)
这不是失败, 而是【过拟合】被我们亲手活捉的全过程 —— 3.2 节将正面迎战它。
"""
import json
import os
import time

import torch
import torch.nn as nn

from person_cnn import PersonCNN
from split_dataset import get_datasets

# =========================== 超参数(唯一事实来源) ===========================
EPOCHS = 8
BATCH = 32
LR = 1e-3                        # Adam 的"出厂设置"
NUM_WORKERS = 4                  # 解码食堂窗口数(1.3 节实测过: 越多越补味)
VAL_EVERY = 1                    # 每 1 个 epoch 验一次(2.5 节再细讲为什么不长不短)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
HISTORY_FILE = "history.json"    # 3.1 节曲线绘画师的数据源
BEST_FILE = "best_model.pt"      # 2.5 节主角


def evaluate(model, loader, loss_fn, device):
    """验证一轮: 返回 (平均 loss, 准确率). 全程 no_grad, 不更新任何参数."""
    model.eval()                                    # 铁律1
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():                           # 铁律2
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            total_loss += loss.item() * x.size(0)   # .item() 铁律3; 加权还原 batch 均
            correct += (out.argmax(dim=1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def main():
    torch.manual_seed(SEED)
    print(f"设备: {DEVICE}")

    # ---- 数据(1.4 节成果: 分层划分, 固定种子) ----
    train_ds, val_ds = get_datasets()
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"))   # pin: GPU 拷贝专用"直连通道", 快一截
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"))   # 验证不用 shuffle: 只求稳定可复现(1.4 节)

    # ---- 三件套: 模型/损失/优化器 ----
    model = PersonCNN().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)   # 2.2 学过的平滑款(防自信病)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ---- 记账本(3.1 节的曲线数据) ----
    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": []}
    best_val_acc, best_state = -1.0, None

    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {n_params:,}  |  训练样本 {len(train_ds)}  |  验证 {len(val_ds)}\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        # ---------------- 训练阶段 ----------------
        model.train()                      # 铁律1 的反面
        run_loss, correct, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)       # 每批数据都要过桥, 不会自动

            optimizer.zero_grad()                    # 2.3 名场面: 清旧账
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()                          # 建图->求梯
            optimizer.step()                         # 迈步

            run_loss += loss.item() * x.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            n += x.size(0)

        train_loss = run_loss / n
        train_acc = correct / n

        # ---------------- 验证阶段 ----------------
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, DEVICE)

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))

        # 跟踪 best: 只存 state_dict(参数), 不存整个模型对象 —— 这叫作"轻存档"
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch:2d}/{EPOCHS} | 训练 loss {train_loss:.4f} acc {train_acc:.1%} "
              f"| 验证 loss {val_loss:.4f} acc {val_acc:.1%} | "
              f"{time.perf_counter() - t0:.0f}s")

    # ---------------- 存档 ----------------
    os.makedirs("runs", exist_ok=True)      # 历史与最优版都进 runs/ 目录
    with open(os.path.join("runs", HISTORY_FILE), "w", encoding="utf-8") as fh:
        json.dump(history, fh, ensure_ascii=False, indent=2)
    torch.save(best_state, os.path.join("runs", BEST_FILE))
    print(f"\nbest 验证准确率: {best_val_acc:.4f} (存档 {BEST_FILE}(state_dict, 轻量))")
    print(f"历史曲线数据: {HISTORY_FILE} (3.1 节绘画原料)")
    print("\n首战点评: 训练 acc 与验证 acc 之间那条【鸿沟】 —— 就是过拟合的一字眉账单。")


if __name__ == "__main__":
    main()
