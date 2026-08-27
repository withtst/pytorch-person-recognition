"""
3.3 混淆矩阵与错题集 - 教学版

主题: 准确率是平均数, 混淆矩阵是"账单"——逐类对比, 谁是多数派的疏忽?
产出:
  1) 11x11 混淆矩阵(计数版+归一化版)  runs/confusion_matrix.png
  2) 热门错配榜: 哪两人最容易互认错?
  3) 错题集: 保存 16 张真实失误图片的拼接图 runs/wrong_grid.png
  4) 小样本警告: 垫底类先检查卷子够了没(1.4 节纪律复读)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from person_cnn import PersonCNN
from split_dataset import get_datasets

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BEST_FILE = os.path.join("runs", "best_model.pt")
TOP_WRONG = 16          # 错题集保留张数


def main():
    torch.manual_seed(42)
    model = PersonCNN().to(DEVICE)
    model.load_state_dict(torch.load(BEST_FILE, map_location=DEVICE))
    model.eval()

    _, val_ds = get_datasets()
    idx_to_class = val_ds.dataset.idx_to_class
    loader = torch.utils.data.DataLoader(val_ds, batch_size=32,
                                          shuffle=False, num_workers=4,
                                          pin_memory=True)

    # ========================================================================
    # 1. 一张张判卷: 生成 11x11 计数矩阵 CM[真] [pred]
    # ========================================================================
    cm = np.zeros((11, 11), dtype=int)
    wrong_samples = []                       # (真实标签, 预测标签, 序号)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(1)
            for yy, pp in zip(y.tolist(), preds.tolist()):
                cm[yy][pp] += 1
                # 从 subset 索引回原数据集序号, 好去磁盘找回那张图
                if yy != pp and len(wrong_samples) < TOP_WRONG:
                    wrong_samples.append((yy, pp))

    print("=" * 64)
    print("【判卷账单】混淆矩阵 CM[真类][预测类], 行=真实, 列=预测")
    names = [idx_to_class[i] for i in range(11)]
    header = "    " + "".join(f"{n[0]:>4}" for n in names)
    print(header)
    for r in range(11):
        row = " ".join(f"{cm[r][c]:>4}" for c in range(11))
        print(f"{names[r][:2]:>2} {row}")

    # ========================================================================
    # 2. 热门错配榜: {真实:预测} 的 top
    # ========================================================================
    print("\n【热门错配榜】")
    confusions = []
    for r in range(11):
        for c in range(11):
            if r != c and cm[r][c] > 0:
                confusions.append((cm[r][c], names[r], names[c]))
    confusions.sort(reverse=True)
    for cnt, a, b in confusions[:5]:
        print(f"  {a:20s} 被认成 {b:20s} x{cnt}")

    # ========================================================================
    # 3. 可视化矩阵 -> runs/confusion_matrix.png
    # ========================================================================
    fig, ax = plt.subplots(figsize=(8.5, 7))
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)     # 归一化(每行真实占比)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(11)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(11)); ax.set_yticklabels(names, fontsize=8)
    for r in range(11):
        for c in range(11):
            ax.text(c, r, cm[r][c], ha="center", va="center", fontsize=7,
                    color="white" if norm[r][c] > 0.5 else "black")
    ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")
    ax.set_title("验证集混淆矩阵 (数字=样本数, 颜色=行归一化)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join("runs", "confusion_matrix.png"), dpi=120)
    print("  矩阵图 -> runs/confusion_matrix.png")

    # ========================================================================
    # 4. 错题集: 拼接 16 张原图 + 标注 真/误 @ runs/wrong_grid.png
    # ========================================================================
    print(f"【错题集】取前 {TOP_WRONG} 张失误样本 -> runs/wrong_grid.png")

    def cell(item, idx):
        yy, pp = item
        idx_in_ds = val_ds.indices[idx]                     # subset -> 原序号
        path, _ = val_ds.dataset.images[idx_in_ds]
        img = Image.open(path).convert("RGB")
        # 固定缩到 144x192 竖幅(人像审美)蒙太奇
        img = img.resize((144, 192))
        tile = Image.new("RGB", (144, 192 + 26), "black")
        tile.paste(img, (0, 0))
        d = ImageDraw.Draw(tile)
        d.text((4, 196), f"真:{names[yy][:6]} 误:{names[pp][:6]}", fill="white")
        return tile

    tiles = [cell(w, i) for i, w in enumerate(wrong_samples)]
    grid = Image.new("RGB", (144 * 4, (192 + 26) * 4), "black")
    for i, t in enumerate(tiles[:16]):
        grid.paste(t, ((i % 4) * 144, (i // 4) * (192 + 26)))
    grid.save(os.path.join("runs", "wrong_grid.png"))
    print("  -> 每格: 上方图片 + 下方 [真:xx 误:yy], 四行四列, 人肉检查开始。")

    # ========================================================================
    # 5. 小样本警告纪律(1.4 节复读)
    # ========================================================================
    diag = np.array([cm[i][i] for i in range(11)])
    totals = cm.sum(axis=1)
    accs = diag / totals.clip(min=1)
    print("\n【小样本警告】按准确率排序(总数 < 30 的标 *):")
    order = np.argsort(accs)
    for i in order:
        star = "*" if totals[i] < 30 else " "
        print(f"  {star} {names[i]:18s} {diag[i]:>3}/{totals[i]:>3} = {accs[i]:.1%}")
    print("\n  * 星号系: 样本太少, 准确率的波动可能只是运气; 结论先说'再给几张图'。")


if __name__ == "__main__":
    main()
