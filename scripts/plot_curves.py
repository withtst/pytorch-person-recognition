"""
3.1 训练曲线诊断 - 教学版

数据说话了。把 2.4 节 history.json 画成四线图, 然后做"曲线医生":
  处方单上要回答:
    1. 训练 loss 还在降吗? (若降 -> 模型还在学; 若平 -> 学不动了)
    2. 验证 loss 在哪一个 epoch 后【拐头向上】? (转折点 = 过拟合发令枪)
    3. 训练/验证准确率间的鸿沟多大? (鸿沟宽 = 背答案型选手)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HISTORY = os.path.join("runs", "history.json")
OUT = os.path.join("runs", "curves.png")


def main():
    with open(HISTORY, encoding="utf-8") as fh:
        h = json.load(fh)

    epochs = list(range(1, len(h["train_loss"]) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    # --- 左图: loss ---
    ax1.plot(epochs, h["train_loss"], "o-", label="train loss")
    ax1.plot(epochs, h["val_loss"], "s--", label="val loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
    ax1.set_title("Loss 曲线: 验证拐头向上 = 过拟合警报")
    ax1.legend(); ax1.grid(alpha=0.3)
    # --- 右图: acc ---
    ax2.plot(epochs, h["train_acc"], "o-", label="train acc")
    ax2.plot(epochs, h["val_acc"], "s--", label="val acc")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy")
    ax2.set_title("准确率曲线: 双线鸿沟 = 背答案的证据")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT, dpi=120)

    print(f"曲线已保存: {OUT}\n")

    # ========================================================================
    # 曲线医生出报告
    # ========================================================================
    print("=" * 64)
    print("【诊断报告】")
    # 1. 训练 loss 趋势
    tl = h["train_loss"]
    last_3 = tl[-3:]
    trend = "下降" if last_3[2] < last_3[0] - max(0.001, abs(last_3[0])*0.01) else ("持平" if abs(last_3[2]-last_3[0]) < 0.02 else "回升")
    print(f"  训练 loss: {tl[0]:.3f} -> {tl[-1]:.3f} (近 3 轮 {last_3} -> {trend})")

    # 2. 验证拐点(去找 val_loss 最小的 epoch)
    vl = h["val_loss"]
    min_ep = vl.index(min(vl)) + 1
    after = vl[min_ep:]
    turned = after[-1] > min(vl) if after else False
    print(f"  验证 loss: 最小出现在 Epoch {min_ep} ({vl[min_ep-1]:.4f}), "
          f"其后 {'持续反弹' if turned and after[-1] > after[0] else '基本徘徊(准拐点)'}")

    # 3. 鸿沟
    gap = (h["train_acc"][-1] - h["val_acc"][-1]) * 100
    print(f"  末轮鸿沟: 训练 acc {h['train_acc'][-1]:.1%} - 验证 acc {h['val_acc'][-1]:.1%} "
          f"= {gap:.1f} 个百分点")
    print(f"  判据: 训练 acc 一直涨 + 验证 acc 徘徊/回落 = 典型的【过拟合】。")
    print("  处方: 3.2 节上药 —— 增强(RandomHorizontalFlip/ColorJitter/真实统计量)、")
    print("        正则化(weight_decay)、Dropout、以及减参(全连接层是 99.9% 的黑洞)。")
    print("\n  医生提示: '拐点后一刻' = 训练应该停的点; 我们 2.4 用的是 best-model 存档,")
    print("          它已经自动停在了 Epoch 7(85.2%), 比 8 轮的 84.6% 更好 —— 这就是 2.5 的意义。")


if __name__ == "__main__":
    main()
