"""
1.2 transforms 管线 - 教学版

本节回答三个问题:
  1. ToTensor 到底替我们做了什么?
  2. Normalize(mean, std) 在干什么, 有什么用?
  3. transform 应该在 __init__ 里建好, 还是 __getitem__ 里现造?
见下方每段注释; 运行后对照打印结果理解。

⚠️ 本节最大的地雷(实测踩坑花费一小时):
  Normalize 的签名是 Normalize(mean, std, inplace=False)。
  T.Normalize(0.5, 0.5, 0.5) 会被解释成 mean=0.5, std=0.5, inplace=0.5!
  inplace=0.5 是"真值", 会开启原地修改 --- 返回的是原对象、还可能悄悄什么都不改。
  正确姿势: 传【列表】 T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]), 只留两个位置参数。
  教训: 任何"看起来返回值一样"的 API 都要用【数值】验证, 而不是用 is 或形状比较。
"""
import os
import random

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from person_dataset import PersonDataset  # 1.1 节的成果, 直接复用(scripts/ 同目录)

# ============================================================================
# 1. 官方管线: Compose 的组合拳
# ============================================================================
# Compose 做的事简单到不可思议: 按顺序执行, 上一个的输出是下一个的输入。
# 它的价值在"可声明性": 用什么变换, 什么顺序, 一目了然; 也方便训练/验证切换管线。
OFFICIAL_PIPELINE = T.Compose([
    # ---- 第零拳: Resize（它不在 ToTensor 里, 第一次运行报错就是证明）----
    # 不 resize 的下场: 张量直接膨胀到 2727x1536, 后面 batch/网络全盘皆输。
    # 必须显式指定 interpolation=LANCZOS, 与 1.1 手写版一致, 否则等价性对不上:
    #  T.Resize 默认 BILINEAR, 与 LANCZOS 有微小像素差异, 复现时就成了"神秘误差"。
    T.Resize((256, 256), interpolation=T.InterpolationMode.LANCZOS),

    # ---- 第一拳: ToTensor ----
    # 它替我们做的"四步"（对照 1.1 手写版）:
    #   a) 接受 PIL Image 或 numpy (H, W, C) uint8
    #   b) 若 PIL 带 α 通道, 自动合成到 RGB
    #   c) (H,W,C) -> (C,H,W)     ==== numpy 是 .transpose, PIL 是 .permute
    #   d) 除以 255 -> float32, 值域 [0, 1]
    # 注意: resize 是"空间变换", 与"转张量"是两回事, 所以 Compose 里要显式两步。
    T.ToTensor(),

    # ---- 第二拳: Normalize ----
    # 数学: out = (x - mean) / std, 每个通道分别做。
    # ToTensor 之后输入是 [0,1]; mean=std=0.5 时: (x-0.5)/0.5 恰好把 [0,1] 映射到 [-1,1]。
    # 为什么零均值重要: 见文件末尾思考题答案。
    # TODO(第 3.2 节): mean/std 应换成"你的数据集"的真实通道统计量, 而不是偷懒的 0.5。
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# ============================================================================
# 2. 手写版(1.1 节遗产): 作为对照组, 感受"官方管线 = 我们的四步 + Normalize"
# ============================================================================
def hand_pipeline(img: Image.Image) -> Image.Image:
    """1.1 节的默认管线, 变成纯函数以便对比."""
    img = img.convert("RGB")
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    return img


def hand_to_tensor(img: Image.Image) -> torch.Tensor:
    """手写的 ToTensor: 接收 PIL(已 resize), 输出 (C,H,W) [0,1] float32."""
    arr = np.array(img)                                # (H,W,C) uint8 [0,255]
    x = torch.from_numpy(arr).permute(2, 0, 1)         # (C,H,W), 保持 uint8
    return x.to(torch.float32) / 255.0                 # [0,1] float32


# ============================================================================
# 3. 对比实验: 同一张图, 两条管线, 打出来看
# ============================================================================
def compare(path: str) -> None:
    print("=" * 70)
    print(f"对比图片: {os.path.basename(path)}")

    img_raw = Image.open(path)

    # --- A. 手写管线(1.1) ---
    x_hand = hand_to_tensor(hand_pipeline(img_raw))
    print(f"  手写  shape={tuple(x_hand.shape)} dtype={x_hand.dtype}")
    print(f"        min={x_hand.min():.4f}  max={x_hand.max():.4f}  mean={x_hand.mean():.4f}")

    # --- B. 官方管线 ---
    x_off = OFFICIAL_PIPELINE(img_raw)
    print(f"  官方  shape={tuple(x_off.shape)} dtype={x_off.dtype}")
    print(f"        min={x_off.min():.4f}  max={x_off.max():.4f}  mean={x_off.mean():.4f}")

    # --- C. 说明: 两条管线输出尺寸一致(都 256), 但值域不同是"故意的" ---
    # 手写版没有 Normalize -> [0,1]; 官方管线 Normalize(0.5,0.5,0.5) 后 -> [-1,1]。
    # 等价性对比放到 compare_apple_to_apple 里做(那才是公平擂台)。


def compare_apple_to_apple(path: str) -> None:
    """公平对比: 手写四步 vs ToTensor(等价性验证); 再验证 Normalize 数学公式."""
    img_raw = Image.open(path)

    # 公平擂台: 手写版 = convert+resize+permute+/255; 官方对齐版 = Resize+ToTensor(无 Normalize)
    tensor_only = T.Compose([
        T.Resize((256, 256), interpolation=T.InterpolationMode.LANCZOS),
        T.ToTensor(),
    ])

    x_hand = hand_to_tensor(hand_pipeline(img_raw))    # 手写
    x_off = tensor_only(img_raw)                        # 官方 ToTensor

    diff = (x_hand - x_off).abs().max().item()
    print(f"  ToTensor 等价性: 最大绝对偏差 = {diff:.2e} "
          f"({'<= 1e-6, 数值完全一致 OK' if diff < 1e-6 else '不一致, 有 bug!'})")
    # 手写与官方的 diff 理论上应该是 1e-6 量级, 因为 (uint8/255) 与 ToTensor 内部实现一致。

    # Normalize 数学验证: (x - 0.5) / 0.5  ==  Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    x_norm_manual = (x_off - 0.5) / 0.5
    x_norm_pipe = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(x_off)
    diff2 = (x_norm_manual - x_norm_pipe).abs().max().item()
    print(f"  Normalize 数学等价: 最大绝对偏差 = {diff2:.2e} "
          f"({'<= 1e-6, = (x-0.5)/0.5 OK' if diff2 < 1e-6 else '不一致!'})")


# ============================================================================
# 4. 真实数据统计 200 张: Normalize 前后的分布
# ============================================================================
def distribution_effect(ds: PersonDataset) -> None:
    print("=" * 70)
    print("全数据集抽样 200 张: Normalize 前后通道均值/标准差")
    idxs = random.sample(range(len(ds)), 200)

    collector_before = []
    collector_after = []
    for i in idxs:
        x = ds[i][0]                                # 手写版: [0,1]
        collector_before.append(x)
        x_after = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(x)   # 非 inplace, x 不被改
        collector_after.append(x_after)

    def chan_stats(list_of_x):
        stack = torch.stack(list_of_x)              # (200, 3, 256, 256)
        per_chan = stack.mean(dim=(0, 2, 3))        # (3,): 每条通道除以 200*256*256
        per_chan_std = stack.std(dim=(0, 2, 3))
        return per_chan, per_chan_std

    m_before, s_before = chan_stats(collector_before)
    m_after, s_after = chan_stats(collector_after)

    print("  通道    前: mean | std       后: mean | std")
    for c, (mb, sb, ma, sa) in enumerate(zip(m_before, s_before, m_after, s_after)):
        print(f"  R/G/B[{c}]  {mb:.3f} | {sb:.3f}     ->  {ma:+.3f} | {sa:.3f}")
    print("  结论: 归一化后各通道均值≈0, 输入整体围绕 0 对称分布(梯度解释见文件尾).")


# ============================================================================
# 5. 思考题答案
# ============================================================================
"""
Q3: transform 在 __init__ 里建好还是 __getitem__ 里现造?
  答: __init__ 里建好, 存 self.transform。理由: 训练 30 epoch x 4384 张 = 13 万次
  __getitem__; 每次现造 Compose 对象是纯浪费。更大的理由是"语义": 管线是数据集的配置,
  从创建那一刻起固定, 不该被取样本劫持。这也符合 1.1 的"预留注入点"设计:
  PersonDataset(data_dir, transform=OFFICIAL_PIPELINE)。

Q1/Q2(梯度原理):
  (Q1 答案见上方 ToTensor 注释的四步; 本节"等价性实验"已用数值证实官方=手写)
  (Q2) 为什么零均值输入训得快? 梯度下降里权重更新 delta_W ~ -lr * dL/dW。
  若输入 x 全部 >0(比如 [0,1]), 同一通道的梯度符号受 x 符号压制:
  整个 batch 的 dL/dW 只能"同向或反向"摆, 像被捆住手脚走下坡, 路径是折线 S 形;
  零均值后才允许各方向自由。此外 0.5/0.5 把各通道拉到同一尺度, 防某一通道过度敏感。
  [进阶] 更严谨: 不归一化会扩大输入分布的条件数, 梯度尺度严重失衡。
"""
# ============================================================================


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)

    ds = PersonDataset()                                # 构造即打印摘要
    first_img_path = ds.images[0][0]

    compare(first_img_path)
    compare_apple_to_apple(first_img_path)
    distribution_effect(ds)
