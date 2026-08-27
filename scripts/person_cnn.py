"""
2.1 手写第一个 CNN - 教学版

本节目标: 亲手搭一个能跑的三层卷积网络, 并把三件事算明白:
  1. 参数量: 手算 vs model.parameters() 对账
  2. 维度:   (B,3,256,256) 每层怎么变, 为什么最后是 (B,11)
  3. 感受野: 顶层一个像素对应原始图像多大一块(直觉建立)

结构: Conv(3x3) -> ReLU -> MaxPool(2x2), 重复三次, 再接线性层。
为什么是"3 卷积 + 3 池化 + 2 全连接"这个配方? 这是 2012 年 AlexNet 以来
"CNN 经典三明治"的简化版, 先照配方做, 后边 2.4/3.2 再解释配方为何有效。
"""
import torch
import torch.nn as nn


class PersonCNN(nn.Module):
    """
    教学要点 1: nn.Module 子类的两个契约
      1. __init__ 里把层声明成"属性"(self.xxx = nn.Conv2d...),
         这样 nn.Module 自动注册它们 -> parameters() 才能遍历到。
         若写成局部变量 out = nn.Conv2d(...) 然后丢弃, 参数就永远失踪了!
      2. forward 里写前向逻辑。nn.Module 的 __call__ 会自动接 forward
         (还替你挂钩子), 所以外面写 model(x) 而不是 model.forward(x)。

    教学要点 2: 卷积全公式
      Conv2d(in_ch, out_ch, kernel, padding):
      输入 (B, C_in, H, W) -> 输出 (B, C_out, H', W')
      H' = (H + 2*padding - kernel) / stride + 1   (stride 默认 1)
      为什么 padding=1? kernel=3 时 3x3 卷积会"吃掉"边缘一圈:
      256  -> (256+2*1-3)/1 + 1 = 256  -> 尺寸不变, 只换通道数。
      尺寸不变最大的好处: 3 个卷积块之间的 H/W 数学一目了然。
    """
    def __init__(self, num_classes=11):
        super().__init__()          # 必须调用父类构造, 否则注册机制不生效

        # ---- 块 1: 3 通道 -> 32 通道 ----
        # 参数手算: 权重_kernel: 3*32*3*3 = 864 (输入通道x输出通道x核面积)
        #           偏置:      32
        #           合计 = 864 + 32 = 896
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)

        # ---- 块 2: 32 -> 64 ----
        # 32*64*9 = 18432 + 64 = 18496
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # ---- 块 3: 64 -> 128 ----
        # 64*128*9 = 73728 + 128 = 73856
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # ---- 线性塔: 输进全连接的分辨率= 256 经过 3 次 MaxPool(2) ----
        # MaxPool2d(2): 每个 2x2 窗口取最大值, 步长 2, 无重叠
        #   H' = (H - kernel) / stride + 1 = H/2  (精确对半切)
        # 256/2 = 128 -> 64 -> 32, 所以 C_out * H * W = 128 * 32 * 32 = 131072
        # 全连接第一层参数量: 131072*512 + 512 = 67,109,376 (!!)
        # 教学重点: 全连接层是参数量黑洞 —— 它占了全网络 99.8% 的参数!
        #   直观理解: 卷积层"参数共享"(同核扫全图), 前一层 896 个参数管 256x256;
        #   线性层"每个输入权重一一对应", 13 万个特征进 512 个神经元 = 6.7 千万个连接。
        #   这也是为什么小数据(4384 张)喂大网络(67M)必过拟合(2.4/3.2 会讲对策)。
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.trace = []             # 记录每层输出形状(教学用, 训练时会去掉)

    def forward(self, x):
        """前向: x: (B, 3, 256, 256) -> logits: (B, 11)"""
        B = x.size(0)                                   # 批大小, 别写死!

        # 3 个三明治块: "卷积提取 + 激活 + 池化压缩"
        # ReLU: 逐元素 max(x,0), 加入非线性。
        #   为什么"必须"非线性? 全线性堆叠 = 一个大矩阵乘(线性还是不变量),
        #   加两层必出曲线才行; 后面 2.2 节拟合函数需要。
        x = torch.relu(self.conv1(x))
        self.trace.append(x.shape)
        x = torch.max_pool2d(x, 2)
        self.trace.append(x.shape)

        x = torch.relu(self.conv2(x))
        self.trace.append(x.shape)
        x = torch.max_pool2d(x, 2)
        self.trace.append(x.shape)

        x = torch.relu(self.conv3(x))
        self.trace.append(x.shape)
        x = torch.max_pool2d(x, 2)
        self.trace.append(x.shape)

        # ---- 压平 + 线性塔 ----
        # x.view(B, -1): "-1" = 自动推算剩下所有维度; B 显式保留 -> 每张图 128*32*32
        # 与 reshape 区别: view 只允许"连续内存"的重排(不拷贝);
        # 前面 conv/pool 的输出都是连续张量, 所以 view 安全; 如果前方出现过
        # transpose/permute, view 会抛错, 那时用 reshape(它自动拷贝回连续)。
        x = x.view(B, -1)
        self.trace.append(x.shape)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)                 # 输出层不再激活: logits, 含义见 2.2 节
        self.trace.append(x.shape)
        return x


def parameter_budget(model):
    """逐层参数对账: 打印每层 (名字, 参数量), 以及全连接占比."""
    print("逐层参数量对账:")
    total = 0
    for name, p in model.named_parameters():
        print(f"  {name:22s} {p.numel():>12,}")
        total += p.numel()
    print(f"  {'总计':22s} {total:>12,}")
    fc_params = sum(p.numel() for n, p in model.named_parameters() if n.startswith("fc"))
    print(f"\n全连接塔占全部参数比例: {fc_params/total*100:.1f}%  <- 参数黑洞")
    return total


if __name__ == "__main__":
    # ========================================================================
    # 实验 1: 手算对账 (预期: 896 + 18496 + 73856 + 67109376 + 5643)
    # ========================================================================
    model = PersonCNN()
    print("=" * 64)
    print("【实验 1】参数量手算对账")
    print("  手算: conv1=896, conv2=18496, conv3=73856, fc1=67,109,376, fc2=5,643")
    for name, p in model.named_parameters():
        n = p.numel()
        if name.startswith("conv"):
            print(f"  {name}: 实算 {n:>10,}")
    parameter_budget(model)

    # ========================================================================
    # 实验 2: 维度变化路径图 (batch=2 更能看清 B 的存在)
    # ========================================================================
    print("\n" + "=" * 64)
    print("【实验 2】维度变化路径 (B=2, 3, 256, 256 起)")
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    for step, s in enumerate(model.trace):
        print(f"  步 {step}: {tuple(s)}")
    print(f"  最终 out.shape = {tuple(out.shape)}  <- (B, 11) 对应 (样本, 11 人打分)")
    # 教学: 共 8 步 -> 卷积尺寸不动只变通道(64->32->16... 不对, 看上面数值),
    # 池化每次对半切(256->128->64->32), 压平后 (B, 131072), 释放出 11 维打分。
    # 中间的 64 -> 16 是通道数, 别和空间尺寸混了。
    print("  观察: 空间尺寸 256->128->64->32 全由池化造成; 128 是通道数不是空间!")

    # ========================================================================
    # 实验 3: 感受野递推 (顶层一个像素"看"多大)
    # ========================================================================
    # 3x3 conv: 感受野 +2 (3x3 覆盖 3x3, 一层后 += 2)
    # MaxPool(2,2): 感受野 x2 (跨 2 次卷了 2x 大的区域, 只留最大值)
    # 递推: conv(3) -> pool(x2->6) -> conv(+2->8) -> pool(x2->16)
    #      -> conv(+2->18) -> pool(x2->36)
    print("\n" + "=" * 64)
    print("【实验 3】感受野理解")
    print("  conv3x3: 感受野 +2; MaxPool2: 感受野 x2; 递推结果 = 36x36 原图")
    print("  顶层一个像素 ≈ 36x36 的人脸局部 -> 这就是'组合特征': 低层边缘, 高层五官。")
    print("  对比: 若想验证全图 256 视角, 需要更多下采样层(4.1 节迁移学习的路)")

    # ========================================================================
    # 实验 4: forward 可跑, 分数归一(预告 2.2: logits -> softmax -> 概率)
    # ========================================================================
    print("\n" + "=" * 64)
    print("【实验 4】logits 初窥")
    first_logits = out[0]
    probs = torch.softmax(first_logits, dim=0)
    print(f"  logits[:5] = {first_logits[:5].tolist()}")
    print(f"  softmax 后总和 = {probs.sum().item():.4f} (概率化, 2.2 节主角)")
