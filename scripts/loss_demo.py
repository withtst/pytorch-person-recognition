"""
2.2 CrossEntropyLoss - 教学版

本节主线: 模型输出的 11 个"打分"(logits) 如何变成"损失"(一个数字)?
                             打分 -> softmax(概率化) -> 取正确答案的概率 -> 取负对数 -> 平均

核心公式(交叉熵):
    L = - (1/N) * sum_i  ln( p_i[target_i] )
    即: 每个样本, 只看模型给"正确答案"分配的概率, 求负对数。

为什么取"负对数"?
    - p=1.0 (猜对了且自信) -> -ln(1) = 0     -> 满分, 不罚
    - p=0.1 (猜对了但心虚) -> -ln(0.1)=2.3   -> 罚小分, 鼓励自信
    - p→0  (几乎没猜到)    -> -ln(0) →  +∞   -> 天罚
    ln 的"陡峭尾"把错误放得极大, 分类任务的"自信心"就是这样逼出来的。
"""
import torch
import torch.nn.functional as F

from person_cnn import PersonCNN        # 2.1 节成果
from person_dataset import PersonDataset

# 经典 4 类小样例: batch=2, 类 = [4 个抽象类]
# 正确答案: 第一个样本是类 3, 第二个是类 1
LOGITS = torch.tensor([[2.0, 1.0, 0.1, 9.0], [1.0, 5.0, 2.0, 0.0]])
TARGET = torch.tensor([3, 1])


def main():
    # ========================================================================
    # 实验 1: 手算与 torch 对账
    # ========================================================================
    print("=" * 64)
    print("【实验 1】手算 vs torch 对账 (4 类, batch=2)")
    # --- 手算第一行: logits [2.0, 1.0, 0.1, 9.0], 答案是第 3 类 ---
    # ① softmax: 分子 = exp(2)=7.389, exp(1)=2.718, exp(0.1)=1.105, exp(9)=8103.1
    #    分母 = 8114.3
    # ② 正确答案"类3"的概率 = 8103.1 / 8114.3 = 0.9986
    # ③ 负对数 = -ln(0.9986) ≈ 0.00138
    # 手算第二行: exp(1)=2.718, exp(5)=148.41, exp(2)=7.389, exp(0)=1
    #    答案"类1"概率 = 148.41 / (148.41+2.718+7.389+1) = 148.41/159.52 = 0.9304
    #    负对数 = -ln(0.9304) ≈ 0.07217
    loss_hand = (0.00138 + 0.07217) / 2          # 两样本求平均(默认 reduction=mean)
    loss_torch = F.cross_entropy(LOGITS, TARGET)
    print(f"  手算(两行取平均) = {loss_hand:.6f}")
    print(f"  torch 输出       = {loss_torch.item():.6f}")
    print(f"  是否吻合: {abs(loss_hand - loss_torch.item()) < 1e-4}")
    print("  常见手算错误(踩坑提醒):")
    print("    a) 忘记把 logits 先 exp 就直接除 -> 概率小于 1/3, loss 偏大")
    print("    b) 取错值: 取了 -ln(p_wrong) 而不是 -ln(p_correct)")
    print("    c) 分母混入别的样本的归一化数值")
    print("  正确姿势: 用 torch 对账(损失函数一行的事, 不要背公式)")

    # ========================================================================
    # 实验 2: 最著名的"torch 组合拳": cross_entropy == nll_loss(log_softmax(x))
    # ========================================================================
    print("\n" + "=" * 64)
    print("【实验 2】logits 谜题: CE 为什么可以不当 softmax?")
    print("  官方文档: nn.CrossEntropyLoss = log_softmax 之后接 nll_loss 的\"合成体\"")
    nll_route = F.nll_loss(F.log_softmax(LOGITS, dim=1), TARGET)
    print(f"  路线A: nll_loss(log_softmax(logits)) = {nll_route.item():.6f}")
    print(f"  路线B: cross_entropy(logits)         = {loss_torch.item():.6f}")
    print(f"  两路结果: {'完全一致 OK' if abs(nll_route.item() - loss_torch.item()) < 1e-6 else '不一致!'}")
    # 为什么"合成"? 分两步走会有两次指数运算, logits 很大时 exp 溢出爆炸(实验 5),
    # 且两步间的精度误差在累积。合成后 torch 单步内做 log-sum-exp(max 平移)又快又稳。
    # 对我们意味着: 模型最后一层 keep logits, loss 里再 softmax —— 接口约定 (B, 11)!

    # ========================================================================
    # 实验 3: 置信度稀释 + 标签平滑
    # ========================================================================
    print("\n" + "=" * 64)
    print("【实验 3】对照: 置信度稀释 / 标签平滑")
    loss_diluted = F.cross_entropy(LOGITS / 100, TARGET)
    print(f"  原始 logits:      loss = {loss_torch.item():.4f}  (模型对答案很有把握)")
    print(f"  除以 100 稀释:    loss = {loss_diluted.item():.4f}")
    print(f"  均匀 4 类的下限 = {-torch.log(torch.tensor(1/4)).item():.4f} (ln(4))")
    # 解读: 稀释后 logits≈0, softmax 逼近均匀 -> 模型等于"没看到", 惩罚 = -ln(1/4)。
    # 训练早期就是这样: 随机初始化模型的 loss 约等于 ln(类别数), 从那儿下坡。

    loss_smooth = F.cross_entropy(LOGITS, TARGET, label_smoothing=0.1)
    print(f"\n  原始版:     loss = {loss_torch.item():.4f}")
    print(f"  smoothing=0.1: loss = {loss_smooth.item():.4f} (变大了一点)")
    # 平滑的实质: 把"1.0 全押正确答案"改成"0.9 押对 + 0.1 平摊给所有 4 类"。
    # 好处: (1) 防止模型自信到 99.9%再出错(对噪声标签极敏感, 会记住脏数据);
    #       (2) 全类别保留一点"路不平"的梯度, 某些类太稀时不至于死路。
    # 代价: loss 不再能降到 0, 读曲线时别被"最小 loss 变大了"吓到。

    # ========================================================================
    # 实验 4: 真实模型(零训练)的 loss 应≈ln(11), 验证"随机猜测基准"
    # ========================================================================
    print("\n" + "=" * 64)
    print("【实验 4】随机初始化模型的真实 loss (思考题现场)")
    model = PersonCNN()
    ds = PersonDataset()
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True, num_workers=0)
    xbatch, ybatch = next(iter(loader))
    with torch.no_grad():
        out = model(xbatch)
    loss_rand = F.cross_entropy(out, ybatch)
    floor = float(torch.log(torch.tensor(11.0)))
    print(f"  11 类均匀猜测的理论 loss = ln(11) = {floor:.4f}")
    print(f"  随机模型实际 loss        = {loss_rand.item():.4f}")
    print(f"  结论: 模型还没学任何东西, 就打出了接近理论下界的成绩 -> "
          f"看到 loss≈{floor:.2f} 不要庆祝, 它只是在\"抛硬币\"。")
    print(f"  学到真东西的标志: loss 稳定低于 2.0(最好 <1.0)。")

    # ========================================================================
    # 实验 5(速览): 为什么不手写 softmax+log? 数值稳定性
    # ========================================================================
    print("\n" + "=" * 64)
    print("【实验 5】数值稳定性: logits 放大 1000 倍")
    huge = LOGITS * 1000
    try:
        exp_all = huge.exp()
        hand = -torch.log((exp_all / exp_all.sum(dim=1, keepdim=True))[range(2), TARGET]).mean()
        print(f"  手写 softmax 路径: {hand.item():.3f}")
    except Exception as e:
        print(f"  手写 softmax 路径: 崩溃 ({type(e).__name__}: {e})")
    print(f"  torch 合成路径: {F.cross_entropy(huge, TARGET).item():.3f}  (稳如老狗)")
    # 解读: exp(9000) 超过最大浮点数, 手写一次 exp 就溢出; F.cross_entropy 内部用
    # max 平移技巧(每个 logit 先减掉所在行的最大值再 softmax), 任何尺度都不炸。
    # 结论: 永远用官方融合函数, 它既正确又稳定 —— 这是教科书级的"别自己造轮子"。


if __name__ == "__main__":
    main()      # 守卫: 若其他脚本将来 import 本模块, 实验不会重复执行
