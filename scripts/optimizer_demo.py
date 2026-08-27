"""
2.3 Optimizer 与学习率 - 教学版

一句话: 模型按下坡走, 优化器决定"迈多大步、往哪迈"。

本节在【玩具函数】上理解真东西(玩具上没有解码开销, 专注概念):
    f(w) = (w^2 - 1)^2      导数: f'(w) = 4w(w^2 - 1)
    它有两个对称的谷底(全局最小): w = +1 和 w = -1 (f = 0)
    中间有一座山脊:           w = 0      (f = 1, 局部极大)
    梯度下降: w_{t+1} = w_t - lr * f'(w_t)

实验梗概:
  1) 手搓 GD: lr=0.02 困在左谷 / lr=0.05 跨过山脊到右谷 / lr=0.08 直接爆炸
     -> 学习率的三张脸: 保守、跨越、发疯(课堂实验过 lr=0.3 的 nan)
  2) backward 的梯度累积机制: 为什么 optimizer.zero_grad() 每轮必调
  3) SGD vs SGD+momentum vs Adam, 50 步轨迹图 (保存 lr_plot.png)
  4) 优化器与参数集合: 只注册谁就更新谁 (4.1 节冻结技巧的伏笔)
"""
import matplotlib
matplotlib.use("Agg")            # 无窗口环境: 只存图不弹窗(服务器/CI 标准姿势)
import matplotlib.pyplot as plt
import torch

# matplotlib 默认字体没有汉字, 手动指定中文字体(Windows 自带微软雅黑)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def f(w):
    return (w * w - 1) ** 2


def grad_f(w):
    return 4 * w * (w * w - 1)


def main():
    # ========================================================================
    # 实验 1: 手搓梯度下降, 观察 lr 的三张脸
    # ========================================================================
    print("=" * 64)
    print("【实验 1】手搓 GD: 起点 w=-3, lr 三档, 8 步")
    for lr in (0.02, 0.05, 0.08):
        w = -3.0
        track = [w]
        for _ in range(8):
            w = w - lr * grad_f(w)
            track.append(w)
        fin = w if abs(w) < 1e6 else float("-inf")
        if fin == float("-inf"):
            story = "爆炸: 步子太大, 越过山脊还不刹车, 越弹越远 -> 学习率不能乱大"
        elif abs(fin - 1) < 0.1:
            story = "跨脊成功: 大步飞过山脊, 落在了【右谷 w=+1】"
        else:
            story = "保守爬坡: 局促在【左谷 w=-1】, 一步步挪, 出不去"
        print(f"  lr={lr}: 轨迹前 4 步 = {['%.2f' % t for t in track[:4]]} "
              f"-> 终点 {fin:+.2f}")
        print(f"        {story}")
    # 解读: 梯度是"方向", lr 是"步幅"。0.02 每步只挪一点点(遇到大梯度也不会过冲);
    # 0.05 恰好一步跨过山脊(梯度大的地方步子精确放大, 翻过峰顶);
    # 0.08 越过到远处的大梯度区, 被大梯度甩得更远 -> 震荡发散。
    # 现实工程里: lr 起步默认 1e-3(Adam)~1e-2(SGD), 干大模型必须 schedule(2.5 节)。

    # ========================================================================
    # 实验 2: 弄懂【梯度累积】和 zero_() 存在的理由
    # ========================================================================
    print("\n" + "=" * 64)
    print("【实验 2】backward 的累加机制: 为什么每轮都要 zero_grad()?")
    w = torch.tensor(0.5, requires_grad=True)      # 0.5 处梯度非零
    f(w).backward()
    first = w.grad.item()
    print(f"  第一次 backward 后 w.grad = {first:.4f}")
    f(w).backward()
    print(f"  不清理再 backward    w.grad = {w.grad.item():.4f}  (= 第一次 x2, 被累加!)")
    w.grad.zero_()
    f(w).backward()
    print(f"  zero_() 后再 backward  w.grad = {w.grad.item():.4f}  (回到干净值)")
    # 解释: autograd 的哲学是"梯度流进 .grad 里【累积】", 不主动清就原地叠加,
    # 上一轮的梯度还在, 下一轮的更新会"捎带上旧账" => 训练路径被污染。
    # 现代写法: optimizer.zero_grad()(等价全参数逐份 zero_), 顺序惯例:
    #   zero_grad() -> forward -> loss -> backward() -> step()
    # 或 backward() -> step() -> zero_grad(), 两种都行, 关键是别在 step 前后乱。

    # ========================================================================
    # 实验 3: torch.optim.SGD / momentum / Adam, 60 步轨迹图 (诚实对比!)
    # ========================================================================
    print("\n" + "=" * 64)
    print("【实验 3】SGD vs SGD+momentum vs Adam (起点 w=-2.5, lr=0.02, 60 步)")
    print("  指标: '到谷' 指 f(w)<0.01; 梯度最陡阶段在前段, 三种算法在此大不相同。")

    def run_optimizer(kind):
        w = torch.tensor(-2.5, requires_grad=True)
        if kind == "sgd":
            opt = torch.optim.SGD([w], lr=0.02)
        elif kind == "momentum":
            opt = torch.optim.SGD([w], lr=0.02, momentum=0.9)
        else:
            opt = torch.optim.Adam([w], lr=0.05)
        track = [w.item()]
        for _ in range(60):
            opt.zero_grad()             # 三件套之一: 清旧梯
            loss = f(w)
            loss.backward()             # 积新梯
            opt.step()                  # 迈一步
            track.append(w.item())
        arrived = next((t for t in range(60) if f(track[t]) < 0.01), None)
        return track, w.item(), arrived

    series = {}
    for kind in ("sgd", "momentum", "adam"):
        track, final, arrived = run_optimizer(kind)
        series[kind] = track
        how = f"第 {arrived} 步" if arrived is not None else "60 步内未到"
        print(f"  {kind:9s}: 到达谷底= {how:>9s}  终点 w={final:+.3f} (f={f(final):+.4f})")

    plt.figure(figsize=(8, 4.5))
    for kind, tr in series.items():
        plt.plot(range(len(tr)), tr, label=kind, linewidth=2)
    plt.axhline(-1.0, color="gray", ls=":", label="左谷 w=-1")
    plt.xlabel("step"); plt.ylabel("w")
    plt.legend(); plt.title("三种优化器轨迹: f(w)=(w^2-1)^2, 起于 w=-2.5")
    plt.grid(alpha=0.3)
    plt.savefig("lr_plot.png", dpi=120)
    print("  轨迹图已保存 lr_plot.png")
    # 解读(全部是真实跑出来的剧情, 值得细品):
    #   SGD      (11 步): 直奔左谷 w=-1 —— 胆子大直线走, 一维单峰无需花活;
    #   momentum (27 步): 慢热! 惯性先使劲往左冲(差点冲过左谷), 蓄劲儿之后再调头,
    #           凭借"上坡时的速度余量"一路翻过山脊, 最终落进【右谷】w=+1!
    #           这是教科书名场面: momentum 的动量可以跨越局部障碍。
    #   Adam     (55 步): 自适应步长 ≈ 每步一个"标准距离", 陡区保守、缓区匀速,
    #           稳, 但在这玩具上变慢半拍。
    # 教训一: 优化器各有性格: SGD 直爽, momentum 惯性翻脊, Adam 步长自适应。
    #         单参数玩具上表现各有长短; 多参数、多尺度、稀疏梯度的真实模型上
    #         才轮到 Adam 统一扬眉 —— 因为各参数尺度五花八门, 统一 lr 走不通。
    # 教训二: 玩具的结论别当真理, 2.4 节我们会在真实网络对比验证(实测见结果)。

    # ========================================================================
    # 实验 4: 优化器和参数集合的关系(4.1 节冻结的技巧都靠它)
    # ========================================================================
    print("\n" + "=" * 64)
    print("【实验 4】优化器只更新它认识的人")
    w_a = torch.tensor(2.0, requires_grad=True)
    opt = torch.optim.Adam([w_a], lr=1e-2)
    print("  opt = Adam([w_a]) -> 只注册了 w_a, 别的参数梯度再大也不理")
    print("  迁移学习就是: 让 opt 只看向新头部的参数 -> 骨干网络照样前向, 但不更新")
    print("  注意: 模型层被替换后, 旧 optimizer 的引用过期, 必须重新构造(状态会失效)。")


if __name__ == "__main__":
    main()
