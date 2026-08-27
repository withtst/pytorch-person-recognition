"""
1.3 DataLoader - 教学版

一句话: DataLoader 是 dataset 的"出餐窗口"。dataset 只是食材仓库(按编号取一份),
DataLoader 按 batch_size 打包(默认还帮你 shuffle、跨进程并行取货)。

本节要搞明白 5 件事:
  1. batch 张量形状 (B, 3, 256, 256) 的 B 是谁打包出来的、凭什么在第一位
  2. shuffle 到底是什么在被打乱(思考题)
  3. drop_last 处理不整除的余数
  4. num_workers 并行读取的原理和 Windows 大地雷
  5. 为什么 DataLoader 是"迭代器"而不是"列表"

⚠️ 速度设计说明: 机械演示用 512 张的 Subset(逻辑与全量一致, 跑得动),
    drop_last 的数学验证只算 len()(零成本), 尾巴样本单独取 16 张看形状。
    实测背景: 全量 4384 张 2700px 大图做 LANCZOS, 单 epoch 解码就要 3-4 分钟
    —— 这本身就是数据管线的第一瓶颈: GPU 永远在等"饭"。
"""
import time

import torch
from torch.utils.data import DataLoader, Subset

from person_dataset import PersonDataset    # 1.1 节成果

BATCH = 32

if __name__ == "__main__":
    # ---- 构造 dataset 放 main 内: Windows 子进程(worker)会重新导入本脚本,
    #      守卫没写 = bootstrapping 递归崩溃(本脚本存在的原因) ----
    ds = PersonDataset()
    # 子集机制: 不复制数据, 只"借看"一批索引。注意: Subset(range(512)) 会全取到
    # 前 512 张(全是一个人, shuffle 演示会失效!) 所以按类分层取样: 每人 47 张。
    stratified = []
    taken = {}
    for i, (_, label) in enumerate(ds.images):
        if taken.get(label, 0) < 47:
            taken[label] = taken.get(label, 0) + 1
            stratified.append(i)
    small_ds = Subset(ds, stratified)
    print(f"分层子集: {len(small_ds)} 张 = 11 类 × 47 张")

    # ========================================================================
    # 1. 构造与第二个灵魂参数: batch_size / shuffle / num_workers / drop_last
    # ========================================================================
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=2)
    # 构造本身很便宜! 它只是"预约"档期: 真正出餐在每次迭代时,
    # 所以 DataLoader 是【迭代器】而非列表(不会等第一批就占满内存)。
    # batch_size 怎么选: 32 是常用起点。1 张=batch 梯度方差大, 更新像"抽风";
    #  4384 张=整批平均, 更新缓慢且显存爆炸; 折中 16/32/64 看显存脸色。
    # num_workers: 0=主进程亲自解图(GPU 干等), >0=起子进程并行"摆饭"。
    print(f"len(loader) = {len(loader)}  <- 一个 epoch 走几批?"
          f" 4384 张 / 32 = {4384/32} -> 整除, 恰好 {len(loader)} 批")

    # ========================================================================
    # 2. 亲眼看到 batch 出炉
    # ========================================================================
    x, y = next(iter(loader))   # 取第一批
    print(f"\nx.shape = {tuple(x.shape)}   <- B=32 是谁包出来的?")
    print(f"y.shape = {tuple(y.shape)} (dtype={y.dtype})")
    # 答案: 默认 collate_fn(default_collate) 把 32 个 (3,256,256) tensor
    # 用 torch.stack 沿新维度拼成 (32, 3, 256, 256): 所以 B 永远在第一位。
    # 32 个 int 标签也被 stack 成 tensor(int64): x[i] 与 y[i] 永远同一样本(pair 不破)。
    print(f"整批 y 值: {y.tolist()}")

    # ========================================================================
    # 3. shuffle 实验: 谁在被随机?(512 张子集, 16 批, 秒级)
    # ========================================================================
    print("\n" + "=" * 70)
    def first_two(loader_):
        res = []
        for i, (_, y_) in enumerate(loader_):
            if i == 2:
                break
            res.append(y_.tolist())
        return res

    ordered = DataLoader(small_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    shuffled = DataLoader(small_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    print("shuffle=False 前两批:")
    for b, lb in enumerate(first_two(ordered)):
        print(f"  批 {b}: {lb}")
    print("shuffle=True 前两批:")
    for b, lb in enumerate(first_two(shuffled)):
        print(f"  批 {b}: {lb}")
    print("解读: 不 shuffle 时每批等于'一个人的连续切片'(每批 32 张同一个人的碎片);")
    print("      shuffle 后一批混进 11 个人。打乱的到底是什么?")

    # ========================================================================
    # 4. drop_last 实验: 制造余数  (4384 = 64*68 + 32, 余 32 张)
    # ========================================================================
    print("\n" + "=" * 70)
    for drop in (False, True):
        l64 = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, drop_last=drop)
        print(f"batch=64 drop_last={drop}: len(loader)={len(l64)} "
              f"(全量 4384 张, 零解码, 只看档期)")
    # 尾巴到底多大? 单独把最后 32 张做成子集, 亲眼验证:
    tail = Subset(ds, range(68 * 64, len(ds)))         # 索引 [4352, 4384) 共 32 张
    l_tail = DataLoader(tail, batch_size=64, shuffle=False, num_workers=0)
    x_t, _ = next(iter(l_tail))
    print(f"尾部子集(32 张)用 batch=64 打包: x.shape={tuple(x_t.shape)} <- 不满批 32")
    print("解读: drop_last=True 会把这条 32 张的尾巴扔掉(批数 69->68)。")
    print("      短批为何不要去? 除形状不一致外, 最后一次优化会把 32 张的梯度")
    print("      当 64 张的量级看待, 学习率被'误调', 尾巴还恒定损失信息(训练不均)。")
    print("      冷知识: 先把类内 shuffle 打乱, 尾巴里各人有份, 风险更小。")

    # ========================================================================
    # 5. num_workers 计时实验 (512 张子集, 1 epoch = 16 批, 快)
    # ========================================================================
    print("\n" + "=" * 70)
    for nw in (0, 2):
        t0 = time.perf_counter()
        for _ in range(2):                     # 2 个 epoch 取平均(首次含 spawn 开销)
            for x_, y_ in DataLoader(small_ds, batch_size=BATCH,
                                     shuffle=False, num_workers=nw):
                pass
        t_epoch = (time.perf_counter() - t0) / 2
        print(f"num_workers={nw}: 每 epoch 平均 {t_epoch:.2f}s (512 张)")
    print("解读(以实测输出为准):")
    print("  nw=0: 主进程读一批解一批——解码时 GPU 在排队等饭。")
    print("  nw=2: 子进程并行解码, 主进程只接货; 若提速不明显, 说明磁盘/CPU 带宽")
    print("        先到瓶颈, 或数据量太小; 理论≠现实, 永远用数字说话。")
    print("  旁白: 你刚看到终端里 PersonDataset 日志重复打印了吧? 那是 workers")
    print("        各自重新实例化了一份 dataset ——轻索引+按需解码的意义(见文件尾)。")

    # ========================================================================
    # 6. 思考题答案: shuffle 打乱的究竟是什么?
    # ========================================================================
    # 答: DataLoader 默认用 RandomSampler(索引采样器)【只重排索引序列】:
    #     epoch 开始 -> 生成 0..N-1 的随机排列 -> 依次喂 __getitem__(idx)。
    #     __getitem__ 必须保持"同索引同样本"的纯函数。
    #     若在 __getitem__ 内部偷偷随机: (a) epoch 不可复现; (b) debug 崩溃;
    #         (c) 与 3.2 节的"变换随机"混淆(那是变换在样本上随机, 和抽样无关)。
