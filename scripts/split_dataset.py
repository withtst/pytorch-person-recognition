"""
1.4 数据集划分 train/val - 教学版

为什么必须划分?
  训练集负责"喂饱"模型, 验证集负责"体检"模型。
  模型会背答案: 让它见过一万遍的数据它记得, 没见过的才算真本事。
  所以 val 的数据【永远不得参与梯度更新】, 它只做裁判:
    1. 帮你判断"训练表现好"是不是"背答案"(过拟合监测, 3.1 节主角)
    2. 帮你挑选最佳模型存档(2.5 节 best model)

本节三件事:
  1. random_split 80/20: 是什么、切完各类剩多少
  2. 随机种子纪律: 同种子必有同划分(实验验证)
  3. 分层划分: 解决"tifa 只有 79 张"的验证集事故
"""
from collections import Counter

import torch
from torch.utils.data import Subset, random_split

from person_dataset import PersonDataset

VAL_RATIO = 0.2


def class_counts(subset, ds):
    """统计一个 subset 里每个类别的张数(用 dataset 的记忆, 不解码一张图)."""
    # subset.indices: 它是"选中的索引列表"(random_split 和 Subset 都一样)
    return Counter(ds.images[i][1] for i in subset.indices)


def print_distribution(ds, subset, title):
    counts = class_counts(subset, ds)
    print(f"\n{title}  ({len(subset)} 张)")
    for name, idx in sorted(ds.class_to_idx.items(), key=lambda kv: kv[1]):
        c = counts.get(idx, 0)
        full = cnt_by_class[idx]
        pct = c / full * 100 if full else 0
        print(f"  [{idx:>2}] {name:20s} {c:>4} 张 (占该人 {pct:5.1f}%)")


if __name__ == "__main__":
    ds = PersonDataset()
    cnt_by_class = Counter(l for _, l in ds.images)     # 每人总数字典

    # ========================================================================
    # 1. random_split: 按索引切, 不搬运数据
    # ========================================================================
    # torch.utils.data.random_split(ds, [0.8, 0.2]) 或 [3507, 877]:
    # 它生成两个子集对象, 内部<只>存选中索引列表(indices) —— 数据是"借看"的,
    # 没复制一张图。这就是划分"便宜"的原因: 复制索引 O(1), 复制数据 O(DB)。
    # 训练时共享同一份文件! 代价是理解陷阱: 任何"在训练里改数据集"的行为都不行
    # (比如把原图裁掉, train/val 同时遭殃), 数据增强请放在读取管线(1.3 节设计)。
    train_all, val_all = random_split(ds, [1 - VAL_RATIO, VAL_RATIO])

    print("=" * 70)
    print("【实验 1】random_split 80/20 一次划分")
    print(f"  随机划分规模: train={len(train_all)}, val={len(val_all)} "
          f"(共 {len(ds)})")
    print(f"  注意不整除: 4384*0.2 = {4384*VAL_RATIO:.2f} -> "
          f"取整差异由 torch 的舍入规则决定(向下/向上累计到总数), 不必背公式, 打印对不上就查")

    # ========================================================================
    # 2. 种子纪律实验
    # ========================================================================
    print("\n" + "=" * 70)
    print("【实验 2】随机种子纪律 (看 tifa 的 val 张数, class_idx=2)")

    print("  不设种子跑 5 次 tifa 的 val 张数:")
    no_seed = []
    for k in range(5):
        _, v = random_split(ds, [1 - VAL_RATIO, VAL_RATIO])
        no_seed.append(class_counts(v, ds).get(2, 0))
    print(f"    -> {no_seed}  (每次皆不同: 79*0.2 = {79*VAL_RATIO:.1f} 附近随机漂移)")

    print("  torch.manual_seed(42) 后跑 5 次:")
    fixed = []
    for k in range(5):
        torch.manual_seed(42)                      # 重置全局 RNG(随机采样的种子)
        _, v = random_split(ds, [1 - VAL_RATIO, VAL_RATIO])
        fixed.append(class_counts(v, ds).get(2, 0))
    print(f"    -> {fixed}  (5 次全等: 同种子必同划分)")
    # 工程规范: split 前务必固定种子(或传入 generator), 否则每次复现 run,
    # train/val 边界在悄悄变 —— 你报告"从 89% 涨到 91%", 可能只是分的运气变了!
    # 更精细姿势: random_split(ds, sizes, generator=torch.Generator().manual_seed(42))
    #   (generator 独立随机源, 不影响别的代码的随机流, 工程上更干净)

    # ========================================================================
    # 3. tifa 事故 + 分层划分修复
    # ========================================================================
    print("\n" + "=" * 70)
    print("【实验 3】分层划分: 小类事故 vs 分层修复")
    val_random = class_counts(val_all, ds)
    print("  随机 20% 打分: 各人 val 占比 (注意小类):")
    for name, idx in sorted(ds.class_to_idx.items(), key=lambda kv: kv[1]):
        full = cnt_by_class[idx]
        got = val_random.get(idx, 0)
        print(f"    [{idx:>2}] {name:20s} val={got:>3}  占 {got/full*100:5.1f}%")
    print("  症状: 小类(如 tifa 79 张)的占比在 15%~21% 间抖动; 再小一些的类(数据不足时)")
    print("       甚至可能切出 0 张的 val -> 该人永远不被体检, 死了都不知道。")

    # ---- 修复: 分层抽样(Stratified) ----
    # 原理: 按类别把人分组, 每组内【分别】按 20% 切 val —— 每类严格 20%。
    #   在分类问题里就是"每班各选 20% 当监考", 而不是"全校抽 20%"。
    bucket = {}                                    # {label: [索引...]}
    for i, (_, label) in enumerate(ds.images):
        bucket.setdefault(label, []).append(i)

    val_idx, train_idx = [], []
    for label, idxs in sorted(bucket.items()):
        n_val = int(len(idxs) * VAL_RATIO + 0.5)    # 四舍五入(79*0.2=15.8 -> 16)
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    # Subset 只是索引集合: 这两个列表互相不重叠, 且各自覆盖所有人
    val_strat = Subset(ds, sorted(val_idx))
    train_strat = Subset(ds, sorted(train_idx))
    print("\n  分层划分后:")
    print_distribution(ds, val_strat, "val(分层)")
    print_distribution(ds, train_strat, "train(分层)")
    print(f"  注意: 分层后 val 总量 {len(val_strat)} 可能和 random_split 的 {len(val_all)} 差几张,")
    print("        这是逐类四舍五入的累计差, 合理范围, 不要迷信总数一致。")

    # ========================================================================
    # 4. 项目正式接口 get_datasets(为 2.x 章节预留)
    # ========================================================================
    def get_datasets(root="./data", val_ratio=VAL_RATIO, seed=42,
                     train_transform=None, val_transform=None):
        """
        返回 (train_ds, val_ds), 分层划分 + 固定种子, 直接喂训练循环。
        train/val 的 transform 未来会不同(train 增强, val 纯净), 所以分开收参。
        (第 3.2 节: train 加 RandomHorizontalFlip/ColorJitter 等增强时,
         只改 train_transform, val 永远不动 —— 裁判不能吃兴奋剂。)
        """
        import numpy as np
        rng = np.random.default_rng(seed)            # 独立于 torch 的随机源
        ds_ = PersonDataset(data_dir=root, transform=train_transform)
        idxs = np.arange(len(ds_))
        rng.shuffle(idxs)                            # 类内洗牌实现"随机落到 train/val"
        idxs = idxs.tolist()                         # 转回 python 列表

        bucket = {}
        for i in idxs:
            bucket.setdefault(ds_.images[i][1], []).append(i)
        v_idx = []
        for label, lst in sorted(bucket.items()):
            v_idx.extend(lst[:int(len(lst) * val_ratio + 0.5)])
        val_set = Subset(ds_, sorted(v_idx))
        train_set = Subset(ds_, sorted(set(idxs) - set(v_idx)))
        # 当前两人共用同一 transform; 第 3.2 节引入训练增强时, 需要各自独立
        # PersonDataset(get_datasets 再扩展), val 永远拿"无增强"版本 —— 裁判不能吃兴奋剂。
        return train_set, val_set

    print("\n" + "=" * 70)
    print("【实验 4】get_datasets(): 项目正式接口自检")
    tr, va = get_datasets()
    print(f"  train={len(tr)}, val={len(va)}, 同执行两次:")
    tr2, va2 = get_datasets()
    print(f"  train2={len(tr2)}, val2={len(va2)}")
    same = class_counts(va, ds) == class_counts(va2, ds)
    print(f"  两次 val 按类分布一致? {same}  (不看顺序, 只看人数账本)")

    # ========================================================================
    # 5. 思考题答案
    # ========================================================================
    # Q: 为什么深度学习不用 K 折交叉验证?
    # A: 1) 成本: K 折要把模型训练 K 次, 我们的训练一次就 5-10 分钟, K=5 就是 25-50 分钟;
    #      换 50 个超参组合 = 100+ 小时, 直接指数爆炸。
    #    2) 职责: 我们的 val 只干两件事 - 防过拟合信号(3.1) + 挑 best model(2.5),
    #      一个 20% 的固定 val 足够提供稳定的信号, 不必为了"少浪费 20% 数据"
    #      去付 K 倍成本(我们本身数据 4k 张不少)。
    #    3) K 折属于"统计学严谨派"(验证集误差估计的方差更小), 深度学习是
    #      "实验快速迭代派", 纪律不同: 快、稳、能复现 > 无偏。
