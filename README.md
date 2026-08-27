# PyTorch 人物识别项目（从零学习到迁移学习）

一个人物识别（11 人分类，4384 张图片）的完整深度学习教学项目：从**数据体检**到
**数据管线**、**手写 CNN**、**训练循环**、**过拟合诊断**、**正则化抗战**，
最终以 **ResNet18 迁移学习**收尾——每章代码均为"教案级注释"，学习进度同步在线上站点。

## 最终成绩单（全部数值自带实验日志）

| 路线 | 模型 | 可训练参数 | 验证准确率 | 备注 |
|---|---|---|---|---|
| 2.4 从头训练 | PersonCNN (67M) | 67,208,267 | **85.18%** | 训练 100% vs 验证 85%：过拟合现形 |
| 3.2 正则化+增强 | PersonCNN + 四件套 | 67,208,267 | **84.61%** | 背答案推迟 2 倍（16 epochs），鸿沟 15.4→8.0pp |
| **4.1 迁移学习** | **ResNet18 + 6 千参数新头** | **5,643 (0.05%)** | **87.57%** ✅ | 训练 88.5% vs 验证 87.6%：过拟合清零 |

**结论一句话**：68M 参数对 4384 张图是容量过剩；站在 ImageNet 数据上的肩膀（预训练骨干）
只需要 5,643 个可训练参数就赢下全程，而且没有过拟合。

## 数据

- 11 个类别（人物），4384 张可用图（jpg/png，97.4% PNG，竖幅为主，宽 720~2732px）
- 类别不均衡：tifa 仅 79 张 vs 洛瑶 763 张（近 10 倍）
- 数据清洗：5 个异常文件（3 HEIC + 1 txt + 1 损坏 PNG）已隔离至 `data_quarantine/`
- 数据本体（大文件）不入仓库（见 `.gitignore`）

## 环境

- Python 3.12.7 · PyTorch 2.11.0+cu128 · NVIDIA RTX 5070 (CUDA 可用) · Windows
- matplotlib（画曲线/矩阵图）

## 目录结构

```
├── scripts/                  # 教学代码（每章一脚本, 教案级注释）
│   ├── inspect_data.py       # 0.1 数据体检（单次扫描档案 + verify 深校验）
│   ├── clean_data.py         # 0.1 隔离式清洗（移动不删除）
│   ├── basics.py             # 0.2 最小 PyTorch 程序（tensor/GPU/autograd）
│   ├── person_dataset.py     # 1.1 手写 Dataset（教学级注释）
│   ├── transforms_demo.py    # 1.2 transforms（双管线等价性实验 0.00e+00）
│   ├── dataloader_demo.py    # 1.3 DataLoader（四参数全实验）
│   ├── split_dataset.py      # 1.4 分层划分 + get_datasets() 正式接口
│   ├── person_cnn.py         # 2.1 手写 CNN（参数量对账 67M/感受野 36x36）
│   ├── loss_demo.py          # 2.2 CrossEntropyLoss 五实验
│   ├── optimizer_demo.py     # 2.3 优化器（lr 三张脸/梯度累积/momentum 翻脊）
│   ├── train_cnn.py          # 2.4 训练循环（首战记录 → runs/history.json）
│   ├── eval_best.py          # 2.5 state_dict 解剖/复活/逐类体检
│   ├── plot_curves.py        # 3.1 曲线医生（拐点检测+鸿沟计算）
│   ├── train_aug.py          # 3.2 四武器正则化（真实统计量+增强+WD+Dropout）
│   ├── confusion_matrix.py   # 3.3 混淆矩阵+错题集（酥酥→女主 x15 榜首）
│   ├── transfer.py           # 4.1 ResNet18 迁移学习（冻结骨干+换头）
│   └── predict.py            # 5.1 推理 CLI（单图 top-3）
├── runs/                     # 训练产物（*.pt 大文件 gitignore）
│   ├── history*.json         #   每 epoch 指标（曲线原料）
│   ├── curves.png            #   训练曲线诊断图（3.1）
│   ├── confusion_matrix.png  #   混淆矩阵热力图（3.3）
│   └── wrong_grid.png        #   错题集蒙太奇（3.3）
├── docs/index.html           # 学习站点（GitHub Pages 发布）
└── README.md
```

## 快速开始

```powershell
python scripts/inspect_data.py        # 数据体检
python scripts/person_dataset.py      # Dataset 自检
python scripts/dataloader_demo.py     # DataLoader 演示（分层子集)
python scripts/train_cnn.py           # 重新训练从头 CNN（~15分钟, 8 epochs）
python scripts/eval_best.py           # 复活 best 并体检
python scripts/train_aug.py           # 正则化版本（16 epochs）
python scripts/transfer.py            # ResNet18 迁移（6 epochs, 推荐）
python scripts/predict.py --demo 5    # 抽查 5 张
python scripts/predict.py --image <路径>   # 猜单张图
```

> 训练产物（`runs/*.pt`）因体积不入仓库；`runs/*.json|png` 保留供复现曲线与矩阵。

## 学习站点

每章学习记录（代码 + 笔记 + 踩坑日志）全部发布在 GitHub Pages：
**https://withtst.github.io/pytorch-person-recognition/**

## 教学大纲与进度

| 阶段 | 章节 | 标题 | 进度 |
|---|---|---|---|
| 0 | 0.1 | 数据体检与清洗（隔离式） | ✅ |
| 0 | 0.2 | 最小 PyTorch（tensor/GPU/autograd） | ✅ |
| 1 | 1.1 | 手写 Dataset | ✅ |
| 1 | 1.2 | transforms 管线（resize 地雷 → 等价性 0.00e+00） | ✅ |
| 1 | 1.3 | DataLoader（四参数 + 提速 1.65x） | ✅ |
| 1 | 1.4 | 数据划分（种子纪律 + 分层修复） | ✅ |
| 2 | 2.1 | 手写 CNN（67M 参数黑洞） | ✅ |
| 2 | 2.2 | CrossEntropyLoss（五实验 + 稳定性） | ✅ |
| 2 | 2.3 | Optimizer（lr 三张脸 + momentum 翻脊） | ✅ |
| 2 | 2.4 | 训练循环首战（过拟合实拍 100% vs 85.2%） | ✅ |
| 2 | 2.5 | state_dict 复活与逐类体检 | ✅ |
| 3 | 3.1 | 曲线诊断（拐点 Epoch 7 / 鸿沟 15.4pp） | ✅ |
| 3 | 3.2 | 四武器正则化（背答案推迟 2 倍） | ✅ |
| 3 | 3.3 | 混淆矩阵 + 错题集 | ✅ |
| 4 | 4.1 | 迁移学习 ResNet18（87.57% 收官） | ✅ |
| 5 | 5.1 | 推理 CLI（top-3 演示） | ✅ |
| 5 | 5.2 | 总结报告（本文件 + 站点收官） | ✅ |

**17 / 17 节全部完成** 🎓
