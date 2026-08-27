# PyTorch 人物识别项目（从零学习）

一个用于深度学习入门教学的完整项目：从**数据体检**开始，逐步手写 Dataset / 训练循环 / CNN，
最终实现"给一张照片 → 认出 TA 是谁"的 11 人分类器。全程以**培养 PyTorch 理解力**为目标，
而非直接调用黑盒 API。

## 数据

`data/` 下 11 个文件夹，每文件夹对应一名人物（文件夹名即人物名），共 **4384** 张可用图片
（jpg / png，尺寸 720~2732 像素宽，竖幅为主，97.4% PNG）。

> 数据文件体量大，已加入 `.gitignore` 不入仓库。

已完成的清洗（`scripts/clean_data.py`，隔离式，不删除）：
- 3 个 `.HEIC`（苹果格式，PIL 无法读取）
- 1 个 `.txt`
- 1 个损坏 PNG（`validate` 无法通过）

以上 5 个文件已移动到 `data_quarantine/`，保留了原始目录结构和命名。

## 环境

- Python 3.12.7
- PyTorch 2.11.0+cu128（CUDA 可用）
- Windows + PowerShell

## 目录结构

```
├── scripts/            # 数据管线脚本
│   ├── inspect_data.py # 数据体检（统计/尺寸/异常文件/分布）
│   └── clean_data.py   # 数据清洗（隔离异常文件）
├── docs/               # 学习站点（逐步构建, 最终 GitHub Pages 发布, 暂未开始）
├── data/               # 数据集（gitignore）
└── .gitignore
```

## 脚本用法

```powershell
python scripts/inspect_data.py                 # 体检（含像素级完整性校验）
python scripts/inspect_data.py --skip-verify   # 快速体检（只读文件头）
python scripts/clean_data.py                   # 隔离异常文件到 data_quarantine/
```

## 学习大纲与进度

| 阶段 | 章节 | 目标 | 进度 |
|---|---|---|---|
| 0 | 0.1 数据体检与清洗 | 用脚本看清数据, 学会隔离式清洗 | ✅ |
| 0 | 0.2 Git 仓库与最小 PyTorch 程序 | 工程化 + GPU 感知 tensor | ✅ |
| 1 | 1.1 手写 Dataset | __getitem__/__len__ 理解 | ✅ |
| 1 | 1.2 transforms | Resize/ToTensor/Normalize | ✅ |
| 1 | 1.3 DataLoader | batch 维度 B,C,H,W | ✅ |
| 1 | 1.4 数据划分 | train/val 分离 | ✅ |
| 2 | 2.1 手写 CNN | 卷积/池化/参数量 | ✅ |
| 2 | 2.2 CrossEntropyLoss | logits 与概率 | ✅ |
| 2 | 2.3 Optimizer | SGD/Adam/lr | ✅ |
| 2 | 2.4 训练循环 | forward/loss/backward/step | ✅ |
| 2 | 2.5 验证与模型保存 | state_dict/best model | ✅ |
| 3 | 3.1 训练曲线诊断 | 过拟合/欠拟合 | ✅ |
| 3 | 3.2 正则化与增强 | weight_decay/dropout/aug | ⬜ |
| 3 | 3.3 混淆矩阵 | 错误复盘 | ⬜ |
| 4 | 4.1 迁移学习(选做) | ResNet18 微调 | ⬜ |
| 5 | 5.1 推理 CLI | 单图预测 | ⬜ |
| 5 | 5.2 总结报告 | README 完善 + 复盘 | ⬜ |

## 学习站点

每完成一章，会向 `docs/` 追加一份学习记录页面（代码 + 笔记 + 心得），
已由 GitHub Pages 发布：

**https://withtst.github.io/pytorch-person-recognition/**

站点结构：
```
docs/
├── index.html          # 学习站点首页（进度 + 各章记录, 随章节持续更新）
└── (章节页面随进度追加)
```
