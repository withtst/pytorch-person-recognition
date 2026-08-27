"""
数据体检脚本 - 第 0.1 节作业

核心改进（相对初版）：
1. 一次扫描生成"文件档案", 所有统计复用, 不再反复遍历磁盘
2. os.listdir 结果一律 sorted(), 保证输出可复现
3. 完整性检查用 img.verify()（真正读取像素数据）, 而不是只开文件头
4. 顶层只 import 一次, 消除函数内重复 import

用法:
    python scripts/inspect_data.py [--data-dir ./data] [--skip-verify]
"""
import argparse
import os
import sys
from collections import Counter

from PIL import Image

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}


def scan_dataset(data_dir):
    """一次遍历, 构建文件档案: [(人物名, 相对路径, 小写扩展名), ...]"""
    if not os.path.isdir(data_dir):
        sys.exit(f"错误: 目录不存在 {data_dir}")

    entries = []
    # sorted() -> 与文件系统的表现无关, 输出永远可复现
    for person in sorted(os.listdir(data_dir)):
        person_dir = os.path.join(data_dir, person)
        if not os.path.isdir(person_dir):
            continue
        for fname in sorted(os.listdir(person_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                entries.append((person, os.path.join(person_dir, fname), ext))
    return entries


def integrity_check(entries, verify=True):
    """
    检查每张图能否被 PIL 打开（verify: 是否再深度读像素）。
    返回 (尺寸统计, 异常列表)。异常列表元素: (人物名, 文件路径, 原因)
    """
    widths, heights = [], []
    ratio_info = []
    anomalies = []

    for person, path, ext in entries:
        try:
            with Image.open(path) as img:
                w, h = img.size          # 从文件头拿尺寸, 不读像素
                if verify:
                    img.verify()        # 真正读一遍文件内容, 检测损坏
                widths.append(w)
                heights.append(h)
                if h != 0:
                    ratio_info.append((path, w / h))
        except Exception as e:
            # verify() 之后的 Image.open 对象会失效, 这里直接记录为异常
            anomalies.append((person, path, str(e)))

    return widths, heights, ratio_info, anomalies


def print_counts(entries):
    """人物数量表 + 格式表 + 总数"""
    per_person = Counter(person for person, _, _ in entries)
    per_format = Counter(ext for _, _, ext in entries)

    print("=== 每人图片数量 ===")
    for person, count in sorted(per_person.items(), key=lambda kv: kv[0]):
        print(f"  {person}: {count} 张")
    print(f"\n总图片数: {len(entries)} 张")

    print("\n=== 每种图片格式数量 ===")
    for ext, count in sorted(per_format.items()):
        print(f"  {ext}: {count} 张")


def print_size_stats(entries, widths, heights, ratio_info):
    """尺寸统计 + 最极端宽高比"""
    print("\n=== 全数据集尺寸统计 ===")
    print(f"图片总数: {len(widths)}")
    print(f"宽度  -> min: {min(widths)}, max: {max(widths)}, avg: {sum(widths) / len(widths):.2f}")
    print(f"高度  -> min: {min(heights)}, max: {max(heights)}, avg: {sum(heights) / len(heights):.2f}")

    if ratio_info:
        widest, best = max(ratio_info, key=lambda x: x[1]), min(ratio_info, key=lambda x: x[1])
        print("\n=== 长宽比最极端的图片 ===")
        print(f"最'宽' (比例 {widest[1]:.2f}): {widest[0]}")
        print(f"最'高' (比例 {best[1]:.2f}): {best[0]}")
        most_extreme = max(ratio_info, key=lambda x: abs(x[1] - 1))
        print(f"离 1 最远 (比例 {most_extreme[1]:.2f}): {most_extreme[0]}")


def print_anomalies(entries, anomalies):
    """异常文件报告（按人物分组, 只报告不删除）"""
    print("\n=== 异常文件报告 ===")
    if not anomalies:
        print("OK: 全部图片均可正常打开（数字/图例输出不依赖 emoji, 兼容 GBK 终端）")
        return

    grouped = {}
    for person, path, reason in anomalies:
        grouped.setdefault(person, []).append((path, reason))

    print(f"共发现 {len(anomalies)} 个异常文件:\n")
    for person in sorted(grouped):
        print(f"📁 {person} ({len(grouped[person])} 个异常文件):")
        for path, reason in grouped[person]:
            print(f"   - {os.path.basename(path)}  [原因: {reason}]")
        print()
    print("提示: 本程序不执行任何删除/移动操作。")


def print_distribution(entries):
    """人数分布表 + 格式分布表"""
    per_person = Counter(person for person, _, _ in entries)
    by_count = Counter(per_person.values())
    total_people = len(per_person)

    print("\n=== 人员图片数分布表 ===")
    print(f"总人数: {total_people}, 总图片数: {len(entries)}")
    for num, people in sorted(by_count.items()):
        print(f"  拥有 {num:>3} 张图的人数: {people} 人 ({people / total_people * 100:5.1f}%)")

    per_format = Counter(ext for _, _, ext in entries)
    print("\n=== 图片格式分布表 ===")
    for ext, count in sorted(per_format.items()):
        print(f"  格式 {ext:<6} 数量: {count:>5} 张 ({count / len(entries) * 100:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="数据体检")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--skip-verify", action="store_true",
                        help="跳过像素级完整性校验(更快)")
    args = parser.parse_args()

    entries = scan_dataset(args.data_dir)
    print(f"找到 {len(set(p for p, _, _ in entries))} 个类别, 共 {len(entries)} 个图片文件\n")

    widths, heights, ratio_info, anomalies = integrity_check(entries, verify=not args.skip_verify)

    print_counts(entries)
    warn_count = len(anomalies)
    if widths:
        print_size_stats(entries, widths, heights, ratio_info)
    print_anomalies(entries, anomalies)
    print_distribution(entries)
    if warn_count:
        print(f"\n⚠️ 共 {warn_count} 个异常文件, 等待人工确认处理方案。")


if __name__ == "__main__":
    main()
