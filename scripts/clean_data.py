"""
数据清洗脚本 - 第 0.1 节作业

安全原则: 任何异常文件一律【移动】到 data_quarantine/（保留原命名和目录结构）,
          绝不删除。数据清洗的第一课: 先隔离, 确认无误后再谈删除。

判断标准:
  1. 扩展名不在 IMAGE_EXTENSIONS 中 -> 非图片格式 (如 .txt, .HEIC)
  2. PIL 无法打开 / verify() 失败  -> 损坏图片

用法:
    python scripts/clean_data.py [--data-dir ./data] [--quarantine-dir ./data_quarantine]
"""
import argparse
import os
import shutil
import sys
from collections import Counter

from PIL import Image

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}


def scan_all_files(data_dir):
    """遍历 data 目录下所有文件(不按扩展名过滤): [(人物名, 完整路径, 小写扩展名), ...]"""
    entries = []
    for person in sorted(os.listdir(data_dir)):
        person_dir = os.path.join(data_dir, person)
        if not os.path.isdir(person_dir):
            continue
        for fname in sorted(os.listdir(person_dir)):
            ext = os.path.splitext(fname)[1].lower()
            entries.append((person, os.path.join(person_dir, fname), ext))
    return entries


def classify(entries):
    """把文件分成三类: 正常 / 非图片格式 / 损坏"""
    normal, wrong_format, corrupted = [], [], []
    for person, path, ext in entries:
        if ext not in IMAGE_EXTENSIONS:
            wrong_format.append((person, path, ext))
            continue
        try:
            with Image.open(path) as img:
                # 打开成功即读入全部像素数据, 损坏(如截断)会在这里暴露
                img.verify()
            normal.append((person, path, ext))
        except Exception as e:
            corrupted.append((person, path, f"{ext} [verify 失败: {e}]"))
    return normal, wrong_format, corrupted


def quarantine(items, quarantine_root):
    """把 items 移入 quarantine_root/<人物名>/ 目录, 保持原名"""
    for person, path, _ in items:
        target_dir = os.path.join(quarantine_root, person)
        os.makedirs(target_dir, exist_ok=True)
        shutil.move(path, os.path.join(target_dir, os.path.basename(path)))


def main():
    parser = argparse.ArgumentParser(description="数据清洗(隔离式, 不删除)")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--quarantine-dir", default="./data_quarantine")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        sys.exit(f"错误: 目录不存在 {args.data_dir}")

    entries = scan_all_files(args.data_dir)
    print(f"扫描完成: {len(entries)} 个文件, 开始分类...")
    normal, wrong_format, corrupted = classify(entries)

    reported = Counter(person for person, _, _ in wrong_format)
    reported.update(person for person, _, _ in corrupted)

    if not wrong_format and not corrupted:
        print("✅ 未发现问题文件, 无需清洗。")
        return

    print(f"\n存在问题: 非图片格式 {len(wrong_format)} 个, 损坏图片 {len(corrupted)} 个")
    if reported:
        print("分布: " + ", ".join(f"{p}({n})" for p, n in sorted(reported.items())))

    quarantine(wrong_format + corrupted, args.quarantine_dir)
    print(f"\n已全部隔离到 {args.quarantine_dir}/ (供人工复核, 未删除任何数据)")

    after = scan_all_files(args.data_dir)
    normal_after = Counter(person for person, _, _ in classify(after)[0])
    print(f"\n=== 清理后 ===")
    print(f"剩余文件: {len(after)} 个, 其中正常图片 {sum(normal_after.values())} 张:")
    for person in sorted(normal_after):
        print(f"  {person}: {normal_after[person]} 张")


if __name__ == "__main__":
    main()
