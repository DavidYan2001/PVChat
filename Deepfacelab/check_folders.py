#!/usr/bin/env python3
import os
import sys
from typing import List, Dict, Tuple


def check_directories(base_path: str) -> Tuple[List[str], List[str], List[str]]:
    """
    检查给定路径下的所有视频目录

    Args:
        base_path: DeepFaceLab的output目录路径

    Returns:
        三元组 (无consisid_generation目录的视频列表,
               有空consisid_generation目录的视频列表,
               正常的视频目录列表)
    """
    # 存储结果的列表
    missing_dirs = []  # 完全没有consisid_generation目录的
    empty_dirs = []  # 有目录但是是空的
    normal_dirs = []  # 正常的目录

    try:
        # 获取output目录下的所有视频文件夹
        video_dirs = [d for d in os.listdir(base_path)
                      if os.path.isdir(os.path.join(base_path, d))]

        for video_dir in video_dirs:
            consisid_path = os.path.join(base_path, video_dir, "consisid_generation")

            # 检查consisid_generation目录是否存在
            if not os.path.exists(consisid_path):
                missing_dirs.append(video_dir)
                continue

            # 检查目录是否为空
            if not os.path.isdir(consisid_path):
                missing_dirs.append(video_dir)
                continue

            # 获取目录中的文件列表
            files = os.listdir(consisid_path)

            if not files:
                empty_dirs.append(video_dir)
            else:
                normal_dirs.append(video_dir)

    except Exception as e:
        print(f"Error: 检查目录时发生错误: {str(e)}")
        sys.exit(1)

    return missing_dirs, empty_dirs, normal_dirs


def print_statistics(missing_dirs: List[str], empty_dirs: List[str],
                     normal_dirs: List[str]) -> None:
    """打印统计信息"""
    total = len(missing_dirs) + len(empty_dirs) + len(normal_dirs)
    if total == 0:
        print("未找到任何视频目录!")
        return

    problem_count = len(missing_dirs) + len(empty_dirs)
    problem_percentage = (problem_count / total) * 100

    print("\n统计信息:")
    print(f"总视频目录数: {total}")
    print(f"问题目录数: {problem_count} ({problem_percentage:.2f}%)")
    print(f"- 完全缺失consisid_generation目录的视频数: {len(missing_dirs)}")
    print(f"- 空的consisid_generation目录的视频数: {len(empty_dirs)}")
    print(f"正常目录数: {len(normal_dirs)} ({(len(normal_dirs) / total * 100):.2f}%)")

    if missing_dirs:
        print("\n缺失consisid_generation目录的视频:")
        for dir_name in missing_dirs:
            print(f"- {dir_name}")

    if empty_dirs:
        print("\n空的consisid_generation目录的视频:")
        for dir_name in empty_dirs:
            print(f"- {dir_name}")


def main():
    # 设置基础路径
    base_path = "/root/autodl-tmp/yufei/DeepFaceLab/output"

    print(f"开始检查目录: {base_path}")

    # 检查基础路径是否存在
    if not os.path.exists(base_path):
        print(f"Error: 目录不存在: {base_path}")
        sys.exit(1)

    # 执行检查
    missing_dirs, empty_dirs, normal_dirs = check_directories(base_path)

    # 打印统计信息
    print_statistics(missing_dirs, empty_dirs, normal_dirs)


if __name__ == "__main__":
    main()