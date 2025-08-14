#!/usr/bin/env python3
import os
import sys
from typing import List, Dict, Tuple


def check_directories(base_path: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Check all video directories under the given path

    Args:
    base_path: The path of the output directory of DeepFaceLab

    Returns:
    Triplet (Video list without consisid_generation directory)
    A list of videos in an empty consisid_generation directory
    A normal list of video directories
    """
    # 存储结果的列表
    missing_dirs = []  # There is no "consisid generation" directory at all
    empty_dirs = []  # There is a directory but it is empty
    normal_dirs = []  # Normal directory

    try:
        # Get all the video folders under the output directory
        video_dirs = [d for d in os.listdir(base_path)
                      if os.path.isdir(os.path.join(base_path, d))]

        for video_dir in video_dirs:
            consisid_path = os.path.join(base_path, video_dir, "consisid_generation")

            # Check if the consisid generation directory exists
            if not os.path.exists(consisid_path):
                missing_dirs.append(video_dir)
                continue

            # Check if the directory is empty
            if not os.path.isdir(consisid_path):
                missing_dirs.append(video_dir)
                continue

            # Get the list of files in the directory
            files = os.listdir(consisid_path)

            if not files:
                empty_dirs.append(video_dir)
            else:
                normal_dirs.append(video_dir)

    except Exception as e:
        print(f"Error: An error occurred when checking the directory: {str(e)}")
        sys.exit(1)

    return missing_dirs, empty_dirs, normal_dirs


def print_statistics(missing_dirs: List[str], empty_dirs: List[str],
                     normal_dirs: List[str]) -> None:
    """Print statistical information"""
    total = len(missing_dirs) + len(empty_dirs) + len(normal_dirs)
    if total == 0:
        print("!")
        return

    problem_count = len(missing_dirs) + len(empty_dirs)
    problem_percentage = (problem_count / total) * 100

    print("\nstatistical information:")
    print(f"The total number of video directories: {total}")
    print(f"The number of problem directories: {problem_count} ({problem_percentage:.2f}%)")
    print(f"- The number of videos completely missing the consisid generation directory: {len(missing_dirs)}")
    print(f"- The number of videos in an empty consisid generation directory: {len(empty_dirs)}")
    print(f"Normal number of directories: {len(normal_dirs)} ({(len(normal_dirs) / total * 100):.2f}%)")

    if missing_dirs:
        print("\nVideos missing the consisid generation directory:")
        for dir_name in missing_dirs:
            print(f"- {dir_name}")

    if empty_dirs:
        print("\nThe video of the empty consisid generation directory:")
        for dir_name in empty_dirs:
            print(f"- {dir_name}")


def main():
    # Set the basic path
    base_path = "/root/autodl-tmp/yufei/DeepFaceLab/output"

    print(f"Start checking the catalogue: {base_path}")

    # Check whether the basic path exists
    if not os.path.exists(base_path):
        print(f"Error: The directory does not exist.: {base_path}")
        sys.exit(1)

    # Carry out the inspection
    missing_dirs, empty_dirs, normal_dirs = check_directories(base_path)

    # Print statistical information
    print_statistics(missing_dirs, empty_dirs, normal_dirs)


if __name__ == "__main__":
    main()