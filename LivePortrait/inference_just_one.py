import os
import subprocess
from pathlib import Path
import argparse


def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    Path(directory).mkdir(parents=True, exist_ok=True)


def process_single_image(source_image, output_dir, ref_video_dir, liveportrait_path, device_id):
    """处理单张图片与所有参考视频"""
    # 获取所有参考视频
    ref_videos = [
        "angry_512.mp4",
        "no_512.mp4",
        "query_512.mp4",
        "smile_512.mp4",
        "yes_512.mp4"
    ]

    # 确保输出目录存在
    ensure_dir(output_dir)

    total_tasks = len(ref_videos)
    print(f"源图片: {source_image}")
    print(f"输出目录: {output_dir}")
    print(f"找到 {total_tasks} 个参考视频")

    for i, video in enumerate(ref_videos, 1):
        print(f"\n处理进度: {i}/{total_tasks}")
        print(f"参考视频: {video}")

        # 创建以视频名称命名的子目录
        video_output_dir = os.path.join(output_dir, os.path.splitext(video)[0])
        ensure_dir(video_output_dir)

        # 构建完整的命令
        cmd = [
            "python", "inference.py",
            "-s", source_image,
            "-d", os.path.join(ref_video_dir, video),
            "--output-dir", video_output_dir,
            "--device-id", str(device_id)
        ]

        # 切换到LivePortrait目录并执行命令
        try:
            os.chdir(liveportrait_path)
            print(f"执行命令: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            print(f"成功处理图片 {source_image} 与视频 {video}")
        except subprocess.CalledProcessError as e:
            print(f"处理失败: {e}")
            print(f"跳过当前任务并继续下一个")
        except Exception as e:
            print(f"发生错误: {e}")
            print(f"跳过当前任务并继续下一个")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='根据单张图片和参考视频生成动画')
    parser.add_argument('--source', '-s', required=True, help='源图片的路径')
    parser.add_argument('--output-dir', '-o', required=True, help='输出目录路径')
    parser.add_argument('--base-path', '-b', default='/root/autodl-tmp/yufei',
                        help='基础路径，包含LivePortrait和参考视频的目录')
    parser.add_argument('--device-id', '-d', type=int, default=1, help='设备ID')

    args = parser.parse_args()

    # 设置路径
    base_path = args.base_path
    ref_video_dir = f"{base_path}/DeepFaceLab/ref_video"
    liveportrait_path = f"{base_path}/LivePortrait"

    print("开始处理图片...")
    process_single_image(
        source_image=args.source,
        output_dir=args.output_dir,
        ref_video_dir=ref_video_dir,
        liveportrait_path=liveportrait_path,
        device_id=args.device_id
    )
    print("\n处理完成!")


if __name__ == "__main__":
    main()