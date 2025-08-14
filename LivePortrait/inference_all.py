import os
import subprocess
from pathlib import Path


def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    Path(directory).mkdir(parents=True, exist_ok=True)


def process_images(base_path, device_id):
    # 基础路径
    deeplabf_output = f"{base_path}/DeepFaceLab/output"
    ref_video_dir = f"{base_path}/DeepFaceLab/ref_video"
    liveportrait_path = f"{base_path}/LivePortrait"

    # 获取所有参考视频
    ref_videos = [
        "angry_512.mp4",
        "no_512.mp4",
        "query_512.mp4",
        "smile_512.mp4",
        "yes_512.mp4"
    ]

    # 获取所有输出目录
    output_dirs = [d for d in os.listdir(deeplabf_output)
                   if os.path.isdir(os.path.join(deeplabf_output, d))]

    total_tasks = len(output_dirs) * len(ref_videos)
    completed_tasks = 0

    print(f"找到 {len(output_dirs)} 个输出目录和 {len(ref_videos)} 个参考视频")
    print(f"总共需要处理 {total_tasks} 个任务")

    for dir_name in output_dirs:
        hq_face_dir = os.path.join(deeplabf_output, dir_name, "HQ_face")

        # 检查HQ_face目录是否存在
        if not os.path.exists(hq_face_dir):
            print(f"警告: {hq_face_dir} 不存在，跳过此目录")
            continue

        # 获取该目录下的所有jpg文件
        face_images = [f for f in os.listdir(hq_face_dir) if f.endswith('.jpg')]

        if not face_images:
            print(f"警告: {hq_face_dir} 中没有找到jpg文件，跳过此目录")
            continue

        # 使用第一张图片作为源图片
        source_image = os.path.join(hq_face_dir, face_images[0])

        # 处理每个参考视频
        for video in ref_videos:
            completed_tasks += 1
            print(f"\n处理进度: {completed_tasks}/{total_tasks}")
            print(f"正在处理目录: {dir_name}")
            print(f"参考视频: {video}")

            # 创建输出目录
            output_dir = os.path.join(deeplabf_output, dir_name, "moved_image")
            ensure_dir(output_dir)

            # 构建完整的命令
            cmd = [
                "python", "inference.py",
                "-s", source_image,
                "-d", os.path.join(ref_video_dir, video),
                "--output-dir", output_dir,
                "--device-id", str(device_id)
            ]

            # 切换到LivePortrait目录并执行命令
            try:
                os.chdir(liveportrait_path)
                print(f"执行命令: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
                print(f"成功处理 {dir_name} 的图片与视频 {video}")
            except subprocess.CalledProcessError as e:
                print(f"处理失败: {e}")
                print(f"跳过当前任务并继续下一个")
            except Exception as e:
                print(f"发生错误: {e}")
                print(f"跳过当前任务并继续下一个")


def main():
    # 基础路径和设备ID
    base_path = "/root/autodl-tmp/yufei"
    device_id = 1

    print("开始批量处理图片...")
    process_images(base_path, device_id)
    print("\n所有处理完成!")


if __name__ == "__main__":
    main()