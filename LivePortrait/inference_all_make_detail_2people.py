import os
import subprocess
import random
import json
from pathlib import Path
import argparse
# 限制线程数
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"


def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_latest_mp4_file(directory):
    """
    从指定目录中获取最新生成的 mp4 文件路径。
    如果 inference.py 只生成单个 mp4 文件，可以用此函数定位后重命名。
    如若生成多个文件，需要根据实际情况做进一步过滤。
    """
    mp4_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.mp4')]
    if not mp4_files:
        return None, None
    # 分类视频文件
    concat_video = None
    normal_video = None
    for video in mp4_files:
        if '_concat.mp4' in video:
            concat_video = video
        else:
            normal_video = video
    return normal_video, concat_video


def process_face_folder(
    current_output_dir,
    hq_face_subdir,
    photomaker_subdir,
    similar_image_subdir,
    final_video_dir,
    ref_videos,
    ref_video_dir,
    liveportrait_path,
    device_id,
    suffix=""
):
    """
    封装出一个通用函数，处理单个人（或单个 face 文件夹）的图像：
    1. 从 hq_face_subdir 中选第一张
    2. 从 photomaker_subdir 的 scene_0-5 随机选 1 张
    3. 从 similar_image_subdir 里取所有图片
    4. 调用 inference.py 生成视频，并存到 final_video_dir
       （若 suffix="_2" 则输出视频名追加 "_2"）
    """
    hq_face_dir = os.path.join(current_output_dir, hq_face_subdir)
    photomaker_dir = os.path.join(current_output_dir, photomaker_subdir)
    similar_dir = os.path.join(current_output_dir, similar_image_subdir)

    # 1) HQ_face -> 第一张图片
    hq_face_image = None
    if os.path.exists(hq_face_dir):
        face_images = [f for f in os.listdir(hq_face_dir) if f.lower().endswith('.jpg')]
        face_images.sort()
        if face_images:
            hq_face_image = os.path.join(hq_face_dir, face_images[0])
        else:
            print(f"[警告] {hq_face_dir} 中没有找到 jpg 文件，将跳过 {hq_face_subdir}")
    else:
        print(f"[提示] 目录不存在: {hq_face_dir}")

    # 2) photomaker -> 随机场景
    photomaker_images = []
    if os.path.exists(photomaker_dir):
        for scene_i in range(6):
            scene_dir = os.path.join(photomaker_dir, f"scene_{scene_i}")
            if os.path.isdir(scene_dir):
                candidate_imgs = [f for f in os.listdir(scene_dir) if f.lower().endswith('.jpg')]
                if candidate_imgs:
                    chosen_img = random.choice(candidate_imgs)
                    photomaker_images.append(os.path.join(scene_dir, chosen_img))
                else:
                    print(f"[警告] {scene_dir} 中没有 jpg 文件，跳过此 scene")
            else:
                # 不报错，只提示
                pass
    else:
        print(f"[提示] {photomaker_subdir} 目录不存在: {photomaker_dir}")

    # 收集所有要处理的源图片
    source_images = []
    if hq_face_image:
        source_images.append(hq_face_image)

    source_images.extend(photomaker_images)

    # 3) similar_image -> 全部图片
    if os.path.isdir(similar_dir):
        sim_imgs = [f for f in os.listdir(similar_dir) if f.lower().endswith('.jpg')]
        sim_imgs.sort()
        sim_imgs_fullpath = [os.path.join(similar_dir, x) for x in sim_imgs]
        if sim_imgs_fullpath:
            source_images.extend(sim_imgs_fullpath)
            print(f"[提示] 在 {similar_image_subdir} 中找到 {len(sim_imgs_fullpath)} 张图片，将一并处理。")
        else:
            print(f"[提示] {similar_image_subdir} 中没有发现 jpg 文件，跳过。")
    else:
        print(f"[提示] {similar_image_subdir} 不存在: {similar_dir}")

    if not source_images:
        print(f"[警告] 在 {hq_face_subdir}, {photomaker_subdir}, {similar_image_subdir} 中均未找到图片，跳过。")
        return

    # 4) 依次调用 inference.py 生成视频
    print(f"\n[处理] 子目录: {hq_face_subdir} (suffix={suffix})")
    print(f" 共有 {len(source_images)} 张待处理的源图片")
    for source_image in source_images:
        source_basename = os.path.splitext(os.path.basename(source_image))[0]
        temp_output_dir = os.path.join(current_output_dir, "moved_image_temp")
        ensure_dir(temp_output_dir)

        for video in ref_videos:
            ref_basename = os.path.splitext(video)[0]
            cmd = [
                "python", "inference.py",
                "-s", source_image,
                "-d", os.path.join(ref_video_dir, video),
                "--output-dir", temp_output_dir,
                "--device-id", str(device_id)
            ]
            try:
                os.chdir(liveportrait_path)
                print(f"\n[执行命令] {' '.join(cmd)}")
                subprocess.run(cmd, check=True)

                generated_mp4, _ = get_latest_mp4_file(temp_output_dir)
                if generated_mp4 is None:
                    print(f"[警告] 未在 {temp_output_dir} 找到新生成的 mp4 文件，跳过重命名。")
                else:
                    # e.g.  HQ_00004_0_angry_512.mp4
                    # 如果 suffix="_2" => HQ_00004_0_angry_512_2.mp4
                    final_name = f"{source_basename}_{ref_basename}{suffix}.mp4"
                    final_path = os.path.join(final_video_dir, final_name)
                    os.rename(generated_mp4, final_path)
                    print(f"生成视频: {final_path}")
            except subprocess.CalledProcessError as e:
                print(f"[错误] inference.py 处理失败: {e}")
            except Exception as e:
                print(f"[异常] 发生错误: {e}")
            finally:
                # 如需清理 temp_output_dir 中的文件，可在此处添加逻辑
                pass


def process_images(base_path, device_id, train_json_path):
    # DeepFaceLab/output 目录
    deeplabf_output = f"{base_path}/DeepFaceLab/output"
    # 参考视频目录
    ref_video_dir = f"{base_path}/DeepFaceLab/ref_video"
    # LivePortrait 脚本所在目录（需要在此目录下执行 inference.py）
    liveportrait_path = f"{base_path}/LivePortrait"

    # 5 个参考视频
    ref_videos = [
        "angry_512.mp4",
        "no_512.mp4",
        "query_512.mp4",
        "smile_512.mp4",
        "yes_512.mp4"
    ]

    # 读取 train.json 文件，筛选出 is_positive 为 True 的视频
    try:
        with open(train_json_path, 'r') as f:
            train_data = json.load(f)
    except Exception as e:
        print(f"[错误] 无法读取 {train_json_path}: {e}")
        return

    positive_entries = [entry for entry in train_data.get("data", []) if entry.get("is_positive")]
    if not positive_entries:
        print("[提示] train.json 中没有 is_positive 为 True 的视频。")
        return

    # 获取正向视频的基础文件名（去掉扩展名），用于匹配 DeepFaceLab/output 下的目录名
    positive_bases = [os.path.splitext(entry["video_name"])[0] for entry in positive_entries]
    print(f"[提示] 从 train.json 中找到 {len(positive_bases)} 个正向视频: {positive_bases}")

    # 获取所有输出目录（例如：d82LKPNesE8_100 等）
    all_output_dirs = [
        d for d in os.listdir(deeplabf_output)
        if os.path.isdir(os.path.join(deeplabf_output, d))
    ]
    # 仅保留目录名中包含正向视频基础名的文件夹
    output_dirs = []
    for d in all_output_dirs:
        for base in positive_bases:
            if base in d:
                output_dirs.append(d)
                break

    if not output_dirs:
        print("[提示] 没有找到与 train.json 中正向视频匹配的 DeepFaceLab/output 目录。")
        return

    print(f"找到 {len(output_dirs)} 个待处理输出目录: {output_dirs}")

    # 遍历每个输出目录进行处理
    for dir_name in output_dirs:
        current_output_dir = os.path.join(deeplabf_output, dir_name)
        final_video_dir = os.path.join(current_output_dir, "5_movie_video")
        ensure_dir(final_video_dir)

        print(f"\n========== 处理目录: {dir_name} ==========")

        # 第 1 人: HQ_face / photomaker / similar_image
        process_face_folder(
            current_output_dir=current_output_dir,
            hq_face_subdir="HQ_face",
            photomaker_subdir="photomaker",
            similar_image_subdir="similar_image",
            final_video_dir=final_video_dir,
            ref_videos=ref_videos,
            ref_video_dir=ref_video_dir,
            liveportrait_path=liveportrait_path,
            device_id=device_id,
            suffix=""
        )

        # 第 2 人: HQ_face2 / photomaker2 / similar_image2
        # 如果目录下确实存在这三个文件夹，则进行同样处理
        # 生成视频时在文件名后加 "_2"
        face2_dir = os.path.join(current_output_dir, "HQ_face2")
        photomaker2_dir = os.path.join(current_output_dir, "photomaker2")
        similar2_dir = os.path.join(current_output_dir, "similar_image2")

        if os.path.isdir(face2_dir) or os.path.isdir(photomaker2_dir) or os.path.isdir(similar2_dir):
            print(f"[提示] 检测到 {dir_name} 有 HQ_face2 / photomaker2 / similar_image2，处理第二人...")
            process_face_folder(
                current_output_dir=current_output_dir,
                hq_face_subdir="HQ_face2",
                photomaker_subdir="photomaker2",
                similar_image_subdir="similar_image2",
                final_video_dir=final_video_dir,
                ref_videos=ref_videos,
                ref_video_dir=ref_video_dir,
                liveportrait_path=liveportrait_path,
                device_id=device_id,
                suffix="_2"
            )
        else:
            print(f"[提示] {dir_name} 并未检测到 HQ_face2 等双人目录，跳过第二人处理。")

    print("\n所有处理全部完成！")
def get_args():
    parser = argparse.ArgumentParser(description="Build dataset for two people and both cases")

    parser.add_argument('--sks1', type=str, default='Sh',
                        help='Name of first person (default: Sh)')
    parser.add_argument('--sks2', type=str, default='Ho',
                        help='Name of second person (default: Ho)')
    return parser.parse_args()


def main():
    args = get_args()
    base_path = "/root/autodl-tmp/yufei"
    device_id = 1
    # 指定 train.json 文件路径
    #train_json = "/root/autodl-tmp/yufei/datasets/cekebv-hq/train_" + args.sks1 + '_' + args.sks2 + '.json'
    #test_json = "/root/autodl-tmp/yufei/datasets/cekebv-hq/test_" + args.sks1 + '_' + args.sks2 + '.json'
    train_json_path = f"{base_path}/datasets/cekebv-hq/train_" + args.sks1 + '_' + args.sks2 + '.json'
    print("开始批量处理图片...")
    process_images(base_path, device_id, train_json_path)
    print("\n所有处理完成!")


if __name__ == "__main__":
    main()
