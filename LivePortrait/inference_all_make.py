import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import subprocess
import random
from pathlib import Path


def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_latest_mp4_file(directory):
    """
    从某个目录中获取最新生成的 mp4 文件路径。
    如果你的 inference.py 只生成单个 mp4，可以用这个函数来定位并重命名。
    如若生成多个文件，需要根据实际需求做进一步过滤。
    """
    mp4_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.mp4')]
    if not mp4_files:
        return None, None
    # 按修改时间排序，获取最新的
    # mp4_files.sort(key=os.path.getmtime, reverse=True)
    # return mp4_files[0]
        # 分类视频文件
    concat_video = None
    normal_video = None
    for video in mp4_files:
        if '_concat.mp4' in video:
            concat_video = video
        else:
            normal_video = video

    return normal_video, concat_video


def process_images(base_path, device_id):
    # DeepFaceLab/output 目录
    deeplabf_output = f"{base_path}/DeepFaceLab/output"
    # 参考视频目录
    ref_video_dir = f"{base_path}/DeepFaceLab/ref_video"
    # LivePortrait 脚本所在目录（需要在此目录下执行 python inference.py）
    liveportrait_path = f"{base_path}/LivePortrait"

    # 5 个参考视频
    ref_videos = [
        "angry_512.mp4",
        "no_512.mp4",
        "query_512.mp4",
        "smile_512.mp4",
        "yes_512.mp4"
    ]

    # 获取所有输出目录（例如 d82LKPNesE8_100 等）
    output_dirs = [
        d for d in os.listdir(deeplabf_output)
        if os.path.isdir(os.path.join(deeplabf_output, d))
    ]

    print(f"找到 {len(output_dirs)} 个输出目录: {output_dirs}")
    # 先不计算总任务量，因为 HQ_face + photomaker 是组合的，这里仅做提示

    for dir_name in output_dirs:
        current_output_dir = os.path.join(deeplabf_output, dir_name)
        hq_face_dir = os.path.join(current_output_dir, "HQ_face")
        photomaker_dir = os.path.join(current_output_dir, "photomaker")

        # --- 1) 从 HQ_face 里获取第一张图（若存在） ---
        hq_face_image = None
        if os.path.exists(hq_face_dir):
            face_images = [f for f in os.listdir(hq_face_dir) if f.endswith('.jpg')]
            face_images.sort()  # 确保顺序
            if face_images:
                # 使用第一张作为源图
                hq_face_image = os.path.join(hq_face_dir, face_images[0])
            else:
                print(f"[警告] {hq_face_dir} 中没有找到jpg文件，将跳过 HQ_face。")

        # --- 2) 从 photomaker/scene_0~5 里各随机选 1 张 ---
        photomaker_images = []
        if os.path.exists(photomaker_dir):
            for scene_i in range(6):
                scene_dir = os.path.join(photomaker_dir, f"scene_{scene_i}")
                if os.path.isdir(scene_dir):
                    candidate_imgs = [
                        f for f in os.listdir(scene_dir) if f.endswith('.jpg')
                    ]
                    if candidate_imgs:
                        chosen_img = random.choice(candidate_imgs)  # 随机选一张
                        photomaker_images.append(os.path.join(scene_dir, chosen_img))
                    else:
                        print(f"[警告] {scene_dir} 中没有 jpg 文件，跳过此 scene")
                else:
                    print(f"[提示] {scene_dir} 不存在，跳过")
        else:
            print(f"[提示] photomaker 目录不存在: {photomaker_dir}")

        # 把 HQ_face 的那一张 + photomaker 选中的 6 张 合并
        source_images = []
        if hq_face_image:
            source_images.append(hq_face_image)
        source_images.extend(photomaker_images)
        similar_image_dir = os.path.join(current_output_dir, "similar_image")
        if os.path.isdir(similar_image_dir):
            sim_imgs = [f for f in os.listdir(similar_image_dir) if f.lower().endswith('.jpg')]
            sim_imgs.sort()
            # 拼接完整路径
            sim_imgs_fullpath = [os.path.join(similar_image_dir, x) for x in sim_imgs]
            if sim_imgs_fullpath:
                source_images.extend(sim_imgs_fullpath)
                print(f"[提示] {dir_name} 的 similar_image 中找到 {len(sim_imgs_fullpath)} 张图片，将一并处理。")
            else:
                print(f"[提示] {dir_name} 的 similar_image 中没有发现 jpg 文件，跳过。")
        if not source_images:
            print(f"[警告] 在 {dir_name} 中既没找到 HQ_face 的图，也没找到 photomaker 的图，跳过此目录。")
            continue

        # --- 3) 为每个源图，执行 5 个参考视频 ---
        # 统一将结果放到 "5_movie_video" 目录下
        final_video_dir = os.path.join(current_output_dir, "5_movie_video")
        ensure_dir(final_video_dir)

        print(f"\n处理目录: {dir_name}")
        print(f"共找到 {len(source_images)} 张待处理的源图片（HQ_face + photomaker）")
        print(f"输出目录: {final_video_dir}")

        # 遍历每张源图
        for source_image in source_images:
            source_basename = os.path.splitext(os.path.basename(source_image))[0]
            # 例如 scene_0_output_2 或 HQ_face_0

            for video in ref_videos:
                ref_basename = os.path.splitext(video)[0]
                # 例如 angry_512, no_512

                # 为了避免不同任务文件互相覆盖，给每次推理都建一个临时输出目录
                temp_output_dir = os.path.join(current_output_dir, "moved_image_temp")
                ensure_dir(temp_output_dir)

                cmd = [
                    "python", "inference.py",
                    "-s", source_image,
                    "-d", os.path.join(ref_video_dir, video),
                    "--output-dir", temp_output_dir,
                    "--device-id", str(device_id)
                ]

                # 进入 LivePortrait 环境并执行
                try:
                    os.chdir(liveportrait_path)
                    print(f"\n[执行命令] {' '.join(cmd)}")
                    subprocess.run(cmd, check=True)

                    # 假设 inference.py 在 temp_output_dir 下会生成一个 mp4 文件
                    generated_mp4,_ = get_latest_mp4_file(temp_output_dir)
                    if generated_mp4 is None:
                        print(f"[警告] 未在 {temp_output_dir} 找到新生成的 mp4，跳过重命名。")
                    else:
                        # 构造新名字，例如: scene_0_output_2_no_512.mp4
                        final_name = f"{source_basename}_{ref_basename}.mp4"
                        final_path = os.path.join(final_video_dir, final_name)

                        # 重命名移动到 5_movie_video
                        os.rename(generated_mp4, final_path)
                        print(f"生成视频: {final_path}")

                except subprocess.CalledProcessError as e:
                    print(f"[错误] inference.py 处理失败: {e}")
                except Exception as e:
                    print(f"[异常] 发生错误: {e}")
                finally:
                    # 如果需要清理临时目录，可在这里做清理
                    pass

    print("\n所有处理全部完成！")


def main():
    # 基础路径和设备ID
    base_path = "/root/autodl-tmp/yufei"
    device_id = 1

    print("开始批量处理图片...")
    process_images(base_path, device_id)
    print("\n所有处理完成!")


if __name__ == "__main__":
    main()
