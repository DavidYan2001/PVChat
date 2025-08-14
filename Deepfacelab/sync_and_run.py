#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import paramiko

# ============ 配置区，根据实际情况修改 ============
HOST = "xxx"
PORT = 20002
USERNAME = "xxx"
PASSWORD = "xxxx"

# 本地输出目录（DeepFaceLab 输出结果所在目录）
LOCAL_OUTPUT_DIR = "/root/autodl-tmp/yufei/DeepFaceLab/output"

# 远程服务器上放置 HQ_face 目录的父目录
REMOTE_TMP_PICTURE = "/mnt/hdd1/yufei/img2dataset/tmp_picture"

# 远程执行 clip-retrieval.py 的相关信息
REMOTE_CLIP_SCRIPT = "/mnt/hdd1/yufei/img2dataset/clip-retrieval.py"
#REMOTE_CLIP_SCRIPT = "/mnt/hdd1/yufei/LAION-Face/clip_retrieve.py"
# 如你的 conda activate 路径有变化，或是用 'conda run -n clip python xxx'，请改为对应命令
REMOTE_CLIP_ENV_ACTIVATE = "/home/yufei/miniconda3/bin/activate clip"
REMOTE_CLIP_RUN_DIR = "/mnt/hdd1/yufei/img2dataset"

# ============ 辅助函数 ============

def is_image_file(filename):
    """判断文件是否为图片（支持常见图片格式）"""
    return filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))

def sftp_mkdir_p(sftp, remote_directory):
    """
    模拟 'mkdir -p'：逐级创建远程目录。
    """
    parts = remote_directory.strip("/").split("/")
    path = ""
    for p in parts:
        path = path + "/" + p
        try:
            sftp.stat(path)
        except FileNotFoundError:
            try:
                sftp.mkdir(path)
            except Exception as e:
                print(f"[WARNING] mkdir failed: {e}")

def sftp_copy_dir_to_remote(sftp, local_dir, remote_dir):
    """
    递归上传本地 local_dir -> 远程 remote_dir（不存在则自动创建）
    """
    for root, dirs, files in os.walk(local_dir):
        rel_path = os.path.relpath(root, local_dir)
        remote_path = os.path.join(remote_dir, rel_path).replace("\\", "/")

        # mkdir -p
        sftp_mkdir_p(sftp, remote_path)

        for file in files:
            local_file = os.path.join(root, file)
            remote_file = os.path.join(remote_path, file).replace("\\", "/")
            sftp.put(local_file, remote_file)
            print(f"[UPLOAD] {local_file} -> {remote_file}")

def sftp_copy_dir_to_local(sftp, remote_dir, local_dir):
    """
    递归下载远程 remote_dir -> 本地 local_dir（不存在则自动创建）
    """
    def _stat_is_dir(st):
        return (st.st_mode & 0o40000) == 0o40000  # S_IFDIR

    def _recursive_download(r_dir, l_dir):

        # print(f"[DEBUG] Checking existence of remote directory: {remote_sim_dir}")
        # try:
        #     sftp.listdir(remote_sim_dir)
        # except FileNotFoundError:
        #     print(f"[ERROR] Remote directory does not exist: {remote_sim_dir}")
        #     return  # 或者继续处理其他步骤

        os.makedirs(l_dir, exist_ok=True)
        for attr in sftp.listdir_attr(r_dir):
            remote_item = r_dir + "/" + attr.filename
            local_item = os.path.join(l_dir, attr.filename)

            if _stat_is_dir(attr):
                _recursive_download(remote_item, local_item)
            else:
                sftp.get(remote_item, local_item)
                print(f"[DOWNLOAD] {remote_item} -> {local_item}")

    _recursive_download(remote_dir, local_dir)

def main():
    # 1) 建立 SSH 连接 & SFTP
    print(f"[INFO] Connecting to {HOST}:{PORT} ...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=HOST, port=PORT, username=USERNAME, password=PASSWORD)
    sftp = client.open_sftp()
    print("[INFO] SSH connection established.")

    # 2) 上传所有需要处理的 HQ_face 目录（若本地 already 存在 similar_image 且图片数>=5 则跳过）
    print("[INFO] Start uploading all HQ_face directories ...")
    for video_name in os.listdir(LOCAL_OUTPUT_DIR):
        local_video_dir = os.path.join(LOCAL_OUTPUT_DIR, video_name)
        if not os.path.isdir(local_video_dir):
            continue

        # 检查是否存在 similar_image 文件夹且图片数 >= 5
        local_sim_dir = os.path.join(local_video_dir, "similar_image")
        if os.path.isdir(local_sim_dir):
            image_files = [f for f in os.listdir(local_sim_dir) if is_image_file(f)]
            if len(image_files) >= 5:
                print(f"[INFO] {video_name} already has similar_image with >=5 images. Skipping upload and processing.")
                continue

        # HQ_face 目录
        local_hq_face = os.path.join(local_video_dir, "HQ_face")
        if not os.path.isdir(local_hq_face):
            continue

        # 远程对应目录
        remote_video_dir = os.path.join(REMOTE_TMP_PICTURE, video_name).replace("\\", "/")
        remote_hq_face = os.path.join(remote_video_dir, "HQ_face").replace("\\", "/")

        print(f"\n[INFO] Uploading: {local_hq_face} => {remote_hq_face}")
        sftp_copy_dir_to_remote(sftp, local_hq_face, remote_hq_face)

    # 3) 在远程服务器执行 clip-retrieval.py 脚本（一次性执行）
    print("\n[INFO] Running clip-retrieval.py on remote server ...")
    command = f"""
        source {REMOTE_CLIP_ENV_ACTIVATE}
        cd {REMOTE_CLIP_RUN_DIR}
        python {REMOTE_CLIP_SCRIPT}
    """
    stdin, stdout, stderr = client.exec_command(command)
    exit_status = stdout.channel.recv_exit_status()

    print("[REMOTE STDOUT]\n" + stdout.read().decode('utf-8'))
    print("[REMOTE STDERR]\n" + stderr.read().decode('utf-8'))
    if exit_status != 0:
        print(f"[ERROR] Remote script exited with status {exit_status}")
    else:
        print("[INFO] Remote script finished successfully.")

    # 4) 下载远程生成的 similar_image 目录到本地（跳过本地 already 存在 similar_image 且图片数>=5 的文件夹）
    print("[INFO] Downloading similar_image directories back to local ...")
    for video_name in os.listdir(LOCAL_OUTPUT_DIR):
        local_video_dir = os.path.join(LOCAL_OUTPUT_DIR, video_name)
        if not os.path.isdir(local_video_dir):
            continue

        # 如果本地 similar_image 存在且图片数>=5，则跳过下载
        local_sim_dir = os.path.join(local_video_dir, "similar_image")
        if os.path.isdir(local_sim_dir):
            image_files = [f for f in os.listdir(local_sim_dir) if is_image_file(f)]
            if len(image_files) >= 5:
                print(f"[INFO] {video_name} already has similar_image with >=5 images. Skipping download.")
                continue
        local_HQ_face_path=os.path.join(local_video_dir, "HQ_face")
        if not os.path.isdir(local_HQ_face_path):
            continue

        # 远程 similar_image 路径
        remote_sim_dir = os.path.join(REMOTE_TMP_PICTURE, video_name, "HQ_face", "similar_image").replace("\\", "/")
        print(f"\n[INFO] Downloading: {remote_sim_dir} => {local_sim_dir}")
        sftp_copy_dir_to_local(sftp, remote_sim_dir, local_sim_dir)

    # 5) 关闭连接
    sftp.close()
    client.close()
    print("\n[INFO] All done.")

if __name__ == "__main__":
    main()
