#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import paramiko

# ============ Configuration area, modify according to the actual situation ============
HOST = "xxx"
PORT = 20002
USERNAME = "xxx"
PASSWORD = "xxxx"

# Local output directory (the directory where DeepFaceLab's output results are located)
LOCAL_OUTPUT_DIR = "/root/autodl-tmp/yufei/DeepFaceLab/output"

# Place the parent directory of the HQ face directory on the remote server
REMOTE_TMP_PICTURE = "/mnt/hdd1/yufei/img2dataset/tmp_picture"

# Remotely execute the relevant information of Clip-retriev.py
REMOTE_CLIP_SCRIPT = "/mnt/hdd1/yufei/img2dataset/clip-retrieval.py"
#REMOTE_CLIP_SCRIPT = "/mnt/hdd1/yufei/LAION-Face/clip_retrieve.py"
# If your conda activate path changes or you use 'conda run -n clip python xxx', please change to the corresponding command
REMOTE_CLIP_ENV_ACTIVATE = "/home/yufei/miniconda3/bin/activate clip"
REMOTE_CLIP_RUN_DIR = "/mnt/hdd1/yufei/img2dataset"

# ============auxiliary function============

def is_image_file(filename):
    """Determine whether the file is an image (supporting common image formats)"""
    return filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))

def sftp_mkdir_p(sftp, remote_directory):
    """
    Simulate 'mkdir -p' : Create remote directories step by step.
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
    Recursively upload local local_dir -> remote remote_dir (automatically created if not present)
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
    Recursively download remote remote_dir -> local local_dir (automatically created if not present)
    """
    def _stat_is_dir(st):
        return (st.st_mode & 0o40000) == 0o40000  # S_IFDIR

    def _recursive_download(r_dir, l_dir):
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
    # 1) Establish an SSH connection & SFTP
    print(f"[INFO] Connecting to {HOST}:{PORT} ...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=HOST, port=PORT, username=USERNAME, password=PASSWORD)
    sftp = client.open_sftp()
    print("[INFO] SSH connection established.")

    # 2) Upload all the HQ_face directories that need to be processed (skip if there are already similar_images locally and the number of images is >=5)
    print("[INFO] Start uploading all HQ_face directories ...")
    for video_name in os.listdir(LOCAL_OUTPUT_DIR):
        local_video_dir = os.path.join(LOCAL_OUTPUT_DIR, video_name)
        if not os.path.isdir(local_video_dir):
            continue

        # Check if there is a "similar_image" folder and the number of images is greater than or equal to 5
        local_sim_dir = os.path.join(local_video_dir, "similar_image")
        if os.path.isdir(local_sim_dir):
            image_files = [f for f in os.listdir(local_sim_dir) if is_image_file(f)]
            if len(image_files) >= 5:
                print(f"[INFO] {video_name} already has similar_image with >=5 images. Skipping upload and processing.")
                continue

        # "HQ face" directory
        local_hq_face = os.path.join(local_video_dir, "HQ_face")
        if not os.path.isdir(local_hq_face):
            continue

        # Remote corresponding directory
        remote_video_dir = os.path.join(REMOTE_TMP_PICTURE, video_name).replace("\\", "/")
        remote_hq_face = os.path.join(remote_video_dir, "HQ_face").replace("\\", "/")

        print(f"\n[INFO] Uploading: {local_hq_face} => {remote_hq_face}")
        sftp_copy_dir_to_remote(sftp, local_hq_face, remote_hq_face)

    # 3) Execute the clip-retrieval.py script on the remote server (in one go)
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

    # 4) Download the remotely generated similar_image directory to your local device (skip the folder where similar_image already exists locally and the number of images is >=5)
    print("[INFO] Downloading similar_image directories back to local ...")
    for video_name in os.listdir(LOCAL_OUTPUT_DIR):
        local_video_dir = os.path.join(LOCAL_OUTPUT_DIR, video_name)
        if not os.path.isdir(local_video_dir):
            continue

        # If the local similar_image exists and the number of images is greater than or equal to 5, skip the download
        local_sim_dir = os.path.join(local_video_dir, "similar_image")
        if os.path.isdir(local_sim_dir):
            image_files = [f for f in os.listdir(local_sim_dir) if is_image_file(f)]
            if len(image_files) >= 5:
                print(f"[INFO] {video_name} already has similar_image with >=5 images. Skipping download.")
                continue
        local_HQ_face_path=os.path.join(local_video_dir, "HQ_face")
        if not os.path.isdir(local_HQ_face_path):
            continue

        # Remote similar image path
        remote_sim_dir = os.path.join(REMOTE_TMP_PICTURE, video_name, "HQ_face", "similar_image").replace("\\", "/")
        print(f"\n[INFO] Downloading: {remote_sim_dir} => {local_sim_dir}")
        sftp_copy_dir_to_local(sftp, remote_sim_dir, local_sim_dir)

    # 5) Close the connection
    sftp.close()
    client.close()
    print("\n[INFO] All done.")

if __name__ == "__main__":
    main()
