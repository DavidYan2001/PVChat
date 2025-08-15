#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import paramiko

# ============ Configuration area, modify according to the actual situation ============
HOST = "xxx"
PORT = 20002
USERNAME = "xxx"
PASSWORD = "xxx"

# Local output directory (the directory where DeepFaceLab's output results are located)
LOCAL_OUTPUT_DIR = "/root/autodl-tmp/yufei/DeepFaceLab/output"

# Place the parent directory of the HQ_face/HQ_face2 directory on the remote server
REMOTE_TMP_PICTURE = "/mnt/hdd1/yufei/img2dataset/tmp_picture"

# Remotely execute the relevant information of Clip-retriev.py
REMOTE_CLIP_SCRIPT = "/mnt/hdd1/yufei/img2dataset/clip-retrieval.py"
# If you use conda environment management, you need to modify the following command according to the actual situation
REMOTE_CLIP_ENV_ACTIVATE = "/home/yufei/miniconda3/bin/activate clip"
REMOTE_CLIP_RUN_DIR = "/mnt/hdd1/yufei/img2dataset"


def is_image_file(filename):
    """Determine whether the file is an image (supports common image formats)"""
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
            sftp.stat(path)  # Check if it already exists
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

    try:
        sftp.stat(remote_dir)  # If the remote directory does not exist, a FileNotFoundError will be thrown
    except FileNotFoundError:
        print(f"[WARNING] Remote directory not found: {remote_dir}")
        return

    _recursive_download(remote_dir, local_dir)


def main():
    # 1)Establish an SSH connection & SFTP
    print(f"[INFO] Connecting to {HOST}:{PORT} ...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=HOST, port=PORT, username=USERNAME, password=PASSWORD)
    sftp = client.open_sftp()
    print("[INFO] SSH connection established.")

    # 2) updata HQ_face & HQ_face2
    #    If the local already exists 'similar_image' (or 'similar_image2') and the number of images is greater than or equal to 5, the person's upload will be skipped
    print("[INFO] Start uploading all HQ_face / HQ_face2 directories ...")

    for video_name in os.listdir(LOCAL_OUTPUT_DIR):
        local_video_dir = os.path.join(LOCAL_OUTPUT_DIR, video_name)
        if not os.path.isdir(local_video_dir):
            continue

        # ---------------- First people HQ_face ----------------
        local_hq_face = os.path.join(local_video_dir, "HQ_face")
        if os.path.isdir(local_hq_face):
            local_sim_dir = os.path.join(local_video_dir, "similar_image")
            need_upload_1 = True  # Let's assume for now that an upload is needed

            if os.path.isdir(local_sim_dir):
                image_files = [f for f in os.listdir(local_sim_dir) if is_image_file(f)]
                if len(image_files) >= 5:
                    # If there are already more than or equal to 5 similar images, skip the upload
                    print(f"[INFO] {video_name} has similar_image (>=5), skip upload for HQ_face.")
                    need_upload_1 = False

            if need_upload_1:
                remote_video_dir = os.path.join(REMOTE_TMP_PICTURE, video_name).replace("\\", "/")
                remote_hq_face = os.path.join(remote_video_dir, "HQ_face").replace("\\", "/")
                print(f"\n[INFO] Uploading: {local_hq_face} => {remote_hq_face}")
                sftp_copy_dir_to_remote(sftp, local_hq_face, remote_hq_face)

        # ---------------- Second people HQ_face2 ----------------
        local_hq_face2 = os.path.join(local_video_dir, "HQ_face2")
        if os.path.isdir(local_hq_face2):
            local_sim_dir2 = os.path.join(local_video_dir, "similar_image2")
            need_upload_2 = True

            if os.path.isdir(local_sim_dir2):
                image_files2 = [f for f in os.listdir(local_sim_dir2) if is_image_file(f)]
                if len(image_files2) >= 5:
                    print(f"[INFO] {video_name} has similar_image2 (>=5), skip upload for HQ_face2.")
                    need_upload_2 = False

            if need_upload_2:
                remote_video_dir2 = os.path.join(REMOTE_TMP_PICTURE, video_name).replace("\\", "/")
                remote_hq_face2 = os.path.join(remote_video_dir2, "HQ_face2").replace("\\", "/")
                print(f"\n[INFO] Uploading: {local_hq_face2} => {remote_hq_face2}")
                sftp_copy_dir_to_remote(sftp, local_hq_face2, remote_hq_face2)

    # 3) Execute the clip-retrieval.py script on the remote server (in one go)
    print("\n[INFO] Running clip-retrieval.py on remote server ...")
    command = f"""
        source {REMOTE_CLIP_ENV_ACTIVATE}
        cd {REMOTE_CLIP_RUN_DIR}
        python {REMOTE_CLIP_SCRIPT}
    """
    stdin, stdout, stderr = client.exec_command(command)
    exit_status = stdout.channel.recv_exit_status()

    remote_out = stdout.read().decode('utf-8')
    remote_err = stderr.read().decode('utf-8')
    print("[REMOTE STDOUT]\n" + remote_out)
    print("[REMOTE STDERR]\n" + remote_err)
    if exit_status != 0:
        print(f"[ERROR] Remote script exited with status {exit_status}")
    else:
        print("[INFO] Remote script finished successfully.")

    # 4) Download the remotely generated similar_image/similar_image2 to your local device
    print("[INFO] Downloading similar_image / similar_image2 directories back to local ...")

    for video_name in os.listdir(LOCAL_OUTPUT_DIR):
        local_video_dir = os.path.join(LOCAL_OUTPUT_DIR, video_name)
        if not os.path.isdir(local_video_dir):
            continue

        # ----------------First people similar_image ----------------
        local_sim_dir = os.path.join(local_video_dir, "similar_image")
        sim_count = 0
        if os.path.isdir(local_sim_dir):
            sim_count = len([f for f in os.listdir(local_sim_dir) if is_image_file(f)])

        if sim_count < 5:
            remote_sim_dir = os.path.join(REMOTE_TMP_PICTURE, video_name, "HQ_face", "similar_image").replace("\\", "/")
            print(f"\n[INFO] Downloading: {remote_sim_dir} => {local_sim_dir}")
            sftp_copy_dir_to_local(sftp, remote_sim_dir, local_sim_dir)
        else:
            print(f"[INFO] {video_name} already has >=5 images in similar_image, skip download.")

        # ---------------- Second people similar_image2 ----------------
        local_sim_dir2 = os.path.join(local_video_dir, "similar_image2")
        sim2_count = 0
        if os.path.isdir(local_sim_dir2):
            sim2_count = len([f for f in os.listdir(local_sim_dir2) if is_image_file(f)])

        # Does the HQ face2 folder exist locally
        local_hq_face2 = os.path.join(local_video_dir, "HQ_face2")
        if os.path.isdir(local_hq_face2):
            if sim2_count < 5:
                remote_sim_dir2 = os.path.join(
                    REMOTE_TMP_PICTURE, video_name, "HQ_face2", "similar_image2"
                ).replace("\\", "/")
                print(f"\n[INFO] Downloading: {remote_sim_dir2} => {local_sim_dir2}")
                sftp_copy_dir_to_local(sftp, remote_sim_dir2, local_sim_dir2)
            else:
                print(f"[INFO] {video_name} already has >=5 images in similar_image2, skip download.")
        else:
            # If HQ face2 is not available locally at all, there is no need to download it
            pass

    # 5) 关闭连接
    sftp.close()
    client.close()
    print("\n[INFO] All done.")


if __name__ == "__main__":
    main()
