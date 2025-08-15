import os
import subprocess
import time


def ensure_remote_dir(remote_host, remote_path):
    """Make sure the remote directory exists"""
    ssh_command = f"ssh -p xxx xxx@xxx 'mkdir -p {remote_path}'"
    subprocess.run(ssh_command, shell=True)


def ensure_local_dir(local_path):
    """"""
    os.makedirs(local_path, exist_ok=True)


def upload_faces():
    """上传所有HQ_face图片到远程服务器"""
    base_dir = "/root/autodl-tmp/yufei/DeepFaceLab/output"
    remote_base = "/mnt/hdd1/yufei/img2dataset/tmp_picture"

    #
    for video_dir in os.listdir(base_dir):
        face_dir = os.path.join(base_dir, video_dir, "HQ_face")
        if not os.path.isdir(face_dir):
            continue

        # Create a remote corresponding directory
        remote_dir = f"{remote_base}/{video_dir}/HQ_face"
        ensure_remote_dir("yufei@gj03.ezfrp.com", remote_dir)

        # Create a local "similar laion picture" directory
        similar_dir = os.path.join(base_dir, video_dir, "similar_laion_picture")
        ensure_local_dir(similar_dir)

        # Upload all the pictures
        for img in os.listdir(face_dir):
            if img.endswith(('.jpg', '.jpeg', '.png')):
                local_path = os.path.join(face_dir, img)
                remote_path = f"{remote_dir}/{img}"
                scp_command = f"scp -P xxxxx {local_path} xxx@xxxx:{remote_path}"
                print(f"Uploading {img}...")
                subprocess.run(scp_command, shell=True)

                # Wait for the remote processing to complete
                print("Waiting for remote processing...")
                time.sleep(10)  # Give the remote server some processing time

                # Download the processing result
                similar_path = os.path.join(similar_dir, img.replace('.jpg', '_similar'))
                ensure_local_dir(similar_path)
                scp_command = f"scp -P xxx -r xxx@xxx:/mnt/hdd1/yufei/img2dataset/similar_results/{video_dir}/{img.replace('.jpg', '')}/* {similar_path}/"
                subprocess.run(scp_command, shell=True)


if __name__ == "__main__":
    upload_faces()