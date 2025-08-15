import subprocess
import sys
import time
import select


def interact_with_process(process):
    try:
        # Just wait and respond to the issue of interactive merging
        while True:
            # 读取输出
            line = process.stdout.readline()
            if not line:
                break

            print(line.strip())

            # Check if it is a prompt for interactive merging
            if "Use interactive merger" in line:
                print("Sending command: n")
                process.stdin.write("n\n")
                process.stdin.flush()
                time.sleep(0.2)

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        process.stdout.close()
        process.stdin.close()


cmd = ["python", "main.py", "merge",
       "--input-dir", "workspace/face_swap/data_dst",
       "--output-dir", "workspace/face_swap/data_dst/merged",
       "--output-mask-dir", "workspace/face_swap/data_dst/merged_mask",
       "--aligned-dir", "workspace/face_swap/data_dst/aligned",
       "--model-dir", "workspace/face_swap/model",
       "--model", "SAEHD"]

process = subprocess.Popen(cmd,
                           stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE,
                           universal_newlines=True,
                           bufsize=1)

interact_with_process(process)

# 打印任何错误
stderr = process.stderr.read()
if stderr:
    print("Errors:", stderr)