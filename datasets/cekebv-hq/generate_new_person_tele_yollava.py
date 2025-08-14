#!/usr/bin/env python
# build_dataset_ciin.py  (Final version: train negative samples keep celebvhq, test negative samples changed to 9 folders)
import os
import cv2
import json
import random
import argparse

# --------------------------------------------------
# Utility functions
# --------------------------------------------------
def create_video_from_image(img_path, out_path, repeat_frames=8, fps=25):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    h, w = img.shape[:2]
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(repeat_frames):
        vw.write(img)
    vw.release()

def load_qa_pairs(qa_path):
    with open(qa_path, "r", encoding="utf-8") as f:
        return json.load(f)

def create_qa_pairs(qa, is_pos, n=15):
    qs = random.sample(qa["questions"], n)
    if is_pos:
        ans = random.sample(qa["yes_answers"], n)
    else:
        ans = random.sample(qa["no_answers"], n)
    return [{"question": q, "answer": a, "is_special": True}
            for q, a in zip(qs, ans)]

def make_info(video_path, is_pos):
    return {
        "video_name": os.path.basename(video_path),
        "video_path": video_path,
        "sks_present": "",
        "gender": "",
        "age": "",
        "is_positive": is_pos,
        "qa_pairs": [],
    }

# --------------------------------------------------
# Positive samples png → mp4
# --------------------------------------------------
def convert_pngs_to_mp4(img_dir, prefix, start_idx, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    pngs = sorted([p for p in os.listdir(img_dir) if p.endswith(".png")],
                  key=lambda x: int(os.path.splitext(x)[0]))
    paths, idx = [], start_idx
    for p in pngs:
        img_path = os.path.join(img_dir, p)
        mp4_name = f"{prefix}{idx}.mp4"
        mp4_path = os.path.join(out_dir, mp4_name)
        create_video_from_image(img_path, mp4_path)
        paths.append(mp4_path)
        idx += 1
    return paths, idx

# --------------------------------------------------
# Main function
# --------------------------------------------------
def build_dataset(celebvhq_path, qa_path,
                  train_img_dir, test_img_dir,
                  test_video_root, tgt_dir,
                  train_json, test_json,
                  pos_folder="Aa"):

    qa = load_qa_pairs(qa_path)

    # 1) Positive samples
    train_pos, next_idx = convert_pngs_to_mp4(train_img_dir, "Aa", 1, tgt_dir)
    test_pos, _         = convert_pngs_to_mp4(test_img_dir,  "Aa", next_idx, tgt_dir)

    # 2) Training set negative samples —— celebvhq 30 entries
    with open(celebvhq_path, "r", encoding="utf-8") as f:
        js = json.loads(f.read().replace("True", "true").replace("False", "false"))
    persons = random.sample(list(js["groups"].keys()), 30)
    train_neg = []
    for pid in persons:
        clip_id, _ = random.choice(list(js["groups"][pid]["clips"].items()))
        mp4_path = os.path.join("/root/autodl-tmp/yufei/datasets/cekebv-hq/35666",
                                f"{clip_id}.mp4")
        train_neg.append(mp4_path)

    # 3) Test set negative samples —— all mp4 files from other 9 folders under test_video
    test_neg = []
    for sub in sorted(os.listdir(test_video_root)):
        sub_dir = os.path.join(test_video_root, sub)
        if not os.path.isdir(sub_dir) or sub == pos_folder:
            continue
        for f in os.listdir(sub_dir):
            if f.lower().endswith(".mp4"):
                test_neg.append(os.path.join(sub_dir, f))
    if not test_neg:
        raise RuntimeError("No negative samples found in test_video!")

    # 4) Assembly
    train, test = [], []
    for p in train_pos:
        info = make_info(p, True)
        info["qa_pairs"] = create_qa_pairs(qa, True)
        train.append(info)
    for p in test_pos:
        info = make_info(p, True)
        info["qa_pairs"] = create_qa_pairs(qa, True)
        test.append(info)

    for p in train_neg:
        info = make_info(p, False)
        info["qa_pairs"] = create_qa_pairs(qa, False)
        train.append(info)
    for p in test_neg:
        info = make_info(p, False)
        info["qa_pairs"] = create_qa_pairs(qa, False)
        test.append(info)

    random.shuffle(train)
    random.shuffle(test)

    # 5) Save
    with open(train_json, "w", encoding="utf-8") as f:
        json.dump({"data": train}, f, indent=2)
    with open(test_json, "w", encoding="utf-8") as f:
        json.dump({"data": test}, f, indent=2)

    # 6) Print statistics
    print("Completed ✅")
    print(f"Training set  Positive:{len(train_pos)}  Negative:{len(train_neg)}  Total:{len(train)}")
    print(f"Test set  Positive:{len(test_pos)}  Negative:{len(test_neg)}  Total:{len(test)}")
    print(f"train.json → {train_json}")
    print(f"test.json  → {test_json}")
    print(f"Positive sample videos written to {tgt_dir}")

# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    celebvhq_path = "/root/autodl-tmp/yufei/datasets/cekebv-hq/celebvhq_info_processed_updated.json"
    qa_path       = "/root/autodl-tmp/yufei/ConsisID/famaliar_question.json"

    train_img_dir = "/root/autodl-tmp/yufei/datasets/cekebv-hq/train_all_video_Aa/train"
    

    test_img_dir  = "/root/autodl-tmp/yufei/datasets/cekebv-hq/train_all_video_Aa/test"


    test_video_root = "/root/autodl-tmp/yufei/datasets/cekebv-hq/train_all_video_Aa/test"


    tgt_dir   = "/root/autodl-tmp/yufei/DeepFaceLab/data_src"
    train_js  = "/root/autodl-tmp/yufei/datasets/cekebv-hq/train.json"
    test_js   = "/root/autodl-tmp/yufei/datasets/cekebv-hq/test.json"

    build_dataset(celebvhq_path, qa_path,
                  train_img_dir, test_img_dir,
                  test_video_root, tgt_dir,
                  train_js, test_js)
