#!/usr/bin/env python
# coding: utf-8

"""
update_qa_pairs.py
Goal: For 3-person scenarios, for entries with is_positive=True, based on sample_type={person1, person2, person3}
     call InternVideo2 to generate answers, process through ChatGPT API, and finally save to new JSON.
"""

import os
import json
import random
import re
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

from glob import glob
from tqdm import tqdm

# If your script still needs decord, PIL and other dependencies, keep them unchanged.
import decord
from decord import VideoReader, cpu
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel

from qwen_video_qa import ask_qwen3_vl, load_qwen_model_and_processor

NEGATIVE_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[3] / "consisid" / "negative_sample_3question.json"
)
CATEGORY_TO_RESPONSE_KEY = {
    "action_questions": "action_responses",
    "clothing_questions": "clothing_responses",
    "location_questions": "location_responses",
    "emotion_questions": "emotion_responses",
}


def load_negative_qa_config():
    with open(NEGATIVE_TEMPLATE_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


NEGATIVE_QA_CONFIG = load_negative_qa_config()


def build_negative_qa_pairs(target_token):
    qa_pairs = []
    for category_name, response_name in CATEGORY_TO_RESPONSE_KEY.items():
        questions = NEGATIVE_QA_CONFIG["questions"][category_name]
        responses = NEGATIVE_QA_CONFIG["negative_responses"][response_name]
        for question in questions:
            qa_pairs.append(
                {
                    "question": question.replace("<sks>", target_token),
                    "answer": random.choice(responses).replace("<sks>", target_token),
                    "is_special": False,
                }
            )
    return qa_pairs


################################
# 1) Video-related functions
################################

def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets

def HD_transform_padding(frames, image_size=224, hd_num=6):
    # Consistent with original logic, no further elaboration
    def _padding_224(frames):
        _, _, H, W = frames.shape
        tar = int(np.ceil(H / 224) * 224)
        top_padding = (tar - H) // 2
        bottom_padding = tar - H - top_padding
        left_padding = 0
        right_padding = 0

        padded_frames = F.pad(
            frames,
            pad=[left_padding, right_padding, top_padding, bottom_padding],
            mode='constant', value=255
        )
        return padded_frames

    _, _, H, W = frames.shape
    trans = False
    if W < H:
        frames = frames.flip(-2, -1)
        trans = True
        width, height = H, W
    else:
        width, height = W, H

    ratio = width / height
    scale = 1
    while scale * np.ceil(scale / ratio) <= hd_num:
        scale += 1
    scale -= 1
    new_w = int(scale * image_size)
    new_h = int(new_w / ratio)

    resized_frames = F.interpolate(
        frames, size=(new_h, new_w),
        mode='bicubic',
        align_corners=False
    )
    padded_frames = _padding_224(resized_frames)

    if trans:
        padded_frames = padded_frames.flip(-2, -1)

    return padded_frames

def HD_transform_no_padding(frames, image_size=224, hd_num=6, fix_ratio=(2,1)):
    # Consistent with original logic, no further elaboration
    min_num = 1
    max_num = hd_num
    _, _, orig_height, orig_width = frames.shape
    aspect_ratio = orig_width / orig_height

    # Example only, no more details
    target_aspect_ratio = fix_ratio or (1,1)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]

    resized_frame = F.interpolate(
        frames, size=(target_height, target_width),
        mode='bicubic', align_corners=False
    )
    return resized_frame

def load_video(video_path, num_segments=8, return_msg=False, resolution=224, hd_num=4, padding=False):
    if not os.path.exists(video_path):
        print(f"[Warning] video not found: {video_path}")
        return None

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Normalize(mean, std)
    ])

    frames = vr.get_batch(frame_indices)
    frames = frames.permute(0, 3, 1, 2).float()

    if padding:
        frames = HD_transform_padding(frames, image_size=resolution, hd_num=hd_num)
    else:
        frames = HD_transform_no_padding(frames, image_size=resolution, hd_num=hd_num)

    frames = transform(frames)
    T_, C, H, W = frames.shape

    # sub_img logic
    sub_img = frames.reshape(
        1, T_, 3, H // resolution, resolution, W // resolution, resolution
    ).permute(0, 3, 5, 1, 2, 4, 6).reshape(-1, T_, 3, resolution, resolution).contiguous()

    glb_img = F.interpolate(
        frames, size=(resolution, resolution), mode='bicubic', align_corners=False
    ).to(sub_img.dtype).unsqueeze(0)

    frames = torch.cat([sub_img, glb_img]).unsqueeze(0)

    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return frames, msg
    else:
        return frames


################################
# 2) ChatGPT-like API
################################

def get_caption_from_api(answer):
    """
    Call third-party ChatGPT-like API to replace person references in the answer content.
    You can implement according to your actual API.
    """
    import openai
    # Example only
    openai.api_base = "https://api.openai-sb.com/v1"
    api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key =api_key

    prompt = (
        "Help me replace all the descriptions of people in the following paragraph "
        "such as human, he, she, etc., with <person>. Just return the modified one:\n"
        + answer
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        replaced = response.choices[0].message.content
        return replaced
    except Exception as e:
        print(f"[Error] call_chatgpt_api failed: {e}")
        return answer  # fallback

def call_chatgpt_api(text):
    """Simple wrapper"""
    replaced_text = get_caption_from_api(text)
    return replaced_text


################################
# 3) Call InternVideo2 for video Q&A
################################

def ask_qwen_video_qa(video_path, question, model=None, processor=None):
    """
    Send a video and question to Qwen3-VL => return answer string
    """
    return ask_qwen3_vl(video_path, question, model=model, processor=processor)


################################
# 4) Question templates (example)
################################

QUESTION_TEMPLATES = {
    "action_questions": [
        "What activity is <sks> engaged in during this video?",
        "Could you describe what <sks> is doing in this footage?",
        "What specific actions can you observe <sks> performing in this recording?",
        "What movements or actions does <sks> perform here?",
        "Can you describe <sks>'s behavior in this sequence?"
    ],
    "clothing_questions": [
        "What is <sks> wearing in this video?",
        "Could you describe <sks>'s outfit in this footage?",
        "What color and style of clothing is <sks> dressed in?",
        "How would you describe <sks>'s appearance and attire?",
        "What notable features can you see in <sks>'s clothing?"
    ],
    "location_questions": [
        "Where is <sks> positioned in this video?",
        "Can you describe <sks>'s location relative to others?",
        "Which part of the scene does <sks> appear in?",
        "How does <sks>'s position change throughout the video?",
        "Where can <sks> be found in this footage?"
    ],
    "emotion_questions": [
        "What visible emotion does <sks> appear to be expressing in this video?",
        "How would you describe <sks>'s facial expression or emotional state in this footage?",
        "What kind of emotional expression does <sks> show in this recording?",
        "Based on the video, what emotion or mood does <sks> seem to display?",
        "How does <sks> appear to feel from their expression and body language here?"
    ]
}


################################
# 5) Core logic: for 3 types of personX
################################

def process_json_file(input_path, output_path, model=None, processor=None):
    if not os.path.exists(input_path):
        print(f"[Warning] {input_path} not found, skip.")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if "data" not in data or not isinstance(data["data"], list):
        print(f"[Warning] {input_path} has no 'data' list, skip.")
        return

    video_items = data["data"]
    print(f"[Info] Loaded {len(video_items)} video entries from {input_path}.")

    # Collect all questions to ask (20 in total)
    all_questions = (
        QUESTION_TEMPLATES["action_questions"] +
        QUESTION_TEMPLATES["clothing_questions"] +
        QUESTION_TEMPLATES["location_questions"] +
        QUESTION_TEMPLATES["emotion_questions"]
    )

    for item in tqdm(video_items, desc=f"Processing {os.path.basename(input_path)}"):
        if not item.get("is_positive", False):
            qa_pairs = item.get("qa_pairs", [])
            qa_pairs.extend(build_negative_qa_pairs("<sks1>"))
            qa_pairs.extend(build_negative_qa_pairs("<sks2>"))
            qa_pairs.extend(build_negative_qa_pairs("<sks3>"))
            item["qa_pairs"] = qa_pairs
            continue

        sample_type = item.get("sample_type", "")
        video_name = item.get("video_name", "")
        video_path = item.get("video_path", "")
        if not video_name or not video_path:
            continue

        qa_pairs = item.get("qa_pairs", [])

        if sample_type == "person1":
            # Ask once: <sks> => "the woman" (example), final answer <person> => <sks1>
            for q in all_questions:
                question_in_intern = q.replace("<sks>", "the woman")
                intern_answer = ask_qwen_video_qa(video_path, question_in_intern, model, processor)
                replaced_with_person = call_chatgpt_api(intern_answer)
                final_answer = replaced_with_person.replace("<person>", "<sks1>")
                final_question = q.replace("<sks>", "<sks1>")

                qa_pairs.append({
                    "question": final_question,
                    "answer": final_answer,
                    "is_special": False
                })

        elif sample_type == "person2":
            # Ask once: <sks> => "the man", <person> => <sks2>
            for q in all_questions:
                question_in_intern = q.replace("<sks>", "the man")
                intern_answer = ask_qwen_video_qa(video_path, question_in_intern, model, processor)
                replaced_with_person = call_chatgpt_api(intern_answer)
                final_answer = replaced_with_person.replace("<person>", "<sks2>")
                final_question = q.replace("<sks>", "<sks2>")

                qa_pairs.append({
                    "question": final_question,
                    "answer": final_answer,
                    "is_special": False
                })

        elif sample_type == "person3":
            # Ask once: <sks> => "the child", <person> => <sks3>
            for q in all_questions:
                question_in_intern = q.replace("<sks>", "the child")
                intern_answer = ask_qwen_video_qa(video_path, question_in_intern, model, processor)
                replaced_with_person = call_chatgpt_api(intern_answer)
                final_answer = replaced_with_person.replace("<person>", "<sks3>")
                final_question = q.replace("<sks>", "<sks3>")

                qa_pairs.append({
                    "question": final_question,
                    "answer": final_answer,
                    "is_special": False
                })

        else:
            # Other sample_types, such as "random" or "both" etc., not processed for now
            continue

        item["qa_pairs"] = qa_pairs

    # Write back to output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"data": video_items}, f, indent=2, ensure_ascii=False)
    print(f"[Info] Updated data saved => {output_path}.")


################################
# 6) Entry point main()
################################

def get_args():
    parser = argparse.ArgumentParser(description="Update QA pairs for 3-person scenario.")
    parser.add_argument('--sks1', type=str, default='Cl', help='Name of first person (default: Cl)')
    parser.add_argument('--sks2', type=str, default='Xo', help='Name of second person (default: Xo)')
    parser.add_argument('--sks3', type=str, default='Ja', help='Name of third person (default: Ja)')
    return parser.parse_args()

def main():
    args = get_args()
    decord.bridge.set_bridge("torch")

    hf_token = os.getenv("HF_TOKEN")
    model, processor = load_qwen_model_and_processor(hf_token=hf_token)

    # Input and output file paths: e.g. train_Cl_Xo_Ja.json => train_all_video_Cl_Xo_Ja.json
    base_dir = "/root/autodl-tmp/yufei/datasets/cekebv-hq"

    train_input  = os.path.join(base_dir, f"train_all_video_{args.sks1}_{args.sks2}_{args.sks3}.json")
    test_input   = os.path.join(base_dir, f"test_all_video_{args.sks1}_{args.sks2}_{args.sks3}.json")

    train_output = os.path.join(base_dir, f"train_all_video_updated_{args.sks1}_{args.sks2}_{args.sks3}.json")
    test_output  = os.path.join(base_dir, f"test_all_video_updated_{args.sks1}_{args.sks2}_{args.sks3}.json")

    print("[Main] Processing train_all_video.json...")
    process_json_file(train_input, train_output, model=model, processor=processor)

    print("[Main] Processing test_all_video.json...")
    process_json_file(test_input, test_output, model=model, processor=processor)

    print("[Main] Done.")

if __name__ == "__main__":
    main()
