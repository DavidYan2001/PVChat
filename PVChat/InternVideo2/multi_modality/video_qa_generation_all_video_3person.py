#!/usr/bin/env python
# coding: utf-8

"""
update_qa_pairs.py
目标: 针对3个人的场景, 对 is_positive=True 的条目, 根据 sample_type={person1, person2, person3}
     去调用 InternVideo2 生成回答, 再经ChatGPT API处理, 最终保存至新的JSON.
"""

import os
import json
import random
import re
import argparse

import torch
import torch.nn.functional as F
import numpy as np

from glob import glob
from tqdm import tqdm

# 如果你的脚本中还需要 decord、PIL 等依赖, 保持不变.
import decord
from decord import VideoReader, cpu
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel


################################
# 1) 视频相关函数
################################

def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets

def HD_transform_padding(frames, image_size=224, hd_num=6):
    # 与原逻辑一致，不再赘述
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
    # 与原逻辑一致，不再赘述
    min_num = 1
    max_num = hd_num
    _, _, orig_height, orig_width = frames.shape
    aspect_ratio = orig_width / orig_height

    # 仅示例, 不再详细
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

    # sub_img 逻辑
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
    调用第三方 ChatGPT-like API，对回答内容中的人称做替换。
    你可根据自己实际API实现。
    """
    import openai
    # 仅示例
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
    """简单包装"""
    replaced_text = get_caption_from_api(text)
    return replaced_text


################################
# 3) 调用 InternVideo2 进行视频问答
################################

def ask_internvideo2(video_path, question, model=None, tokenizer=None):
    """
    加载并处理视频 => InternVideo2模型 => 返回回答字符串
    """
    video_tensor = load_video(video_path, num_segments=8, resolution=224, hd_num=4)
    if video_tensor is None:
        return "No video found"

    if torch.cuda.is_available():
        video_tensor = video_tensor.to(model.device)

    chat_history = []
    response, chat_history = model.chat(
        tokenizer,
        '',    # system prompt
        question,
        media_type='video',
        media_tensor=video_tensor,
        chat_history=chat_history,
        return_history=True,
        generation_config={'do_sample': False}
    )
    return response


################################
# 4) 问题模板 (示例)
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
    ]
}


################################
# 5) 核心逻辑: 针对 3 种 personX
################################

def process_json_file(input_path, output_path, model=None, tokenizer=None):
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

    # 收集要问的所有问题(总计15条)
    all_questions = (
        QUESTION_TEMPLATES["action_questions"] +
        QUESTION_TEMPLATES["clothing_questions"] +
        QUESTION_TEMPLATES["location_questions"]
    )

    for item in tqdm(video_items, desc=f"Processing {os.path.basename(input_path)}"):
        if not item.get("is_positive", False):
            # 负样本或其它情况不处理
            continue

        sample_type = item.get("sample_type", "")
        video_name = item.get("video_name", "")
        video_path = item.get("video_path", "")
        if not video_name or not video_path:
            continue

        qa_pairs = item.get("qa_pairs", [])

        if sample_type == "person1":
            # 问一次: <sks> => "the woman" (示例), 最终回答中 <person> => <sks1>
            for q in all_questions:
                question_in_intern = q.replace("<sks>", "the woman")
                intern_answer = ask_internvideo2(video_path, question_in_intern, model, tokenizer)
                replaced_with_person = call_chatgpt_api(intern_answer)
                final_answer = replaced_with_person.replace("<person>", "<sks1>")
                final_question = q.replace("<sks>", "<sks1>")

                qa_pairs.append({
                    "question": final_question,
                    "answer": final_answer,
                    "is_special": False
                })

        elif sample_type == "person2":
            # 问一次: <sks> => "the man", <person> => <sks2>
            for q in all_questions:
                question_in_intern = q.replace("<sks>", "the man")
                intern_answer = ask_internvideo2(video_path, question_in_intern, model, tokenizer)
                replaced_with_person = call_chatgpt_api(intern_answer)
                final_answer = replaced_with_person.replace("<person>", "<sks2>")
                final_question = q.replace("<sks>", "<sks2>")

                qa_pairs.append({
                    "question": final_question,
                    "answer": final_answer,
                    "is_special": False
                })

        elif sample_type == "person3":
            # 问一次: <sks> => "the child", <person> => <sks3>
            for q in all_questions:
                question_in_intern = q.replace("<sks>", "the child")
                intern_answer = ask_internvideo2(video_path, question_in_intern, model, tokenizer)
                replaced_with_person = call_chatgpt_api(intern_answer)
                final_answer = replaced_with_person.replace("<person>", "<sks3>")
                final_question = q.replace("<sks>", "<sks3>")

                qa_pairs.append({
                    "question": final_question,
                    "answer": final_answer,
                    "is_special": False
                })

        else:
            # 其余 sample_type，比如 "random" 或 "both" 等，先不处理
            continue

        item["qa_pairs"] = qa_pairs

    # 写回 output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"data": video_items}, f, indent=2, ensure_ascii=False)
    print(f"[Info] Updated data saved => {output_path}.")


################################
# 6) 入口 main()
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

    # 这里放你的 Huggingface token or any other required
    HF_TOKEN = os.environ['HF_TOKEN']
    token = HF_TOKEN
    decord.bridge.set_bridge("torch")

    model_path = '/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality/InternVideo2_chat_8B_HD'

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
        token=token
    )
    if torch.cuda.is_available():
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).cuda()
    else:
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

    # 输入、输出文件路径:  e.g. train_Cl_Xo_Ja.json => train_all_video_Cl_Xo_Ja.json
    base_dir = "/root/autodl-tmp/yufei/datasets/cekebv-hq"

    train_input  = os.path.join(base_dir, f"train_all_video_{args.sks1}_{args.sks2}_{args.sks3}.json")
    test_input   = os.path.join(base_dir, f"test_all_video_{args.sks1}_{args.sks2}_{args.sks3}.json")

    train_output = os.path.join(base_dir, f"train_all_video_updated_{args.sks1}_{args.sks2}_{args.sks3}.json")
    test_output  = os.path.join(base_dir, f"test_all_video_updated_{args.sks1}_{args.sks2}_{args.sks3}.json")

    print("[Main] Processing train_all_video.json...")
    process_json_file(train_input, train_output, model=model, tokenizer=tokenizer)

    print("[Main] Processing test_all_video.json...")
    process_json_file(test_input, test_output, model=model, tokenizer=tokenizer)

    print("[Main] Done.")

if __name__ == "__main__":
    main()
