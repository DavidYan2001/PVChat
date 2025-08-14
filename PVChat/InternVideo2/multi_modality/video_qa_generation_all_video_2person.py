# update_qa_pairs.py
import os
import json
import random
import re
from tqdm import tqdm
import torch
import gc
import base64
import openai
from decord import VideoReader, cpu
from PIL import Image
import numpy as np
import decord
from decord import VideoReader, cpu
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import PILToTensor
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torch
import random
from transformers import AutoTokenizer, AutoModel
import argparse
################################
# 1) Video related functions (similar to existing ones)
################################

def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets

def HD_transform_padding(frames, image_size=224, hd_num=6):
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
    min_num = 1
    max_num = hd_num
    _, _, orig_height, orig_width = frames.shape
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # If fix_ratio is not empty, force using it
    if fix_ratio:
        target_aspect_ratio = fix_ratio
    else:
        target_aspect_ratio = (1,1)  # Or write your own find_closest_aspect_ratio(...)

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
    frames = frames.permute(0, 3, 1, 2)

    if padding:
        frames = HD_transform_padding(frames.float(), image_size=resolution, hd_num=hd_num)
    else:
        frames = HD_transform_no_padding(frames.float(), image_size=resolution, hd_num=hd_num)

    frames = transform(frames)
    T_, C, H, W = frames.shape

    sub_img = frames.reshape(
        1, T_, 3, H // resolution, resolution, W // resolution, resolution
    ).permute(0, 3, 5, 1, 2, 4, 6).reshape(-1, T_, 3, resolution, resolution).contiguous()

    glb_img = F.interpolate(
        frames.float(), size=(resolution, resolution), mode='bicubic', align_corners=False
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
# 2) API calls (example)
################################

def get_caption_from_api(answer):
    """
    Call third-party ChatGPT-like API to replace pronouns in answers with <person>.
    Using gpt-4o-mini as example, you need to implement this yourself.
    """
    # Your own GPT proxy
    import openai
    api_key = os.getenv("OPENAI_API_KEY")

    openai.api_base = "https://api.openai-sb.com/v1"
    openai.api_key = api_key


    prompt = (
        "Help me replace all the descriptions of people in the following paragraph "
        "such as human, he, she, etc., with <person>. Just return the modified one:\n"
        + answer
    )
    try:
        # Using ChatCompletion with stream = False
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
        return answer  # Fallback

def call_chatgpt_api(text):
    """Simple wrapper"""
    replaced_text = get_caption_from_api(text)
    return replaced_text

################################
# 3) InternVideo2 interaction
################################

def ask_internvideo2(video_path, question, model=None, tokenizer=None):
    """
    Load and preprocess video, then send to InternVideo2 for Q&A.
    Return answer string
    """
    video_tensor = load_video(video_path, num_segments=8, resolution=224, hd_num=4)
    if video_tensor is None:
        return "No video found"

    if torch.cuda.is_available():
        video_tensor = video_tensor.to(model.device)

    chat_history = []
    response, chat_history = model.chat(
        tokenizer,
        '',    # system prompt (can set others)
        question,
        media_type='video',
        media_tensor=video_tensor,
        chat_history=chat_history,
        return_history=True,
        generation_config={'do_sample': False}
    )
    return response


################################
# 4) Question templates
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
# 5) Main logic: Process based on different sample_type
#    person1 -> "the man", finally replace with <sks1>
#    person2 -> "the woman", finally replace with <sks2>
#    both -> ask twice
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

    # All questions (15 total)
    all_questions = (
        QUESTION_TEMPLATES["action_questions"] +
        QUESTION_TEMPLATES["clothing_questions"] +
        QUESTION_TEMPLATES["location_questions"]
    )

    for item in tqdm(video_items, desc=f"Processing {os.path.basename(input_path)}"):
        # Only process is_positive = true
        if not item.get("is_positive", False):
            continue

        sample_type = item.get("sample_type", "")
        video_name = item.get("video_name", "")
        video_path = item.get("video_path", "")
        if not video_name or not video_path:
            continue

        qa_pairs = item.get("qa_pairs", [])

        # Based on sample_type, decide how many times to ask and what pronouns to use
        # person1 => ask once: "the man" => <sks1>
        # person2 => ask once: "the woman" => <sks2>
        # both    => ask twice: (man-><sks1>), (woman-><sks2>)
        # Otherwise skip
        if sample_type == "person1":
            # Ask once
            for q in all_questions:
                # Question: <sks> => "the man"
                question_in_intern = q.replace("<sks>", "the man")
                # InternVideo2 answer
                intern_answer = ask_internvideo2(video_path, question_in_intern, model, tokenizer)
                # ChatGPT API => replace pronouns in answer with <person>
                replaced_with_person = call_chatgpt_api(intern_answer)
                # Finally replace <person> => <sks1>
                final_answer = replaced_with_person.replace("<person>", "<sks1>")
                # Also in question <sks> => <sks1>
                final_question = q.replace("<sks>", "<sks1>")

                qa_pairs.append({
                    "question": final_question,
                    "answer": final_answer,
                    "is_special": False
                })

        elif sample_type == "person2":
            # Ask once
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

        elif sample_type == "both":
            # Ask twice (left man -> <sks1>), (right man -> <sks2>)
            for q in all_questions:
                # 1) For the man => <sks1>
                question_in_intern = q.replace("<sks>", "the left child")
                intern_answer = ask_internvideo2(video_path, question_in_intern, model, tokenizer)
                replaced_with_person = call_chatgpt_api(intern_answer)
                final_answer = replaced_with_person.replace("<person>", "<sks1>")
                final_question = q.replace("<sks>", "<sks1>")

                qa_pairs.append({
                    "question": final_question,
                    "answer": final_answer,
                    "is_special": False
                })

                # 2) For the woman => <sks2>
                question_in_intern = q.replace("<sks>", "the right person")
                intern_answer = ask_internvideo2(video_path, question_in_intern, model, tokenizer)
                replaced_with_person = call_chatgpt_api(intern_answer)
                final_answer = replaced_with_person.replace("<person>", "<sks2>")
                final_question = q.replace("<sks>", "<sks2>")

                qa_pairs.append({
                    "question": final_question,
                    "answer": final_answer,
                    "is_special": False
                })
        else:
            # Other cases, like random (negative examples) or empty sample_type, don't ask
            continue

        # Update
        item["qa_pairs"] = qa_pairs

    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"data": video_items}, f, indent=2, ensure_ascii=False)
    print(f"[Info] Updated data saved to {output_path}.")

def get_args():
    parser = argparse.ArgumentParser(description="Build dataset for two people and both cases")

    parser.add_argument('--sks1', type=str, default='Sh',
                        help='Name of first person (default: Sh)')
    parser.add_argument('--sks2', type=str, default='Ho',
                        help='Name of second person (default: Ho)')
    return parser.parse_args()

def main():
    # Initialize InternVideo2
    args = get_args()
    HF_TOKEN= os.getenv("HF_TOKEN")
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

    # Input and output files
    train_input = "/root/autodl-tmp/yufei/datasets/cekebv-hq/train_all_video_" + args.sks1 + '_' + args.sks2 + '.json'
    test_input= "/root/autodl-tmp/yufei/datasets/cekebv-hq/test_all_video_" + args.sks1 + '_' + args.sks2 + '.json'
    # train_input = "/root/autodl-tmp/yufei/datasets/cekebv-hq/train_all_video.json"
    # test_input = "/root/autodl-tmp/yufei/datasets/cekebv-hq/test_all_video.json"
    train_output = "/root/autodl-tmp/yufei/datasets/cekebv-hq/train_all_video_updated_" + args.sks1 + '_' + args.sks2 + '.json'
    test_output = "/root/autodl-tmp/yufei/datasets/cekebv-hq/test_all_video_updated_" + args.sks1 + '_' + args.sks2 + '.json'

    print("[Main] Processing train_all_video.json...")
    process_json_file(train_input, train_output, model=model, tokenizer=tokenizer)

    print("[Main] Processing test_all_video.json...")
    process_json_file(test_input, test_output, model=model, tokenizer=tokenizer)

    print("[Main] Done.")


if __name__ == "__main__":
    main()
