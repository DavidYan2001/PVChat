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
from pathlib import Path

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

def ask_qwen_video_qa(video_path, question, model=None, processor=None):
    """
    Send a video and question to Qwen3-VL for Q&A.
    Return answer string
    """
    return ask_qwen3_vl(video_path, question, model=model, processor=processor)


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
# 5) Main logic: Process based on different sample_type
#    person1 -> "the man", finally replace with <sks1>
#    person2 -> "the woman", finally replace with <sks2>
#    both -> ask twice
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

    # All questions (20 total)
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
            item["qa_pairs"] = qa_pairs
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
                intern_answer = ask_qwen_video_qa(video_path, question_in_intern, model, processor)
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
                question_in_intern = q.replace("<sks>", "the woman")
                intern_answer = ask_qwen_video_qa(video_path, question_in_intern, model, processor)
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
                intern_answer = ask_qwen_video_qa(video_path, question_in_intern, model, processor)
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
                intern_answer = ask_qwen_video_qa(video_path, question_in_intern, model, processor)
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
    hf_token = os.getenv("HF_TOKEN")
    model, processor = load_qwen_model_and_processor(hf_token=hf_token)

    # Input and output files
    train_input = "/root/autodl-tmp/yufei/datasets/cekebv-hq/train_all_video_" + args.sks1 + '_' + args.sks2 + '.json'
    test_input= "/root/autodl-tmp/yufei/datasets/cekebv-hq/test_all_video_" + args.sks1 + '_' + args.sks2 + '.json'
    # train_input = "/root/autodl-tmp/yufei/datasets/cekebv-hq/train_all_video.json"
    # test_input = "/root/autodl-tmp/yufei/datasets/cekebv-hq/test_all_video.json"
    train_output = "/root/autodl-tmp/yufei/datasets/cekebv-hq/train_all_video_updated_" + args.sks1 + '_' + args.sks2 + '.json'
    test_output = "/root/autodl-tmp/yufei/datasets/cekebv-hq/test_all_video_updated_" + args.sks1 + '_' + args.sks2 + '.json'

    print("[Main] Processing train_all_video.json...")
    process_json_file(train_input, train_output, model=model, processor=processor)

    print("[Main] Processing test_all_video.json...")
    process_json_file(test_input, test_output, model=model, processor=processor)

    print("[Main] Done.")


if __name__ == "__main__":
    main()
