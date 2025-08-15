# update_qa_pairs.py
import os
import json
import pdb
import random
import re
from tqdm import tqdm
import re
import torch
import gc
from tqdm import tqdm
import openai
from openai import OpenAI
import base64

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
import json
import os




def load_video(video_path, num_segments=8, return_msg=False, resolution=224, hd_num=4, padding=False):
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
    # print(frames.shape)
    T_, C, H, W = frames.shape

    sub_img = frames.reshape(
        1, T_, 3, H//resolution, resolution, W//resolution, resolution
    ).permute(0, 3, 5, 1, 2, 4, 6).reshape(-1, T_, 3, resolution, resolution).contiguous()

    glb_img = F.interpolate(
        frames.float(), size=(resolution, resolution), mode='bicubic', align_corners=False
    ).to(sub_img.dtype).unsqueeze(0)

    frames = torch.cat([sub_img, glb_img]).unsqueeze(0)

    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return frames, msg
    else:
        return frames
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

    # calculate the existing video aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    if fix_ratio:
        target_aspect_ratio = fix_ratio
    else:
        target_aspect_ratio = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the frames
    resized_frame = F.interpolate(
        frames, size=(target_height, target_width),
        mode='bicubic', align_corners=False
    )
    return resized_frame

################################
# 1) This part is an example: InternVideo2 inference functions
#    In actual use, please integrate your complete InternVideo2 model loading and processing logic
################################
def get_caption_from_api(answer):
    original = "Help me replace all the descriptions of people in the following paragraph, such as human, he, she, etc., with <person>.Just return the modified one:"
    prompt = original + answer
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(
        api_key="api_key",
        base_url="https://api.openai-sb.com/v1"
    )

    for attempt in range(3):  # Retry up to 3 times
        try:
            response = client.chat.completions.create(

                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                stream=True,
            )

            reply = ""
            for res in response:
                content = res.choices[0].delta.content
                if content:
                    reply += content

            return reply
        except Exception as e:
            print(f"Error processing answer on attempt {attempt + 1}: {e}")
    return answer  # Fallback to the original answer if API fails
def load_video_for_internvideo2(video_path):
    """
    This is just an example.
    You can refer to your previous load_video(...) function for 8-frame sampling, preprocessing, etc.
    Return a tensor or other structure that can be fed to the InternVideo2 model.
    """
    # TODO: Use your actual implementation
    return torch.zeros((1, 10, 3, 224, 224))  # Pretend to return a Tensor


def ask_internvideo2(video_path, question, model=None, tokenizer=None):
    """
    For a single video and question, call the InternVideo2 model to get an answer.
    Here you need to actually load the video tensor and feed it to the model along with the question.
    Return the answer as a string.
    """
    # Load and preprocess video
    chat_history = []
    video_tensor = load_video(video_path)
    video_tensor = video_tensor.to(model.device)
    # Call your own model inference logic here
    response, chat_history = model.chat(
        tokenizer,
        '',
        question,
        media_type='video',
        media_tensor=video_tensor,
        chat_history=chat_history,
        return_history=True,
        generation_config={'do_sample': False}
    )
    # Below is just a mock, returning a fixed answer
    answer = response
    return answer


################################
# 2) This part is an example: Call ChatGPT or other API to replace person descriptions with <person>
################################

def call_chatgpt_api(text):
    """
    This is used to simulate using ChatGPT / openai / etc. API,
    to replace all references to people like 'he','she','man','woman','boy','girl','the person' etc. with <person>.
    You can use regex or more sophisticated language models for more precise replacement.
    This is just a demonstration.
    """
    # Simple replacement using regex
    # Note: In actual use, you can call openai.ChatCompletion.create(...) etc.
    # and write in the prompt: "Please replace all person references in the answer with <person>" etc.
    # Here's just a simple demonstration:
    # patterns = [
    #     r'\bhe\b', r'\bshe\b', r'\bman\b', r'\bwoman\b',
    #     r'\bboy\b', r'\bgirl\b', r'the person'
    # ]
    # replaced_text = text
    # for pat in patterns:
    #     replaced_text = re.sub(pat, "<person>", replaced_text, flags=re.IGNORECASE)
    # Call large language model
    replaced_text = get_caption_from_api(text)
    return replaced_text


################################
# 3) 15 question templates
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
# 4) Processing functions
################################

def process_json_file(input_path, output_path, model=None, tokenizer=None):
    """
    Read input_path (train_all_video.json or test_all_video.json),
    add 15 Q&A pairs to videos with "is_positive": true.
    Save to output_path.
    """
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

    for item in tqdm(video_items, desc=f"Processing {os.path.basename(input_path)}"):
        # Only process is_positive = true

        if not item.get("is_positive", False):
            continue

        video_name = item.get("video_name", "")
        video_path = item.get("video_path", "")
        if not video_name or not video_path:
            continue

        # Get existing qa_pairs, create empty list if none
        qa_pairs = item.get("qa_pairs", [])

        # For 15 questions, generate answers one by one
        # First replace <sks> -> "the person" to feed into InternVideo2
        # After getting answer -> ChatGPT API (replace all person references -> <person>)
        # Then replace <person> -> <sks>
        # question keeps original (with <sks>)
        # answer is finally stored in qa_pairs
        all_questions = (
                QUESTION_TEMPLATES["action_questions"] +
                QUESTION_TEMPLATES["clothing_questions"] +
                QUESTION_TEMPLATES["location_questions"]
        )

        for q in all_questions:
            # 1) Replace <sks> => "the person" (only when asking InternVideo2)
            intern_q = q.replace("<sks>", "the person")

            # 2) InternVideo2 answer
            intern_answer = ask_internvideo2(video_path, intern_q, model, tokenizer)

            # 3) Pass to ChatGPT API to replace with <person>
            replaced_with_person = call_chatgpt_api(intern_answer)

            # 4) Finally replace <person> => <sks>
            final_answer = replaced_with_person.replace("<person>", "<sks>")

            # Append to qa_pairs
            qa_pairs.append({
                "question": q,  # Question still keeps <sks>
                "answer": final_answer,  # Answer is the replaced version
                "is_special": False
            })

        # Update back
        item["qa_pairs"] = qa_pairs

    # Save new file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"data": video_items}, f, indent=2, ensure_ascii=False)
    print(f"[Info] Updated data saved to {output_path}.")


def main():
    # Assume you have loaded InternVideo2 model/tokenizer
    HF_TOKEN = os.environ['HF_TOKEN']
    token = HF_TOKEN
    decord.bridge.set_bridge("torch")
    tokenizer = AutoTokenizer.from_pretrained(
        '/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality/InternVideo2_chat_8B_HD',
        trust_remote_code=True,
        use_fast=False,
        token=token)
    if torch.cuda.is_available():
        model = AutoModel.from_pretrained(
            '/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality/InternVideo2_chat_8B_HD',
            torch_dtype=torch.bfloat16,
            trust_remote_code=True).cuda()
    else:
        model = AutoModel.from_pretrained(
            '/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality/InternVideo2_chat_8B_HD',
            torch_dtype=torch.bfloat16,
            trust_remote_code=True)

    # Input/output file examples
    train_input = "/root/autodl-tmp/yufei/datasets/cekebv-hq/train_all_video.json"
    test_input = "/root/autodl-tmp/yufei/datasets/cekebv-hq/test_all_video.json"
    train_output = "/root/autodl-tmp/yufei/datasets/cekebv-hq/train_all_video_updated.json"
    test_output = "/root/autodl-tmp/yufei/datasets/cekebv-hq/test_all_video_updated.json"

    print("[Main] Processing train_all_video.json...")
    process_json_file(train_input, train_output, model=model, tokenizer=tokenizer)

    print("[Main] Processing test_all_video.json...")
    process_json_file(test_input, test_output, model=model, tokenizer=tokenizer)

    print("[Main] Done.")


if __name__ == "__main__":
    main()