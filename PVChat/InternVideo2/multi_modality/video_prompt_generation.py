# -*- coding: utf-8 -*-
import os
import pdb

#token = os.environ['HF_TOKEN']=
HF_TOKEN= os.environ['HF_TOKEN']
token = HF_TOKEN
import torch
import random
from transformers import AutoTokenizer, AutoModel
import json
import re
import torch
import gc
from tqdm import tqdm
from openai import OpenAI
import os
import os
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
decord.bridge.set_bridge("torch")
tokenizer =  AutoTokenizer.from_pretrained('/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality/InternVideo2_chat_8B_HD',
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



def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets


def get_prompts_from_class_data(class_data, gender, age):
    """
    Obtain all prompts corresponding to gender and age from class_data
    """
    prompts = []

    if gender not in class_data or age not in class_data[gender]:
        return prompts

    scenarios = class_data[gender][age]

    # Traverse each large scene
    for main_scenario, sub_scenarios in scenarios.items():
        # For each small scene within a large one
        for sub_scenario, specific_scenarios in sub_scenarios.items():
            # If specific_scenarios is a dictionary, randomly select a description
            if isinstance(specific_scenarios, dict):
                if specific_scenarios:  #Make sure it is not an empty dictionary

                    random_description = random.choice(list(specific_scenarios.values()))
                    prompts.append(random_description)
            # If specific scenarios are strings, add them directly
            elif isinstance(specific_scenarios, str):
                prompts.append(specific_scenarios)

    return prompts
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

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio


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

def parse_gender_age(response):
    # "Analysis of Gender
    gender = None
    if 'male' in response.lower():
        gender = 'male'
    if 'female' in response.lower():
        gender = 'female'

    # 解析年龄
    age = None
    age_keywords = ['child', 'young', 'middle-aged', 'elderly']
    for keyword in age_keywords:
        if keyword in response.lower():
            age = keyword
            break

    return gender, age
# process video
def get_caption_from_api(answer):
    original = "Help me replace all the descriptions of people in the following paragraph, such as human, he, she, etc., with <person>.Just return the modified one:"
    prompt = original + answer

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

def process_question(question, sks_name, is_special=False):
    """
    Handle the problem text and replace it according to different situations
    """
    if is_special:
        # For special issues, add Angle brackets to sks name
        return question#.replace(sks_name, f'<{sks_name}>')
    elif question == "Describe the action step by step.":
        # For descriptive questions, keep them as they are
        return question
    else:
        # For other questions, replace "the person" or "person" with <sks_name>
        return question.replace('the person', f'<{sks_name}>').replace('person', f'<{sks_name}>')

def process_qa_dataset(input_file, output_file):
    with open(input_file, "r") as f:
        data = json.load(f)

    for video in tqdm(data["videos"], desc="Processing videos"):
        for qa in video["qa_pairs"]:
            if not qa["is_special"]:  # Skip if is_special is True
                qa["answer"] = get_caption_from_api(qa["answer"])

    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)
def replace_person_with_name(text, name):
    """
Replace <person> in the text with <name>
    """
    return text.replace('<person>', f'<{name}>')


# In the video processing loop, modify the section for handling special issues
def get_special_qa_pairs(sks_name):
    # Randomly select 10 questions and their corresponding affirmative answers
    selected_indices = random.sample(range(len(confirm_data["questions"])), 10)
    special_qa_pairs = []

    for idx in selected_indices:
        # Replace <sks> in the problem with the specific <sks name>
        question = confirm_data["questions"][idx].replace("<sks>", f"<{sks_name}>")
        # Randomly select an affirmative answer and replace <sks>
        answer = random.choice(confirm_data["yes_answers"]).replace("<sks>", f"<{sks_name}>")

        special_qa_pairs.append({
            "question": question,
            "answer": answer,
            "is_special": True
        })

    return special_qa_pairs



print("Starting...")

# set path
video_dir = "/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality/Finetune_datasets/YUFEI"
output_json = "qa_dataset4.json"
# Get the list of video files
# Debugging Information


# Get the list of video files (case sensitive)
video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
print("Found video files:", video_files)
# Initialize or load the dataset
if os.path.exists(output_json):
    with open(output_json, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    processed_files = {v['video_name'] for v in dataset['videos']}
    video_files = [f for f in video_files if f not in processed_files]
else:
    dataset = {"videos": []}
# Load the JSON file for confirming the issue
with open("/root/autodl-tmp/yufei/ConsisID/confirm_question.json", 'r', encoding='utf-8') as f:
    confirm_data = json.load(f)

# Load the JSON files of the three types of questions
with open("/root/autodl-tmp/yufei/ConsisID/negative_sample_3question.json", 'r', encoding='utf-8') as f:
    three_type_questions = json.load(f)
# Load the class data before the main processing loop
with open("/root/autodl-tmp/yufei/ConsisID/class_data_modified.json", 'r', encoding='utf-8') as f:
    class_data = json.load(f)

original_action_questions = three_type_questions["questions"]["action_questions"]
original_clothing_questions = three_type_questions["questions"]["clothing_questions"]
original_location_questions = three_type_questions["questions"]["location_questions"]
questions = [
    "Describe the action step by step.",
    *(q.replace("<sks>", "the person") for q in original_action_questions),
    *(q.replace("<sks>", "the person") for q in original_clothing_questions),
    *(q.replace("<sks>", "the person") for q in original_location_questions),
    'According to the video and just told me two words about gender from "male, female" and age from "child, young, middle-aged, elderly" about the protagonist of the video'
]

client = OpenAI(
    api_key="xxxx",
    base_url="https://api.openai-sb.com/v1"
)

# Configure OpenAI API
openai.api_key = "xxxx"
openai.api_base = "https://api.openai-sb.com/v1"




print("video_files",video_files)
for video_file in tqdm(video_files):

    print(f"\nProcessing {video_file}")

    # clear cache
    torch.cuda.empty_cache()
    gc.collect()

    try:
        # load video
        video_path = os.path.join(video_dir, video_file)
        video_tensor = load_video(video_path, num_segments=8, return_msg=False, resolution=224, hd_num=6)
        video_tensor = video_tensor.to(model.device)

        # obtain sks_name
        sks_name = re.match(r'([a-zA-Z]+)', video_file)
        sks_name = sks_name.group(1) if sks_name else None

        # Have a question-and-answer session
        chat_history = []
        qa_pairs = []
        gender = None
        age = None

        # Add specific questions
        qa_pairs.extend(get_special_qa_pairs(sks_name))

        # Handle other issues
        for question in questions:
           
            print(f"Processing question: {question}")
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
            # If it is a gender or age issue, analyze the result
            if "gender" in question and "age" in question:
                gender, age = parse_gender_age(response)
                same_face_prompts = []
                if gender and age:
                    same_face_prompts = get_prompts_from_class_data(class_data, gender, age)

            else:
                processed_question = process_question(question, sks_name)
                processed_response = get_caption_from_api(response)
                if sks_name:
                    processed_response = replace_person_with_name(processed_response, sks_name)

                qa_pairs.append({
                    "question": processed_question,
                    "answer": processed_response,
                    "is_special": False
                })
                print(f"Answer: {response}")

        # save result
        video_data = {
            "video_name": video_file,
            "video_path": video_path,
            "sks_present": sks_name,
            "gender": gender,
            "age": age,
            "qa_pairs": qa_pairs,
            "same_face_prompt": same_face_prompts
        }

        dataset["videos"].append(video_data)

        # input file
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        # clear cache
        del video_tensor
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        print(f"Error processing {video_file}: {str(e)}")
        continue

print(f"\nProcessing completed. Total videos processed: {len(dataset['videos'])}")