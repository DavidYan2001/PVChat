import os
import json
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import re
import decord
from decord import VideoReader, cpu
import torch.nn.functional as F
import torchvision.transforms as transforms

decord.bridge.set_bridge("torch")


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

def load_video(video_path, num_segments=8, return_msg=False, resolution=224, hd_num=6, padding=False):
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
    return frames


def parse_gender_age(response):
    # Parse gender
    gender = None
    if 'male' in response.lower():
        gender = 'male'
    if 'man' in response.lower():
        gender = 'male'
    if 'female' in response.lower():
        gender = 'female'
    if 'woman' in response.lower():
        gender = 'female'

    # 如果没有解析到性别，强制设置为 'male'
    if gender is None:
        gender = 'male'

    # Parse age
    age = None
    age_keywords = ['young', 'middle-aged', 'elderly']
    for keyword in age_keywords:
        if keyword in response.lower():
            age = keyword
            break

    return gender, age
def parse_age(response):
    # Parse gender

    # Parse age
    age = None
    age_keywords = ['young', 'middle-aged', 'elderly']
    for keyword in age_keywords:
        if keyword in response.lower():
            age = keyword
            break

    return  age

def update_json_files(train_path, test_path, video_data):
    # Load and update train.json
    with open(train_path, 'r') as f:
        train_data = json.load(f)

    # Load and update test.json
    with open(test_path, 'r') as f:
        test_data = json.load(f)

    # Update each file
    for data in [train_data, test_data]:
        for item in data['data']:
            video_name = item['video_name']
            if video_name in video_data:
                item['gender'] = video_data[video_name]['gender']
                item['age'] = video_data[video_name]['age']

    # Save updated files
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)

    with open(test_path, 'w') as f:
        json.dump(test_data, f, indent=2)


def main():
    # Initialize model
    token = "xxx"
    model_path = '/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality/InternVideo2_chat_8B_HD'

    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True,
                                              use_fast=False,
                                              token=token)

    if torch.cuda.is_available():
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True).cuda()
    else:
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True)

    # Get list of videos
    video_dir = "/root/autodl-tmp/yufei/DeepFaceLab/data_src"
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

    # Store results
    video_data = {}

    # Process each video
    for video_file in video_files:
        print(f"Processing {video_file}...")
        video_path = os.path.join(video_dir, video_file)

        try:
            # Load video
            video_tensor = load_video(video_path, num_segments=8,
                                      return_msg=False, resolution=224, hd_num=6)
            if torch.cuda.is_available():
                video_tensor = video_tensor.to(model.device)

            # Initialize chat and get response
            chat_history = []
            response, chat_history = model.chat(
                tokenizer,
                '',
                'According to the video and just told me two words about gender from "male, female" and age from "young, middle-aged, elderly" about the protagonist of the video',
                media_type='video',
                media_tensor=video_tensor,
                chat_history=chat_history,
                return_history=True,
                generation_config={'do_sample': False}
            )


            # Parse response
            print(response)
            gender, age = parse_gender_age(response)
            time=0
            while age ==None:
                response, chat_history = model.chat(
                    tokenizer,
                    '',
                    'Looking at the person in the video, which age group does this person belong to - young (under 35), middle-aged (35-60), or elderly (over 60)?',
                    media_type='video',
                    media_tensor=video_tensor,
                    chat_history=chat_history,
                    return_history=True,
                    generation_config={'do_sample': False}
                )
                age =  parse_age(response)
                if age != None:
                    break
                time+=time
                if time>3:
                    print("age is hard to got")
                    break
                print(response)
            # Store results
            video_data[video_file] = {
                'gender': gender,
                'age': age
            }

            print(f"Extracted - Gender: {gender}, Age: {age}")

        except Exception as e:
            print(f"Error processing {video_file}: {str(e)}")
            continue

    # Update JSON files
    train_path = "/root/autodl-tmp/yufei/datasets/cekebv-hq/train.json"
    test_path = "/root/autodl-tmp/yufei/datasets/cekebv-hq/test.json"
    update_json_files(train_path, test_path, video_data)


if __name__ == "__main__":
    main()