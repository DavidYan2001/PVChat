import os
import json
import argparse
import pdb

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import decord
from decord import VideoReader, cpu
import torch.nn.functional as F
import torchvision.transforms as transforms

decord.bridge.set_bridge("torch")

def get_args():
    parser = argparse.ArgumentParser(description="Build dataset for up to three people.")
    parser.add_argument('--small_dataset', action='store_true',
                        help='Use smaller negative sample count for quick testing')
    parser.add_argument('--trace', action='store_true',
                        help='If True, human_0 is person1, etc.; else reversed.')
    parser.add_argument('--sks1', type=str, default='Sh',
                        help='Name of first person (default: Sh)')
    parser.add_argument('--sks2', type=str, default='Ho',
                        help='Name of second person (default: Ho)')
    parser.add_argument('--sks3', type=str, default='Ho',
                        help='Name of second person (default: Ho)')
    return parser.parse_args()

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
        # Swap dimensions
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
            # You can decide how to handle ties based on your needs
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

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

    if fix_ratio:
        target_aspect_ratio = fix_ratio
    else:
        target_aspect_ratio = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]

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
    resp = response.lower()

    # First check "woman"/"female" then check "man"/"male"
    if "woman" in resp or "female" in resp:
        gender = "female"
    elif "man" in resp or "male" in resp:
        gender = "male"
    else:
        gender = None

    age = None
    if 'middle' in resp:
        age='middle-aged'
    if 'elder' in resp:
        age='elderly'
    age_keywords = ['young', 'middle-aged', 'elderly']
    for keyword in age_keywords:
        if keyword in resp:
            age = keyword
            break

    return gender, age

def parse_age(response):
    age = None
    age_keywords = ['young', 'middle-aged', 'elderly']
    for keyword in age_keywords:
        if keyword in response.lower():
            age = keyword
            break
    return age

def read_train_test_json(train_path, test_path):
    """Read train.json and test.json, return their data structures."""
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    return train_data, test_data

def build_sample_type_map(train_data, test_data):
    """
    Generate a dict: video_name -> sample_type
    Could be 'person1' / 'person2' / 'both' / 'random' / 'three'.
    """
    sample_type_map = {}
    combined = train_data['data'] + test_data['data']
    for item in combined:
        vid_name = item['video_name']
        stype = item.get('sample_type', None)
        sample_type_map[vid_name] = stype
    return sample_type_map

def update_json_files(train_path, test_path, video_data):
    """
    Write gender, age, gender2, age2, gender3, age3 from video_data back to train.json and test.json.
    video_data[video_name] = {
       'gender': ...,
       'age': ...,
       'gender2': ...,
       'age2': ...,
       'gender3': ...,
       'age3': ...
    }
    """
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    with open(test_path, 'r') as f:
        test_data = json.load(f)

    for dataset in (train_data, test_data):
        for item in dataset['data']:
            video_name = item['video_name']
            if video_name in video_data:
                info = video_data[video_name]
                # Update gender, age
                item['gender'] = info.get('gender')
                item['age'] = info.get('age')
                # Add gender2, age2
                item['gender2'] = info.get('gender2')
                item['age2'] = info.get('age2')
                # Add third person
                item['gender3'] = info.get('gender3')
                item['age3'] = info.get('age3')

    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    with open(test_path, 'w') as f:
        json.dump(test_data, f, indent=2)

def inference_on_video(video_path, model, tokenizer):
    """
    Given a single video file path, infer (gender, age) through the InternVideo model.
    If age is not recognized, it will ask again a few times.
    """
    # Load video tensor
    video_tensor = load_video(
        video_path, num_segments=8,
        return_msg=False, resolution=224, hd_num=6
    )
    if torch.cuda.is_available():
        video_tensor = video_tensor.to(model.device)

    # Initiate inference
    chat_history = []
    response, chat_history = model.chat(
        tokenizer,
        '',
        'According to the video, just tell me two words about gender from "male,female" and age from "young,middle-aged,elderly" about the protagonist of the video.',
        media_type='video',
        media_tensor=video_tensor,
        chat_history=chat_history,
        return_history=True,
        generation_config={'do_sample': False}
    )
    gender, age = parse_gender_age(response)

    # If age is not recognized, continue asking, up to 3 times
    retry_count = 0
    while age is None and retry_count < 3:
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
        possible_age = parse_age(response)
        if possible_age is not None:
            age = possible_age
        else:
            retry_count += 1

    print(f"[inference_on_video] -> gender={gender}, age={age}")
    return gender, age

def main():
    args = get_args()

    # Paths
    train_path = "/root/autodl-tmp/yufei/datasets/cekebv-hq/train_"+args.sks1+'_'+args.sks2+args.sks3+'.json'
    test_path = "/root/autodl-tmp/yufei/datasets/cekebv-hq/test_"+args.sks1+'_'+args.sks2+args.sks3+'.json'
    video_dir = "/root/autodl-tmp/yufei/DeepFaceLab/data_src"
    qa_model_path = '/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality/InternVideo2_chat_8B_HD'
    token = "xxxx"

    # Read JSON data to get sample_type for each video
    train_data, test_data = read_train_test_json(train_path, test_path)
    sample_type_map = build_sample_type_map(train_data, test_data)

    # Initialize model
    tokenizer = AutoTokenizer.from_pretrained(
        qa_model_path,
        trust_remote_code=True,
        use_fast=False,
        token=token
    )

    if torch.cuda.is_available():
        model = AutoModel.from_pretrained(
            qa_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).cuda()
    else:
        model = AutoModel.from_pretrained(
            qa_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

    # Dict to collect results: { video_name: {gender, age, gender2, age2, gender3, age3} }
    video_data = {}

    # Get list of all mp4 files
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

    for video_file in video_files:
        print(f"\n=== Processing {video_file}... ===")
        video_path = os.path.join(video_dir, video_file)

        # Check if there are multiple people based on sample_type
        stype = sample_type_map.get(video_file, None)

        # If not found in JSON, skip or give default value
        if stype is None:
            print(f"Warning: {video_file} not found in train/test JSON. Skipping or set default.")
            video_data[video_file] = {
                'gender': None, 'age': None,
                'gender2': None, 'age2': None,
                'gender3': None, 'age3': None
            }
            continue

        if stype == "both":
            # Two people
            video_folder = os.path.splitext(video_file)[0]  # e.g. "Sheldon&Howard2"
            group_dir = os.path.join("/root/autodl-tmp/yufei/DeepFaceLab/output", video_folder, "groups")
            human0_path = os.path.join(group_dir, "human_0.mp4")
            human1_path = os.path.join(group_dir, "human_1.mp4")

            if not os.path.exists(human0_path) or not os.path.exists(human1_path):
                print(f"Warning: Both sub-videos for {video_file} not found. Skipping.")
                video_data[video_file] = {
                    'gender': None, 'age': None,
                    'gender2': None, 'age2': None,
                    'gender3': None, 'age3': None
                }
                continue

            try:
                print(f" -> Inferring {human0_path}")
                g0, a0 = inference_on_video(human0_path, model, tokenizer)
                print(f" -> Inferring {human1_path}")
                g1, a1 = inference_on_video(human1_path, model, tokenizer)

                if args.trace:
                    # human_0->1, human_1->2
                    video_data[video_file] = {
                        'gender': g0, 'age': a0,
                        'gender2': g1, 'age2': a1,
                        'gender3': None, 'age3': None
                    }
                else:
                    # reversed
                    video_data[video_file] = {
                        'gender': g1, 'age': a1,
                        'gender2': g0, 'age2': a0,
                        'gender3': None, 'age3': None
                    }

            except Exception as e:
                print(f"Error processing both for {video_file}: {str(e)}")
                video_data[video_file] = {
                    'gender': None, 'age': None,
                    'gender2': None, 'age2': None,
                    'gender3': None, 'age3': None
                }
            continue

        elif stype == "three":
            # Three people
            video_folder = os.path.splitext(video_file)[0]
            group_dir = os.path.join("/root/autodl-tmp/yufei/DeepFaceLab/output", video_folder, "groups")
            human0_path = os.path.join(group_dir, "human_0.mp4")
            human1_path = os.path.join(group_dir, "human_1.mp4")
            human2_path = os.path.join(group_dir, "human_2.mp4")

            if not (os.path.exists(human0_path) and os.path.exists(human1_path) and os.path.exists(human2_path)):
                print(f"Warning: Not all sub-videos (human_0,1,2) found for {video_file}. Skipping.")
                video_data[video_file] = {
                    'gender': None, 'age': None,
                    'gender2': None, 'age2': None,
                    'gender3': None, 'age3': None
                }
                continue

            try:
                print(f" -> Inferring {human0_path}")
                g0, a0 = inference_on_video(human0_path, model, tokenizer)
                print(f" -> Inferring {human1_path}")
                g1, a1 = inference_on_video(human1_path, model, tokenizer)
                print(f" -> Inferring {human2_path}")
                g2, a2 = inference_on_video(human2_path, model, tokenizer)

                if args.trace:
                    # human_0->1, human_1->2, human_2->3
                    video_data[video_file] = {
                        'gender': g0, 'age': a0,
                        'gender2': g1, 'age2': a1,
                        'gender3': g2, 'age3': a2
                    }
                else:
                    # reversed
                    video_data[video_file] = {
                        'gender': g2, 'age': a2,
                        'gender2': g1, 'age2': a1,
                        'gender3': g0, 'age3': a0
                    }

            except Exception as e:
                print(f"Error processing three-person video {video_file}: {str(e)}")
                video_data[video_file] = {
                    'gender': None, 'age': None,
                    'gender2': None, 'age2': None,
                    'gender3': None, 'age3': None
                }
            continue

        elif stype == "random":
            # This is a negative example, both people/all three may not appear, or recognition may not be needed
            print(f"Video {video_file} is 'random' negative. Skipping inference.")
            video_data[video_file] = {
                'gender': None, 'age': None,
                'gender2': None, 'age2': None,
                'gender3': None, 'age3': None
            }
            continue

        else:
            # stype == 'person1' or 'person2' => normal single person video => continue with original inference
            try:
                gender, age = inference_on_video(video_path, model, tokenizer)
                # Single person video doesn't have gender2, age2, gender3, age3
                video_data[video_file] = {
                    'gender': gender, 'age': age,
                    'gender2': None, 'age2': None,
                    'gender3': None, 'age3': None
                }

            except Exception as e:
                print(f"Error processing {video_file}: {str(e)}")
                # Give default value
                video_data[video_file] = {
                    'gender': None, 'age': None,
                    'gender2': None, 'age2': None,
                    'gender3': None, 'age3': None
                }
                continue

    # Finally, update train.json / test.json
    update_json_files(train_path, test_path, video_data)
    print("Done. All videos processed.")

if __name__ == "__main__":
    main()