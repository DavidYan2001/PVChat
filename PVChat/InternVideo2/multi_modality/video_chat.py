import os
import pdb

#token = os.environ['HF_TOKEN']
token =   os.environ['HF_TOKEN']
import torch

from transformers import AutoTokenizer, AutoModel

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
def parse_gender_age(response):
    # 解析性别
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

from decord import VideoReader, cpu
from PIL import Image
import numpy as np
import numpy as np
import decord
from decord import VideoReader, cpu
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import PILToTensor
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
decord.bridge.set_bridge("torch")

def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets


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

#NZ test
chat_history = []

video_path = "/root/autodl-tmp/yufei/DeepFaceLab/all_kind_of_data/good_doctor_3people/Shaun/Shaun7.mp4"
video_tensor = load_video(video_path, num_segments=8, return_msg=False, resolution=224, hd_num=6)
video_tensor = video_tensor.to(model.device)
response, chat_history = model.chat(tokenizer, '', "This is video about <Xo>", media_type='video', media_tensor=video_tensor, chat_history= chat_history, return_history=True,generation_config={'do_sample':False})
print(response)

video_path = "/root/autodl-tmp/yufei/DeepFaceLab/all_kind_of_data/good_doctor_3people/Jared/Jared4.mp4"
video_tensor = load_video(video_path, num_segments=8, return_msg=False, resolution=224, hd_num=6)
video_tensor = video_tensor.to(model.device)
response, chat_history = model.chat(tokenizer, '', "This is video about <Ja>", media_type='video', media_tensor=video_tensor, chat_history= chat_history, return_history=True,generation_config={'do_sample':False})
print(response)

video_path = "/root/autodl-tmp/yufei/DeepFaceLab/all_kind_of_data/good_doctor_3people/cl&xo&ja/cl_xo_ja2.mp4"
video_tensor = load_video(video_path, num_segments=8, return_msg=False, resolution=224, hd_num=6)
video_tensor = video_tensor.to(model.device)
response, chat_history = model.chat(tokenizer, '', "Can you detect <Cl>, <Xo>, or <Ja> in this video clip?", media_type='video', media_tensor=video_tensor, chat_history= chat_history, return_history=True,generation_config={'do_sample':False})
print( "Can you detect <Cl>, <Xo>, or <Ja> in this video clip?")
print(response)

video_path = "/root/autodl-tmp/yufei/DeepFaceLab/all_kind_of_data/good_doctor_3people/cl&xo&ja/cl_xo_ja2.mp4"
video_tensor = load_video(video_path, num_segments=8, return_msg=False, resolution=224, hd_num=6)
video_tensor = video_tensor.to(model.device)
response, chat_history = model.chat(tokenizer, '', "If I want to write a ins style poster about <Cl>,<Xo> and <Ja> doing today, can you give me a detailed example?", media_type='video', media_tensor=video_tensor, chat_history= chat_history, return_history=True,generation_config={'do_sample':False})
print("If I want to write a ins style poster about <Cl>,<Xo> and <Ja> doing today, can you give me a detailed example?")
print(response)

video_path = "/root/autodl-tmp/yufei/DeepFaceLab/all_kind_of_data/good_doctor_3people/cl&xo&ja/cl_xo_ja2.mp4"
video_tensor = load_video(video_path, num_segments=8, return_msg=False, resolution=224, hd_num=6)
video_tensor = video_tensor.to(model.device)
response, chat_history = model.chat(tokenizer, '', "If I want to buy three gift for <Cl>,<Xo> and <Ja>, can you give me some idea?", media_type='video', media_tensor=video_tensor, chat_history= chat_history, return_history=True,generation_config={'do_sample':False})
print("If I want to buy three gift for <Cl>,<Xo> and <Ja>, can you give me some idea?")
print(response)
pdb.set_trace()


video_path = "/root/autodl-tmp/yufei/DeepFaceLab/all_kind_of_data/NZAB/NZAB/Nzab5.mp4"
video_tensor = load_video(video_path, num_segments=8, return_msg=False, resolution=224, hd_num=6)
video_tensor = video_tensor.to(model.device)
response, chat_history = model.chat(tokenizer, '', "Could you describe what <Nz> and <Ab> is doing during this video if you could find they are in this video??", media_type='video', media_tensor=video_tensor, chat_history= chat_history, return_history=True,generation_config={'do_sample':False})
print(response)
pdb.set_trace()



