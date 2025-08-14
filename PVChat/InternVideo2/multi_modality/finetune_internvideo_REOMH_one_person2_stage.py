import pdb
import random  # For random.choice
import os
import shutil
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import json
from pathlib import Path

import logging
import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.utils.rnn as rnn_utils
from decord import VideoReader, cpu
from PIL import Image
import numpy as np
import decord
import warnings
from decord import VideoReader, cpu
from random import shuffle
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import PILToTensor
from torchvision import transforms
from transformers import AutoConfig
from torchvision.transforms.functional import InterpolationMode
import re
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()  # Use root logger directly


class LayerLossLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.pending_loss_data = []
        self.layer_data = {}
        self.current_layer = None
        self.all_loss_re = re.compile(
            r"all_loss:([\d.eE+-]+),\s*lb_loss:([\d.eE+-]+),\s*lamdba:\s*([\d.eE+-]+)\s*,\s*active_rate:([\d.eE+-]+),\s*alignment_loss:([\d.eE+-]+)\s*,\s*alignment_weight:([\d.eE+-]+)"
        )
        self.layer_re = re.compile(r"\[Layer (\d+)\]")

    def emit(self, record):
        msg = record.getMessage()
        m_all = self.all_loss_re.search(msg)
        if m_all:
            try:
                loss_entry = {
                    "all_loss": float(m_all.group(1)),
                    "lb_loss": float(m_all.group(2)),
                    "lamdba": float(m_all.group(3)),
                    "active_rate": float(m_all.group(4)),
                    "alignment_loss": float(m_all.group(5)),
                    "alignment_weight": float(m_all.group(6))
                }
            except ValueError:
                return
            self.pending_loss_data.append(loss_entry)
            return

        m_layer = self.layer_re.search(msg)
        if m_layer:
            try:
                layer_num = int(m_layer.group(1))
            except ValueError:
                return
            if layer_num < 6:
                self.current_layer = layer_num
                self.pending_loss_data = []
                return

            self.current_layer = layer_num
            if self.pending_loss_data:
                if layer_num not in self.layer_data:
                    self.layer_data[layer_num] = {"self": [], "cross": []}
                if len(self.pending_loss_data) >= 1:
                    self.layer_data[layer_num]["self"].append(self.pending_loss_data[0])
                if len(self.pending_loss_data) > 1:
                    self.layer_data[layer_num]["cross"].extend(self.pending_loss_data[1:])
                self.pending_loss_data = []
            return

    def dump_to_file(self, filename):
        with open(filename, "w") as f:
            json.dump(self.layer_data, f, indent=4)

layer_handler = LayerLossLogHandler()
logger.addHandler(layer_handler)

decord.bridge.set_bridge("torch")
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'
warnings.filterwarnings("ignore", message="Keyword arguments {'add_special_tokens': False} not recognized.")
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.")
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)
IMG_TOKEN = "[<IMG_PLH>]"
VID_TOKEN = "[<VID_PLH>]"

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_BOS_TOKEN = '<s>'
DEFAULT_EOS_TOKEN = '</s>'
DEFAULT_UNK_TOKEN = "<unk>"

DEFAULT_IMAGE_TOKEN = "[IMAGETOKEN]"
DEFAULT_VIDEO_TOKEN = "[VIDEOTOKEN]"

DEFAULT_IMG_PLACEHOLDER = "[<IMG_PLH>]"
DEFAULT_VID_PLACEHOLDER = "[<VID_PLH>]"
# Set HF_TOKEN environment variable
os.environ["HF_TOKEN"] =os.environ['HF_TOKEN']


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
        1, T_, 3, H // resolution, resolution, W // resolution, resolution
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


def HD_transform_no_padding(frames, image_size=224, hd_num=6, fix_ratio=(2, 1)):
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


class PersonalizedVideoDataset(Dataset):
    def __init__(
            self,

            json_path: str,  # /root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality/<yufei>.json
            tokenizer,
            device,
            config,
            split='train'  # Can add train/val split
    ):
        # Load json file
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.videos = data['videos']
        self.sks_name = Path(json_path).stem
        # 获取sks名称 (从json路径中提取)
        #  self.sks_name = args.sks_name
        self.split = split
        self.tokenizer = tokenizer
        self.config=config
        self.all_qa_pairs = self.flatten_qa_pairs(self.videos)
        # 如果需要划分训练集和验证集
        # print(self.all_qa_pairs )

    def __len__(self):
        return len(self.all_qa_pairs)

    def flatten_qa_pairs(self, videos):
        all_qa = []
        for video_idx, video in enumerate(videos):
            for qa_pair in video['qa_pairs']:
                all_qa.append((video_idx, qa_pair))
        return all_qa

    def __getitem__(self, idx):
        video_idx, qa_pair = self.all_qa_pairs[idx]
        video_data = self.videos[video_idx]

        # 加载视频
        video_tensor = load_video(
            video_data['video_path'],
            num_segments=8,
            return_msg=False,
            resolution=224,
            hd_num=6
        )

        # Randomly select a QA pair
        # qa_pair = random.choice(video_data['qa_pairs'])
        # Build conversation
        question = qa_pair['question']
        answer = qa_pair['answer']
        is_special = qa_pair['is_special']
        # Replace placeholders in question
        # question = qa_pair['question']#.replace(f'<{self.sks_name}>', self.sks_name)
        # answer = qa_pair['answer']#.replace(f'<{self.sks_name}>', f'<{self.sks_name}>')

        if self.split == 'train':
            # 构建对话格式 - 与chat函数保持一致
            conversation = ""
            # 注意：训练时可能不需要instruction，但保持格式统一
            # if instruction:
            #     conversation += instruction

            # print("video_tensor.shape_1",video_tensor.shape)
            conversation += "[INST] "

            # 添加视频token - 使用相同的格式
            if video_tensor.shape[1] == 1:  # 单帧，当作图像
                ilen = video_tensor.shape[0]
                conversation += ("<Image>" + IMG_TOKEN + "</Image>") * ilen
            else:  # 视频
                ilen = video_tensor.shape[1]
                conversation += ("<Video>" + VID_TOKEN + "</Video>") * ilen
            conversation += "[/INST] "
            # 添加问题和历史对话（如果需要）
           # prefix_tokens = [f'<token{i}>' for i in range(config.model_config["bridge"]["num_prefix_token"])]
            sks_and_tokens = f"[{self.sks_name}]" + 'is' + ''.join([f"<token{i}>" for i in range(self.config.model_config["bridge"]["num_prefix_token"])]) + " "
            if self.config.model_config["bridge"]["num_prefix_token"]==0:
                sks_and_tokens = f"[{self.sks_name}]"
            conversation += "[INST]" + sks_and_tokens + question + "[/INST]"

            # 添加答案
            conversation += answer + "</s>"

            # 3. 使用build_input_ids处理文本
            tokenized = build_input_ids(
                self.tokenizer,
                conversation,
                max_length=512,
                add_special_tokens=True,
                truncation=True,
                padding='longest',
                return_tensors='pt',
                video_placeholder="[<VID_PLH>]"

            )

            # 准备labels（用于计算生成损失）

            labels = tokenized['input_ids'].clone()
            # print("labels",labels)
            # inst0_tokens = self.tokenizer.encode("[INST]", add_special_tokens=False)
            # Video0_tokens = self.tokenizer.encode("<Video>", add_special_tokens=False)
            # Video_tokens = self.tokenizer.encode("/<Video>", add_special_tokens=False)
            # VID_TOKEN_tokens = self.tokenizer.encode(VID_TOKEN, add_special_tokens=False)
            inst0_tokens = self.tokenizer.encode("[INST]")
            Video0_tokens = self.tokenizer.encode("<Video>")
            Video_tokens = self.tokenizer.encode("/<Video>")
            VID_TOKEN_tokens = self.tokenizer.encode(VID_TOKEN)
            # print("inst0_tokens", inst0_tokens) #[29473, 3]
            # print("Video0_tokens ", Video0_tokens) #[1291, 12529, 29535]
            # print("Video_tokens", Video_tokens) # [1500, 29557, 12529, 29535]
            # print("VID_TOKEN_tokens", VID_TOKEN_tokens) #[1501, 29557, 7568, 29498, 4666, 29537, 29535, 29561]

            # print("labels.shape",labels.shape) # torch.Size([411])
            # 找到第二个[/INST]的位置（问题后面的那个）
            #inst_tokens = self.tokenizer.encode("[/INST]", add_special_tokens=False)
            inst_tokens = self.tokenizer.encode("[/INST]")#
            # print("inst_tokens",inst_tokens) #[29473, 4]29473是"[/"的token ID 4是"INST]"的token ID
            # print("torch.where(labels == inst_tokens[-1])",torch.where(labels == inst_tokens[-1])) # (tensor([358, 393]),)
            inst_end_indices = torch.where(labels == inst_tokens[-1])[0]  # 所以这里相当于只是选了第一个
            # print("inst_end_indices", inst_end_indices) #[358, 393]
            second_inst_end = inst_end_indices[1]  # 第二个[/INST]的位置
            # print("second_inst_end",second_inst_end) #393
            # 将问题部分和视频token部分的label设为-100
            labels[:second_inst_end + 1] = -100  # 包括第二个[/INST]之前的所有内容
            labels[tokenized['index']] = -100  # 视频token位置

        else:
            conversation = ""
            # 注意：训练时可能不需要instruction，但保持格式统一
            # if instruction:
            #     conversation += instruction

            # print("video_tensor.shape_1",video_tensor.shape)
            conversation += "[INST] "

            # 添加视频token - 使用相同的格式
            if video_tensor.shape[1] == 1:  # 单帧，当作图像
                ilen = video_tensor.shape[0]
                conversation += ("<Image>" + IMG_TOKEN + "</Image>") * ilen
            else:  # 视频
                ilen = video_tensor.shape[1]
                conversation += ("<Video>" + VID_TOKEN + "</Video>") * ilen
            conversation += "[/INST] "
            # 添加问题和历史对话（如果需要）
            sks_and_tokens = f"[{self.sks_name}]" + 'is' + ''.join(
                [f"<token{i}>" for i in range(self.config.model_config["bridge"]["num_prefix_token"])]) + " "
            if self.config.model_config["bridge"]["num_prefix_token"] == 0:
                sks_and_tokens = f"[{self.sks_name}]"
            conversation += "[INST]" + sks_and_tokens + question + "[/INST]"
            # 3. 使用build_input_ids处理文本
            tokenized = build_input_ids(
                self.tokenizer,
                conversation,
                max_length=512,
                add_special_tokens=False,
                truncation=True,
                padding='longest',
                return_tensors='pt',
                video_placeholder="[<VID_PLH>]"

            )
            labels = None  # 测试时不需要labels
        return_dict = {
            'video': video_tensor,
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'video_idx': tokenized['index'],
            'question': question,
            'answer': answer,
            'is_special': is_special,
            'video_path': video_data['video_path']
        }

        if labels is not None:
            return_dict['labels'] = labels
            return_dict['is_special'] = qa_pair.get('is_special', False)
            return_dict['sks_present'] = video_data['sks_present'] == f'<{self.sks_name}>'
        return return_dict


def collate_fn(batch):
    """
    整理batch数据
    """
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    video_idx = [item['video_idx'] for item in batch]
    videos = [item['video'] for item in batch]
    questions = [item['question'] for item in batch]
    answer = [item['answer'] for item in batch]
    is_special = [item['is_special'] for item in batch]
    video_path = [item['video_path'] for item in batch]
    # 基础数据处理
    input_ids_padded = rnn_utils.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask_padded = rnn_utils.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    video_idx_padded = rnn_utils.pad_sequence(video_idx, batch_first=True, padding_value=0)
    videos = torch.stack([video.squeeze(0) if video.shape[0] == 1 else video for video in videos])

    # 准备返回字典
    batch_dict = {
        'video': videos,
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded,
        'video_idx': video_idx_padded,
        'question': questions,
        'answer': answer,
        'is_special': is_special,
        'video_path': video_path
    }
    # 训练模式特有的字段
    if 'labels' in batch[0]:
        labels = [item['labels'] for item in batch]
        labels_padded = rnn_utils.pad_sequence(labels, batch_first=True, padding_value=-100)
        labels_padded = labels_padded.to(torch.int64)
        batch_dict['labels'] = labels_padded

        # 其他训练特有字段
        batch_dict['is_special'] = torch.tensor([item['is_special'] for item in batch])
        batch_dict['sks_present'] = torch.tensor([item['sks_present'] for item in batch])

    return batch_dict


# 使用方式
def create_dataloader(args):
    dataset = PersonalizedVideoDataset(
        json_path=f"/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality/{args.sks_name}.json",
        tokenizer=tokenizer,
        device=model.device
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )

    return dataloader


def build_input_ids(tokenizer, conversation, max_length, add_special_tokens,
                    truncation, video_placeholder="[<VID_PLH>]", padding=False,
                    return_tensors="pt"):
    input_ids = []
    indexs = []
    attention_mask = []
    start, total_len = 0, 0

    while True:
        # Find video placeholder
        index = conversation.find(video_placeholder, start)

        if index == -1:  # Process the last text part
            inputs = tokenizer(
                conversation[start:],
                max_length=max_length - total_len,
                truncation=truncation,
                padding=padding,
                return_tensors=return_tensors
            )
            # Add text part
            input_ids.append(inputs.input_ids[0])
            attention_mask.append(inputs.attention_mask[0])
            indexs.append(torch.zeros_like(inputs.input_ids[0]))
            break

        else:  # Process text before placeholder
            inputs = tokenizer(
                conversation[start:index],
                max_length=max_length,
                truncation=truncation,
                padding=padding,
                return_tensors=return_tensors
            )

            # Add text part
            input_ids.append(inputs.input_ids[0])
            attention_mask.append(inputs.attention_mask[0])
            indexs.append(torch.zeros_like(inputs.input_ids[0]))

            # Add video token position

            input_ids.append(torch.zeros(96 + 16))
            attention_mask.append(torch.ones(96 + 16))
            indexs.append(torch.ones(96 + 16))

            start = index + len(video_placeholder)

    # Concatenate all parts
    input_ids = torch.cat(input_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    indexs = torch.cat(indexs, dim=0).to(torch.bool)

    # Print shape of each part
    # print(f"Input IDs shape: {input_ids.shape}")
    # print(f"Attention mask shape: {attention_mask.shape}")
    # print(f"Index shape: {indexs.shape}")
    # Connect all parts
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'index': indexs
    }


def get_args():
    parser = argparse.ArgumentParser()
    # Model related
    parser.add_argument("--model_path", type=str,
                        default="/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality/Internvideo2_chat_8B_HD_finetune_REMOH")
    parser.add_argument("--sks_name", type=str, default="<Me>")
    parser.add_argument("--num_personal_token", type=int, default=16)

    # Training related
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--personal_token_lr", type=float, default=1e-4)
    parser.add_argument("--qformer_lr", type=float, default=1e-5)
    parser.add_argument("--lora_lr", type=float, default=1e-5)

    # Data related
    parser.add_argument("--data_root", type=str,
                        default="/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality")

    # New args for file paths
    parser.add_argument("--train_json", type=str, default=None,
                        help="Path to the full training json file. If not provided, will use {data_root}/{sks_name}.json")
    parser.add_argument("--short_train_json", type=str, default=None,
                        help="Path to the short training json file. If not provided, will use {data_root}/{sks_name}_short_train.json")
    parser.add_argument("--test_json", type=str, default=None,
                        help="Path to the test json file. If not provided, will use {data_root}/{sks_name}test.json")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save test results. If not provided, will use {data_root}")

    # Log related
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")



    parser.add_argument("--log_steps", type=int, default=8)
    parser.add_argument("--save_epochs", type=int, default=4)
    # 添加新参数
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"],
                        help="train: train and test; test: only test")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to checkpoint for testing")

    args = parser.parse_args()
    return args


def ensure_complete_config(config, reference_config_path):
    """
    自动检测并补充 config 中的缺失字段，基于参考 JSON 配置文件。

    Args:
        config: 当前加载的配置对象 (AutoConfig 实例)。
        reference_config_path: 参考配置文件路径 (JSON 文件)。

    Returns:
        补充后的 config 对象。
    """
    with open(reference_config_path, "r") as f:
        reference_config = json.load(f)  # 加载完整的 JSON 配置

    # 遍历参考配置文件，补充缺失的字段
    for key, value in reference_config.items():
        if not hasattr(config, key) or getattr(config, key, None) is None:
            print(f"补充字段: {key} -> {value}")
            setattr(config, key, value)  # 动态添加缺失字段

    return config


def train_epoch(model, train_loader, optimizer, epoch, writer, orig_embeds, tokenizer, device, sks_tokens,
                prefix_tokens):
    model.train()
    total_loss = 0
    print("Vocabulary size:", len(tokenizer))  # Vocabulary size: 32777

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}') as pbar:
        for step, batch in enumerate(train_loader):
            # print("Video token positions:", batch['input_ids'][batch['video_idx']])
            # print("Labels:", batch['labels'].unique())
            # #tensor([ -100, 2,  1040,  1065,  1072,  1083,  1117,  1164,  1224,  1274,  1717,
            # # 1800,  4566,  4775,  6355, 29417, 29473, 29475, 29491, 29493,
            # print("Video_idx shape:", batch['video_idx'].shape)   #Video_idx shape: torch.Size([2, 433])
            # print("attention_mask shape:", batch['attention_mask'].shape) #attention_mask shape: torch.Size([2, 433])
            # print("Input_ids shape:", batch['input_ids'].shape)    #Input_ids shape: torch.Size([2, 433])
            # print("Video_idx unique values:", batch['video_idx'].unique()) #Video_idx unique values: tensor([False,  True])
            # print("Video_idx shape", batch['video_idx'].shape) #Video_idx shape torch.Size([2, 433])
            # valid_range = (batch['labels'] >= 0) & (batch['labels'] < len(tokenizer))
            # print("Invalid labels:", batch['labels'][~valid_range])
            batch = {key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in
                     batch.items()}
            # 1. 前向传播

            outputs = model(
                input_ids=batch['input_ids'].cuda(),
                attention_mask=batch['attention_mask'].cuda(),
                video=batch['video'].cuda(),
                labels=batch['labels'].cuda(),
                video_idx=batch['video_idx'].cuda()
            )

            loss = outputs.loss
            total_loss += loss.item()

            # 2. 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 3. 保持非个性化token的embedding不变
            with torch.no_grad():
                # 获取特殊token的索引
                special_token_ids = tokenizer.convert_tokens_to_ids(
                    sks_tokens + prefix_tokens
                )
                # 假设我们在初始化时保存了原始权重
                current_embeds = model.get_input_embeddings().weight.data

                # 只保持非特殊token的embedding
                keep_indices = torch.ones(current_embeds.size(0), dtype=torch.bool)
                keep_indices[special_token_ids] = False
                current_embeds[keep_indices] = orig_embeds[keep_indices]  #相当于选中了其他的不是我这些sks和prefix的token，还原，

            # 4. 更新进度条和记录
            pbar.update(1)
            pbar.set_postfix({'loss': loss.item()})

            # 5. 记录到tensorboard
            global_step = epoch * len(train_loader) + step
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Loss/token-norm', model.get_input_embeddings().weight[special_token_ids].norm().item(),
                              global_step)

    return total_loss / len(train_loader)


def train(model, config, tokenizer, sks_tokens, prefix_tokens):
    args = get_args()
    if args.output_dir:
        save_dir = Path(args.output_dir) / "checkpoints" / args.sks_name
        log_dir = Path(args.output_dir) / "logs" / args.sks_name
    else:
        save_dir = Path(args.save_dir) / args.sks_name
        log_dir = Path(args.log_dir) / args.sks_name
    # 设置保存目录
    # save_dir = Path(args.save_dir) / args.sks_name
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)
    #writer = SummaryWriter(Path(args.log_dir) / args.sks_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 2. 添加特殊tokens

    # 3. 加载模型

    # 构造两个数据集对象
    # 第一阶段：使用短视频数据 (_short_train.json)
    if args.train_json:
        full_train_json = args.train_json
    else:
        full_train_json = os.path.join(args.data_root, f"{args.sks_name}.json")

    # Short training data path
    if args.short_train_json:
        short_train_json = args.short_train_json
    else:
        short_train_json = os.path.join(args.data_root, f"{args.sks_name}_short_train.json")

    # 第二阶段：使用原始数据 (<Pe>.json)
    #full_train_json = os.path.join(args.data_root, f"{args.sks_name}.json")

    # 这里我们先检查短数据集是否存在，如果不存在则回退到原始数据
    if os.path.exists(short_train_json):
        short_train_dataset = PersonalizedVideoDataset(

            json_path=short_train_json,
            tokenizer=tokenizer,
            device=model.device,
            config=config,
            split='train'
        )
        print(f"Using short training dataset: {short_train_json}")
    else:
        print(f"Short training dataset not found, using full dataset for all epochs.")
        short_train_dataset = None
    # 4. 准备数据加载器


    train_dataset = PersonalizedVideoDataset(
        json_path=full_train_json,
        tokenizer=tokenizer,
        device=model.device,
        config=config,
        split='train'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )

    # 5. 准备优化器
    # 保存原始embedding
    # orig_embeds = model.lm.model.embed_tokens.weight.data.clone()
    orig_embeds = model.get_input_embeddings().weight.data.clone()
    # 获取原始的词嵌入权重
    # 克隆一份，以便后续用于恢复那些我们不想修改的 token 的 embedding
    # 用于在训练过程中保持非个性化 token 的 embedding 不变：

    # 设置需要训练的参数
    # 获取新增token的索引
    sks_token_ids = tokenizer.convert_tokens_to_ids(sks_tokens + prefix_tokens)
    moh_params = []
    qformer_params = []
    for name, param in model.named_parameters():
        if 'router' in name or 'alpha_proj' in name:
            moh_params.append(param)
    for name, param in model.qformer.named_parameters():
        if 'router' not in name and 'alpha_proj' not in name:
            qformer_params.append(param)

    trainable_params = [
        {'params': model.personal_query_tokens, 'lr': config.personal_token_lr},  # 使用属性访问
        {'params': model.get_input_embeddings().weight[sks_token_ids], 'lr': config.token_lr},
        {'params': qformer_params, 'lr': config.qformer_lr},
        {'params': moh_params, 'lr': 1e-6},
        #{'params': model.qformer.parameters(), 'lr': config.qformer_lr},
    ]
    # 这里再加一组

    lora_params = [(n, p.shape) for n, p in model.named_parameters() if 'lora' in n.lower()]
    # print("Found LoRA parameters:")
    # for name, shape in lora_params:
    #     print(f"{name}: {shape}")
    # pdb.set_trace()
    if config.model_config['llm']['use_lora']:
        trainable_params.append(
            {'params': [p for n, p in model.named_parameters() if 'lora' in n.lower()],
             'lr': config.lora_lr}
        )
    optimizer = torch.optim.AdamW(trainable_params)

    # 6. 设置日志
    sks_name = getattr(config, 'sks_name', 'default_name')
    writer = SummaryWriter(f'runs/{sks_name.strip("<>")}')
    # save_dir = Path(f'checkpoints/{sks_name.strip("<>")}')
    # save_dir.mkdir(parents=True, exist_ok=True)

    # 7. 训练循环
    best_acc = 0
    # for name, param in model.named_parameters():
    #     if 'router' in name or 'alpha_proj' in name:
    #         print(name, param.requires_grad)
    #num of blocks
    num_blocks = len(model.vision_encoder.blocks)
    last_block_idx = num_blocks - 1

    writer = SummaryWriter(log_dir="/root/tf-logs/")
    for epoch in range(args.num_epochs):

        if epoch == 0 and short_train_dataset is not None:
            current_dataset = short_train_dataset
            print("Epoch 0: using short_train dataset for coarse training.")
        else:
            #stage2
            current_dataset = train_dataset
            if epoch == 1 and short_train_dataset is not None:
                print("Epoch >=1: switching to full dataset for fine training.")

            for i, block in enumerate(model.vision_encoder.blocks):
                if i >= (num_blocks // 2):
                    for param in block.parameters():
                        param.requires_grad = True
                else:
                    for param in block.parameters():
                        param.requires_grad = False
            # Optionally unfreeze the final aggregator
            if hasattr(model.vision_encoder, 'clip_projector'):
                for param in model.vision_encoder.clip_projector.parameters():
                    param.requires_grad = True


        train_loader = DataLoader(
            current_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4
        )
        # 训练一个epoch
        train_loss = train_epoch(model, train_loader, optimizer, epoch, writer, orig_embeds, tokenizer, device,
                                 sks_tokens, prefix_tokens)
        # 程序结束前可以将捕捉到的数据保存到文件

        # 定期保存checkpoint
        if (epoch + 1) % args.save_epochs == 0:
            checkpoint_dir = save_dir / f'checkpoint_epoch_{epoch + 1}'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_dir / 'pytorch_model.bin')
            # 2. 保存tokenizer
            tokenizer.save_pretrained(checkpoint_dir)

            # 3. 保存配置
            config.save_pretrained(checkpoint_dir)  # 会保存config.json
            tokenizer.save_pretrained(checkpoint_dir)  # 会保存tokenizer相关文件
            extra_info = {
                'sks_tokens': sks_tokens,
                'prefix_tokens': prefix_tokens,
                'optimizer_state_dict': optimizer.state_dict(),
                'training_args': {
                    'batch_size': config.batch_size,
                    'personal_token_lr': config.personal_token_lr,
                    'qformer_lr': config.qformer_lr,
                    'lora_lr': config.lora_lr,
                    'use_lora': config.model_config['llm']['use_lora']
                }
            }
            torch.save(extra_info, checkpoint_dir / 'training_info.bin')
            # 4. 保存训练参数
            src_dir = Path(config_path)
            for file_name in ['modeling_base.py', 'modeling_internvideo2.py', 'modeling_videochat2.py'
                                                                              'modeling_qformer_MOH.py',
                              'flash_attention_class.py', 'modeling_internvideo2_vit.py']:
                src_file = src_dir / file_name
                if src_file.exists():
                    shutil.copy2(src_file, checkpoint_dir / file_name)

            with open(checkpoint_dir / 'training_args.json', 'w') as f:
                json.dump({
                    'batch_size': config.batch_size,
                    'personal_token_lr': config.personal_token_lr,
                    'qformer_lr': config.qformer_lr,
                    'lora_lr': config.lora_lr,
                    'use_lora': config.model_config['llm']['use_lora'],
                    'sks_tokens': sks_tokens,
                    'prefix_tokens': [t for t in prefix_tokens]
                }, f, indent=4)
        print(f'Epoch {epoch}: Loss = {train_loss:.4f}')
    if args.output_dir:
        save_file_for_loss_path = os.path.join(args.output_dir, f"{config.sks_name.strip('<>')}_layer_loss_data.json")
    else:
        save_file_for_loss_path = os.path.join(args.data_root, f"{config.sks_name.strip('<>')}_layer_loss_data.json")
    #save_file_for_loss_path = "/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality/" + config.sks_name.strip("<>") + "_layer_loss_data.json"
    layer_handler.dump_to_file(save_file_for_loss_path)

    final_save_dir = save_dir / 'final_model'
    final_save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), final_save_dir / 'pytorch_model.bin')

    config.save_pretrained(final_save_dir)  # 会保存config.json
    tokenizer.save_pretrained(final_save_dir)  # 会保存tokenizer相关文件

    # 2. 保存模型权重和safetensors文件
    # 4. 保存额外的训练信息
    extra_info = {
        'sks_tokens': sks_tokens,
        'prefix_tokens': prefix_tokens,
        'optimizer_state_dict': optimizer.state_dict(),
        'training_args': {
            'batch_size': config.batch_size,
            'personal_token_lr': config.personal_token_lr,
            'qformer_lr': config.qformer_lr,
            'lora_lr': config.lora_lr,
            'use_lora': config.model_config['llm']['use_lora']
        }
    }
    torch.save(extra_info, final_save_dir / 'training_info.bin')
    # 3. 保存其他必要的文件
    src_dir = Path(config_path)
    for file_name in ['modeling_base.py', 'modeling_internvideo2.py', 'modeling_videochat2.py'
                                                                      'modeling_qformer_MOH.py', 'flash_attention_class.py',
                      'modeling_internvideo2_vit.py']:
        src_file = src_dir / file_name
        if src_file.exists():
            shutil.copy2(src_file, final_save_dir / file_name)

    # 4. 保存训练参数
    with open(final_save_dir / 'training_args.json', 'w') as f:
        json.dump({
            'batch_size': config.batch_size,
            'personal_token_lr': config.personal_token_lr,
            'qformer_lr': config.qformer_lr,
            'lora_lr': config.lora_lr,
            'use_lora': config.model_config['llm']['use_lora'],
            'sks_tokens': sks_tokens,
            'prefix_tokens': [t for t in prefix_tokens]
        }, f, indent=4)

    return final_save_dir


def test(model, tokenizer, test_path, device):
    if args.test_json:
        test_json_path = args.test_json
    else:
        test_json_path = os.path.join(args.data_root, f"{config.sks_name}test.json")
    print(f"Loading test data from: {test_json_path}")
    test_dataset = PersonalizedVideoDataset(
        json_path=test_json_path,
        tokenizer=tokenizer,
        device=device,
        config=config,
        split='test'
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # 单个问题逐个测试
        shuffle=False,
        collate_fn=collate_fn
    )

    model.eval()

    # 使用字典来按video_path分组存储结果
    video_results = {}
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch['input_ids'], tuple):
                batch['input_ids'] = torch.stack(batch['input_ids'])
            batch = {key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in
                     batch.items()}
            # 调用 generate_caption 生成答案

            outputs = model.generate_caption(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                video=batch['video'],  # 如果是视频
                video_idx=batch['video_idx'],  # 视频 token 位置索引
                num_beams=5,
                max_new_tokens=512,
                top_k=50,
                top_p=0.95
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            question = batch['question'][0]  # 获取原问题
            answer = batch['answer'][0]  # 获取原问题
            is_special = batch['is_special'][0]  # 假设is_special是标量或张量
            video_path = batch['video_path'][0]  # 假设video_path是字符串
            # 创建QA对
            qa_pair = {
                'question': question,
                'answer': answer,
                'generated_answer': generated_text,
                'is_special': is_special
            }
            # 按video_path分组存储
            if video_path not in video_results:
                video_results[video_path] = {
                    'video_path': video_path,
                    'qa_pairs': []
                }
            video_results[video_path]['qa_pairs'].append(qa_pair)
            # 打印进度信息
            print(f"Question: {question}")
            print(f"Generated Answer: {generated_text}")
            print(f"Original Answer: {answer}")
            print(f"Video Path: {video_path}")
            print("-" * 50)

    # 将字典转换为列表格式
    final_results = list(video_results.values())
    # 打印结果
    # 保存结果到json文件
    output = {
        "model_name": test_path.strip("<>").replace("test.json", ""),  # 从test_path提取模型名称
        "results": final_results
    }
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(test_json_path)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    #output_path = os.path.join(os.path.dirname(test_path), f"test_results_RE_MOH{output['model_name']}1person_facny.json")
    output_path = os.path.join(output_dir, f"test_results_RE_MOH_{output['model_name']}_1person.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {output_path}")
    #  return results
    if final_results:
        example = final_results[0]
        print(f"Video Path: {example['video_path']}")
        print("First QA pair:")
        print(json.dumps(example['qa_pairs'][0], indent=2, ensure_ascii=False))

    return final_results


def evaluate(model, val_loader, device):
    """评估函数"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            # 只评估特殊问题（是否存在目标人物）的准确率
            if batch['is_special'].any():
                outputs = model(
                    input_ids=batch['input_ids'].cuda(),
                    attention_mask=batch['attention_mask'].cuda(),
                    video=batch['video'].cuda(),
                    video_idx=batch['video_idx'].cuda()
                )

                special_idx = batch['is_special'].nonzero().squeeze()
                pred = outputs.logits[special_idx].argmax(-1)
                correct += (pred == batch['sks_present'][special_idx].cuda()).sum().item()
                total += len(special_idx)

    return correct / total if total > 0 else 0


if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    if args.mode == "train":
        config_path = "/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality/Internvideo2_chat_8B_HD_finetune_REMOH"
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
        reference_config_path = config_path + "/config.json"  # 原始 JSON 配置文件
        config.sks_name=args.sks_name
        # 自动补充缺失字段
        config = ensure_complete_config(config, reference_config_path)
        # 1. 初始化模型和tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config_path,
            trust_remote_code=True,
            use_fast=False
        )
        model = AutoModel.from_pretrained(
            config_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).cuda()


        # 调整model的token embedding大小
        sks_tokens = [config.sks_name]  # "<yufei>"
        # print("Vocabulary size before:", len(tokenizer))
      #  pdb.set_trace()
        prefix_tokens = [f'<token{i}>' for i in range(config.model_config["bridge"]["num_prefix_token"])]

        tokenizer.add_tokens(sks_tokens + prefix_tokens)
        # print("Vocabulary size after:", len(tokenizer))

        model.tokenizer = tokenizer
        model.lm.resize_token_embeddings(len(tokenizer))
        # 如果有必要，也调整lm_head
        if hasattr(model.lm, 'lm_head'):
            old_lm_head = model.lm.lm_head
            model.lm.lm_head = nn.Linear(
                old_lm_head.in_features,
                len(tokenizer),
                bias=False
            ).to(model.lm.device)
            model.lm.lm_head.weight.data[:old_lm_head.out_features] = old_lm_head.weight.data
        final_save_dir = train(model, config, tokenizer, sks_tokens, prefix_tokens)

        # 将模型设置为评估模式
        model.eval()

        # 运行测试
        print("Starting testing...")

        test_path = config.sks_name + "test.json"  # 测试文件路径
        test_results = test(model, tokenizer, test_path, device)
    elif args.mode == "test":
        # 仅测试模式
        checkpoint_path = args.checkpoint_path  # 这应该是final_model目录的路径

        # 1. 加载config
        # 1. 加载额外的训练信息(包含special tokens信息)
        extra_info = torch.load(os.path.join(checkpoint_path, 'training_info.bin'))
        sks_tokens = extra_info['sks_tokens']
        prefix_tokens = extra_info['prefix_tokens']

        # 2.加载config和tokenizer
        config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            use_fast=False
        )

        # 3. Initialize model (using original pretrained model path)
        config_path = "./Internvideo2_chat_8B_HD_finetune_REMOH"
        model = AutoModel.from_pretrained(
            config_path,  # Use original model path
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).cuda()

        # 4. Expand vocabulary
        # print("Vocabulary size before:", len(tokenizer))
        tokenizer.add_tokens(sks_tokens + prefix_tokens)
        # print("Vocabulary size after:", len(tokenizer))

        # 5. Adjust model embedding size
        model.tokenizer = tokenizer
        model.lm.resize_token_embeddings(len(tokenizer))
        if hasattr(model.lm, 'lm_head'):
            old_lm_head = model.lm.lm_head
            model.lm.lm_head = nn.Linear(
                old_lm_head.in_features,
                len(tokenizer),
                bias=False
            ).to(model.lm.device)

        # 6. Load trained model weights
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'pytorch_model.bin')))

        # 4. Set to evaluation mode
        model.eval()

        # 运行测试
        print("Starting testing...")
        test_path = config.sks_name + "test.json"
        test_results = test(model, tokenizer, test_path, device)